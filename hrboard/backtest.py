"""Fit and test the plate-appearance model.  python -m hrboard.backtest

Train: every 2024 and 2025 plate appearance.  Test: every 2026 plate appearance through yesterday (never seen in fit).
Skills are always as of the morning of the game (this season before that day + weighted prior seasons), park factors
are the three years ending the season before, and weather is what MLB recorded at first pitch.

Writes hrboard/model.json (coefficients, shrinkage, lineup tables) and data/hrboard/backtest/*.parquet.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from breakout.config import DATA
from . import features as F
from . import dataset as D

HERE = Path(__file__).resolve().parent
BT = DATA / "hrboard" / "backtest"
SEASONS = [2022, 2023, 2024, 2025, 2026]
TRAIN, TEST = [2024, 2025], [2026]


def build_matrix(d: pd.DataFrame, book: F.SkillBook, seasons) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = d[d["season"].isin(seasons)].reset_index(drop=True)
    q = rows[["batter", "pitcher", "season", "game_date", "stand", "p_throws", "game_pk", "is_home", "cls", "vs_sp",
              "bat_team", "fld_team", "at_bat_number", "opp_sp"]]
    s = D.skill_rows(book, q)
    sch = D.game_context(seasons); parks = D.park_table(seasons)
    s = D.attach_context(s, sch, parks)
    X = D.design(s)
    return s, X


def fit_shrinkage(d, book, seasons):
    """Grid for the HR shrinkage constants (the ones that matter most for this board). Scored by one-feature-each
    logistic log loss on home runs."""
    rows = d[d["season"].isin(seasons)]; rows = rows.sample(min(200_000, len(rows)), random_state=1)
    bq = rows[["batter", "season", "game_date"]].copy(); pq = rows[["pitcher", "season", "game_date"]].copy()
    bc = book._attach(bq, book.b_cur, book.b_tot, "batter", F.B_COUNTS)
    pc = book._attach(pq, book.p_cur, book.p_tot, "pitcher", F.P_COUNTS)
    L = F.attach_league(rows, book.lga)
    y = rows["hr"].to_numpy()
    res = []
    for kb in (15, 30, 50, 100, 200):
        for kbr in (10, 20, 40, 80):
            rb = (bc["hr"].to_numpy() + kb * L["hr"].to_numpy()) / (bc["pa"].to_numpy() + kb)
            rbr = (bc["brl"].to_numpy() + kbr * L["brl"].to_numpy()) / (bc["pa"].to_numpy() + kbr)
            X = np.c_[F.lg_logit(rb), F.lg_logit(rbr)]
            m = LogisticRegression(C=100, max_iter=300).fit(X, y)
            res.append(("bat", kb, kbr, log_loss(y, m.predict_proba(X)[:, 1])))
    best_b = min((r for r in res if r[0] == "bat"), key=lambda r: r[3])
    for kp in (200, 400, 800, 1500, 3000):
        for kpb in (150, 300, 600, 1200):
            rp = (pc["hr"].to_numpy() + kp * L["hr"].to_numpy()) / (pc["pa"].to_numpy() + kp)
            rpb = (pc["brl"].to_numpy() + kpb * L["brl"].to_numpy()) / (pc["pa"].to_numpy() + kpb)
            X = np.c_[F.lg_logit(rp), F.lg_logit(rpb)]
            m = LogisticRegression(C=100, max_iter=300).fit(X, y)
            res.append(("pit", kp, kpb, log_loss(y, m.predict_proba(X)[:, 1])))
    best_p = min((r for r in res if r[0] == "pit"), key=lambda r: r[3])
    return best_b, best_p, res


def fit_model(X: pd.DataFrame, y: np.ndarray, C: float = 1.0) -> LogisticRegression:
    m = LogisticRegression(C=C, max_iter=2000, tol=1e-6)
    m.fit(X.to_numpy(), y)
    return m


def fit_pa(d: pd.DataFrame, train: list[int], grid: bool = True, verbose: bool = True) -> dict:
    """Fit shrinkage (optional grid) and the multinomial PA model on `train` seasons. Returns the model dict."""
    book = F.SkillBook(d)
    if grid:
        best_b, best_p, g = fit_shrinkage(d, book, train)
        Kb = dict(hr=best_b[1], brl=best_b[2]); Kp = dict(hr=best_p[1], brl=best_p[2])
        if verbose: print("  shrinkage", best_b, best_p)
    else:
        Kb, Kp, g = {}, {}, []
    book = F.SkillBook(d, K_bat=Kb, K_pit=Kp)
    s_tr, X_tr = build_matrix(d, book, train)
    from . import model as M
    m = M.fit_mnl(X_tr.to_numpy(dtype="float64"), s_tr["cls"].to_numpy(), D.offsets(s_tr))
    if verbose: print("  mnl converged", m["converged"], round(m["nll"], 5))
    return dict(classes=m["classes"], columns=list(X_tr.columns), coef=m["coef"], intercept=m["intercept"],
                K_bat=book.K_bat, K_pit=book.K_pit, k_split_bat=book.ksb, k_split_pit=book.ksp, w_prev=list(F.W_PREV),
                lg=book.lg.reset_index().rename(columns={"index": "season"}).to_dict(orient="records"),
                shrink_grid=[list(r) for r in g], trained_on=list(train))


def pa_report(d: pd.DataFrame, mj: dict, test: list[int]) -> dict:
    from . import model as M
    book = F.SkillBook(d, K_bat=mj["K_bat"], K_pit=mj["K_pit"])
    s_te, X_te = build_matrix(d, book, test)
    order = ["out", "k", "bb", "s1", "xb", "hr"]
    P = M.predict(X_te, mj, D.offsets(s_te), order)
    y = s_te["cls"].to_numpy()
    base = np.exp(D.offsets(s_te, order))
    hi = order.index("hr")
    lab = sorted(order); ix = [order.index(c) for c in lab]
    return dict(pa_logloss=float(log_loss(y, P[:, ix], labels=lab)), pa_logloss_league=float(log_loss(y, base[:, ix], labels=lab)),
                hr_auc=float(roc_auc_score(y == "hr", P[:, hi])), hr_logloss=float(log_loss(y == "hr", P[:, hi])),
                hr_logloss_league=float(log_loss(y == "hr", base[:, hi])), n=int(len(y)))


def main():
    d = D.load_pa(SEASONS)
    mj = fit_pa(d, TRAIN)
    print(pa_report(d, mj, TEST))
    coef = pd.DataFrame(mj["coef"], index=mj["classes"], columns=mj["columns"])
    print(coef.round(3).T.to_string())


if __name__ == "__main__":
    main()

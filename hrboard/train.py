"""Fit everything, test it honestly, and write the production model.   python -m hrboard.train

  1. PA model on 2024            -> forecast every 2025 game  (calibration set)
  2. PA model on 2024-25          -> forecast every 2026 game  (test set; calibrated with the maps from step 1)
  3. PA model on 2024-26          -> production; calibration maps refit on the 2025 + 2026 out-of-sample forecasts

Writes hrboard/model.json (production), data/hrboard/backtest/report.json and the 2026 test frames the page's
"how it has done" panel reads.
"""
from __future__ import annotations
import json, time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss

from . import dataset as D, game as G
from .backtest import fit_pa, pa_report, BT, HERE, SEASONS
from .backtest_games import run_games


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4); return np.log(p / (1 - p))


def fit_platt(p, y) -> list[float]:
    m = LogisticRegression(C=1e4, max_iter=500).fit(logit(p)[:, None], np.asarray(y))
    return [float(m.intercept_[0]), float(m.coef_[0, 0])]


def platt(p, ab):
    return 1 / (1 + np.exp(-(ab[0] + ab[1] * logit(p))))


def k_long(spk: pd.DataFrame, ns=range(3, 11)):
    p = np.concatenate([spk[f"p_ge{n}"].to_numpy() for n in ns]); y = np.concatenate([(spk["K"] >= n).to_numpy() for n in ns])
    return p, y.astype(int)


def fit_runs(tg: pd.DataFrame) -> dict:
    """Runs = a + b * model rate; NB dispersion r by maximum likelihood."""
    from scipy.stats import nbinom
    b, a = np.polyfit(tg["rpg"], tg["runs"], 1)
    mu = a + b * tg["rpg"].to_numpy(); y = tg["runs"].to_numpy().astype(int)
    def nll(lr):
        r = np.exp(lr[0]); return -nbinom.logpmf(y, r, r / (r + mu)).sum()
    r = float(np.exp(minimize(nll, [np.log(4.0)]).x[0]))
    return dict(a=float(a), b=float(b), r=r)


def game_frame(tg: pd.DataFrame, runs: dict) -> pd.DataFrame:
    h = tg[tg["is_home"] == 1].set_index("game_pk"); a = tg[tg["is_home"] == 0].set_index("game_pk")
    g = h[["rpg", "runs", "game_date"]].join(a[["rpg", "runs"]], rsuffix="_away", how="inner")
    g["mu_h"] = runs["a"] + runs["b"] * g["rpg"]; g["mu_a"] = runs["a"] + runs["b"] * g["rpg_away"]
    g["p_home"] = [G.win_prob(x, z, runs["r"])["p_home"] for x, z in zip(g["mu_h"], g["mu_a"])]
    g["y_home"] = (g["runs"] > g["runs_away"]).astype(int)
    return g.reset_index()


def score(p, y) -> dict:
    y = np.asarray(y).astype(int); p = np.asarray(p, dtype=float)
    return dict(n=int(len(y)), base=float(y.mean()), mean_p=float(p.mean()), logloss=float(log_loss(y, p)),
                logloss_const=float(log_loss(y, np.full(len(y), y.mean()))), auc=float(roc_auc_score(y, p)), brier=float(brier_score_loss(y, p)))


def calib_table(p, y, edges) -> list[dict]:
    df = pd.DataFrame(dict(p=p, y=y)); df["bin"] = pd.cut(df["p"], edges)
    t = df.groupby("bin", observed=True).agg(n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean")).reset_index()
    t["bin"] = t["bin"].astype(str)
    return t.round(4).to_dict(orient="records")


def main():
    t0 = time.time()
    d = D.load_pa(SEASONS)
    print("step 1: fit 2024, forecast 2025")
    m24 = fit_pa(d, [2024])
    g25 = run_games(d, m24, 2025, [2024])
    print("step 2: fit 2024-25, forecast 2026")
    m2425 = fit_pa(d, [2024, 2025])
    pa26 = pa_report(d, m2425, [2026])
    g26 = run_games(d, m2425, 2026, [2024, 2025])
    st25, st26 = g25["st"], g26["st"]
    cal = {k: fit_platt(st25[f"p_{k}"], st25[f"y_{k}"]) for k in ("hr", "hit", "tb2")}
    pk25, yk25 = k_long(g25["spk"]); cal["k"] = fit_platt(pk25, yk25)
    runs25 = fit_runs(g25["tg"]); gm25 = game_frame(g25["tg"], runs25); cal["win"] = fit_platt(gm25["p_home"], gm25["y_home"])
    rep = dict(pa_2026=pa26, calib_from_2025=cal, runs_from_2025=runs25)
    for k in ("hr", "hit", "tb2"):
        st26[f"c_{k}"] = platt(st26[f"p_{k}"], cal[k])
        rep[f"{k}_raw"] = score(st26[f"p_{k}"], st26[f"y_{k}"]); rep[k] = score(st26[f"c_{k}"], st26[f"y_{k}"])
    rep["hr_naive"] = score(st26["p_hr_naive"], st26["y_hr"])
    rep["hr_calib"] = calib_table(st26["c_hr"], st26["y_hr"], [0, .05, .07, .09, .11, .13, .15, .17, .19, .22, .26, 1])
    rep["hit_calib"] = calib_table(st26["c_hit"], st26["y_hit"], [0, .45, .5, .55, .6, .65, .7, .75, 1])
    rep["tb2_calib"] = calib_table(st26["c_tb2"], st26["y_tb2"], [0, .2, .25, .3, .35, .4, .45, .5, 1])
    st26["rank"] = st26.groupby("game_date")["c_hr"].rank(ascending=False, method="first")
    st26["rank_naive"] = st26.groupby("game_date")["p_hr_naive"].rank(ascending=False, method="first")
    rep["top"] = {}
    for n in (1, 3, 5, 10, 25):
        a = st26[st26["rank"] <= n]; b = st26[st26["rank_naive"] <= n]
        rep["top"][n] = dict(hit_rate=float(a["y_hr"].mean()), predicted=float(a["c_hr"].mean()), naive_hit_rate=float(b["y_hr"].mean()),
                             days=int(a["game_date"].nunique()), days_with_a_hr=float(a.groupby("game_date")["y_hr"].max().mean()),
                             hrs_per_day=float(a.groupby("game_date")["HR"].sum().mean()))
    spk = g26["spk"]
    pk, yk = k_long(spk); pkc = platt(pk, cal["k"])
    rep["k_ladder_raw"] = score(pk, yk); rep["k_ladder"] = score(pkc, yk)
    rep["k_calib"] = calib_table(pkc, yk, [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
    rep["sp_k"] = dict(n=len(spk), mae_model=float((spk["e_k"] - spk["K"]).abs().mean()), mae_naive=float((spk["k_prior"] - spk["K"]).abs().mean()),
                       mean_pred=float(spk["e_k"].mean()), mean_actual=float(spk["K"].mean()),
                       corr_model=float(np.corrcoef(spk["e_k"], spk["K"])[0, 1]), corr_naive=float(np.corrcoef(spk["k_prior"], spk["K"])[0, 1]))
    gm26 = game_frame(g26["tg"], runs25); gm26["c_home"] = platt(gm26["p_home"], cal["win"])
    rep["win_raw"] = score(gm26["p_home"], gm26["y_home"]); rep["win"] = score(gm26["c_home"], gm26["y_home"])
    rep["win_home_only"] = score(np.full(len(gm26), gm25["y_home"].mean()), gm26["y_home"])
    rep["win_calib"] = calib_table(gm26["c_home"], gm26["y_home"], [0, .35, .4, .45, .5, .55, .6, .65, 1])
    tot = gm26["runs"] + gm26["runs_away"]; tp = gm26["mu_h"] + gm26["mu_a"]
    rep["totals"] = dict(mae=float((tot - tp).abs().mean()), mae_const=float((tot - tot.mean()).abs().mean()), corr=float(np.corrcoef(tot, tp)[0, 1]),
                         mean_pred=float(tp.mean()), mean_actual=float(tot.mean()))
    for k in ("hr", "hr_naive", "hit", "tb2", "k_ladder", "win", "win_home_only"):
        print(k, {a: round(b, 4) for a, b in rep[k].items()})
    print("top", json.dumps(rep["top"], indent=0))
    print("sp_k", rep["sp_k"]); print("totals", rep["totals"])
    print("step 3: production fit on 2024-26")
    mp = fit_pa(d, [2024, 2025, 2026])
    cal_p = {}
    both = pd.concat([st25, st26], ignore_index=True)
    for k in ("hr", "hit", "tb2"):
        cal_p[k] = fit_platt(both[f"p_{k}"], both[f"y_{k}"])
    pk_b, yk_b = k_long(pd.concat([g25["spk"], spk], ignore_index=True)); cal_p["k"] = fit_platt(pk_b, yk_b)
    tg_b = pd.concat([g25["tg"], g26["tg"]], ignore_index=True)
    runs_p = fit_runs(tg_b); gm_b = game_frame(tg_b, runs_p); cal_p["win"] = fit_platt(gm_b["p_home"], gm_b["y_home"])
    tb_all = dict(q=np.mean([g25["q"], g26["q"]], axis=0).tolist(), curve=g26["curve"], resid=g26["resid"])
    mp.update(calib=cal_p, runs=runs_p, tables=tb_all, backtest=dict(test_season=2026, fit_on=[2024, 2025], calibrated_on=[2025]))
    (HERE / "model.json").write_text(json.dumps(mp, indent=1, default=float))
    BT.mkdir(parents=True, exist_ok=True)
    (BT / "report.json").write_text(json.dumps(rep, indent=1, default=float))
    keep = ["game_date", "game_pk", "batter", "bat_team", "slot", "c_hr", "c_hit", "c_tb2", "HR", "H", "TB", "PA", "rank", "y_hr"]
    st26[keep].to_parquet(BT / "games_2026.parquet", index=False)
    spk.assign(c_ge5=platt(spk["p_ge5"], cal["k"])).to_parquet(BT / "spk_2026.parquet", index=False)
    gm26.to_parquet(BT / "wins_2026.parquet", index=False)
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

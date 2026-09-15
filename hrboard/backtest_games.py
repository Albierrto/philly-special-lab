"""Game-level test on 2026 (model fitted on 2024-25): home runs, hits, total bases, starter strikeouts, team runs and
win probability, each scored as a pre-game forecast with the real lineup and starter.  python -m hrboard.backtest_games
"""
from __future__ import annotations
import json, time
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss

from . import features as F, dataset as D, game as G, model as M
from .backtest import BT, HERE

SEASONS = [2022, 2023, 2024, 2025, 2026]


def side_rows(book, bpbook, st, hands_b, hands_p, sch, parks, bshare):
    """Design matrices for each starter against the opposing starter and against the opposing bullpen."""
    st = st.merge(hands_b, on="batter", how="left").merge(hands_p.rename(columns={"pitcher": "opp_sp"}), on="opp_sp", how="left")
    st["bats"] = st["bats"].fillna("R"); st["throws"] = st["throws"].fillna("R")
    # vs starter
    q = pd.DataFrame(dict(batter=st["batter"], pitcher=st["opp_sp"], season=st["season"], game_date=st["game_date"],
                          p_throws=st["throws"], stand=G.stand_vs(st["bats"], st["throws"]), game_pk=st["game_pk"],
                          is_home=st["is_home"]), index=st.index)
    s = D.attach_context(D.skill_rows(book, q), sch, parks)
    X_sp = D.design(s)
    # vs bullpen: a switch hitter mostly sees right-handers, so he bats left
    bstand = np.where(st["bats"] == "S", "L", st["bats"])
    qb = pd.DataFrame(dict(batter=st["batter"], pitcher=st["fld_team"], season=st["season"], game_date=st["game_date"],
                           p_throws="R", stand=bstand, game_pk=st["game_pk"], is_home=st["is_home"]), index=st.index)
    b = book.batter_features(qb[["batter", "season", "game_date", "p_throws"]])
    p = bpbook.pitcher_features(qb[["pitcher", "season", "game_date", "stand"]])
    lgb = F.attach_league(qb, book.lga).add_prefix("lg_")
    sb = D.attach_context(pd.concat([qb, b, p, lgb], axis=1), sch, parks)
    X_bp = D.design(sb)
    share = qb[["pitcher", "stand", "season", "game_date"]].rename(columns={"pitcher": "fld_team"}).sort_values("game_date")
    share["_i"] = np.arange(len(share))
    bs = bshare.sort_values("game_date")
    m = pd.merge_asof(share.reset_index(), bs, on="game_date", by=["fld_team", "stand", "season"], allow_exact_matches=False)
    m = m.set_index("index").reindex(st.index)
    dflt = np.where(qb["stand"].to_numpy() == "R", 0.62, 0.30)
    X_bp["platoon"] = np.where(m["same_share"].isna(), dflt, m["same_share"].to_numpy(dtype=float))
    return st, X_sp, X_bp, D.offsets(s), D.offsets(sb)


def calib(p, y, bins):
    df = pd.DataFrame(dict(p=p, y=y)); df["bin"] = pd.cut(df["p"], bins)
    return df.groupby("bin", observed=True).agg(n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean")).round(4)


def tables(d: pd.DataFrame, train: list[int]) -> dict:
    """Lineup reach, starter share and batters-faced spread, from the training seasons only."""
    work = G.sp_workload(d)
    tr = d[d["season"].isin(train)]
    return dict(q=G.pa_reach_table(G.starters(tr)), curve=G.vs_sp_curve(tr, work),
                resid=G.bf_residuals(work[work["season"].isin(train) & (work["n_before"] >= 3)]), work=work)


def books(d: pd.DataFrame, mj: dict):
    book = F.SkillBook(d, K_bat=mj["K_bat"], K_pit=mj["K_pit"])
    bp = G.bullpen_frame(d)
    bpbook = F.SkillBook(bp, K_pit=dict(mj["K_pit"], hr=400, brl=300))
    return book, bpbook, G.bullpen_same_share(bp), G.batter_hands(d), G.pitcher_hands(d)


def run_games(d: pd.DataFrame, mj: dict, test: int, train: list[int]) -> dict:
    """Pre-game forecasts for every starter, starting pitcher and team-game of `test`, with the actual results."""
    t0 = time.time()
    tb = tables(d, train); q, curve, resid, work = tb["q"], tb["curve"], tb["resid"], tb["work"]
    book, bpbook, bshare, hands_b, hands_p = books(d, mj)
    sch = D.game_context([test]); parks = D.park_table([test])
    st = G.starters(d[d["season"] == test]).reset_index(drop=True)
    st = st.merge(work[["pitcher", "game_pk", "bf_exp", "bf"]].rename(columns={"pitcher": "opp_sp"}), on=["opp_sp", "game_pk"], how="left")
    st["bf_exp"] = st["bf_exp"].fillna(G.LG_BF)
    st, X_sp, X_bp, O_sp, O_bp = side_rows(book, bpbook, st, hands_b, hands_p, sch, parks, bshare)
    P_sp, P_bp = M.predict(X_sp, mj, O_sp), M.predict(X_bp, mj, O_bp)
    r = G.batter_game(P_sp, P_bp, st["slot"], st["is_home"], st["bf_exp"], q, curve)
    for k, v in r.items(): st[k] = v
    st["y_hr"] = (st["HR"] > 0).astype(int); st["y_hit"] = (st["H"] > 0).astype(int); st["y_tb2"] = (st["TB"] >= 2).astype(int)
    bc = book._attach(st[["batter", "season", "game_date"]], book.b_cur, book.b_tot, "batter", F.B_COUNTS)
    lg = F.attach_league(st, book.lga)
    naive = (bc["hr"].to_numpy() + 50 * lg["hr"].to_numpy()) / (bc["pa"].to_numpy() + 50)
    st["p_hr_naive"] = 1 - (1 - naive) ** 4.1
    # starters' strikeouts
    sp_rows = []
    for (gpk, team), g in st.groupby(["game_pk", "bat_team"]):
        g = g.sort_values("slot")
        if len(g) < 9: continue
        res = G.pitcher_ks(P_sp[g.index, G.CLS.index("k")], float(g["bf_exp"].iat[0]), resid)
        sp_rows.append(dict(game_pk=gpk, pitcher=g["opp_sp"].iat[0], game_date=g["game_date"].iat[0], e_k=res["e_k"], e_bf=res["e_bf"],
                            **{f"p_ge{n}": res["p_ge"][n] for n in range(2, 13)}, bf_exp=float(g["bf_exp"].iat[0])))
    spk = pd.DataFrame(sp_rows)
    kact = d[(d["season"] == test) & (d["vs_sp"] == 1)].groupby(["game_pk", "opp_sp"]).agg(K=("k", "sum"), BF=("pa", "sum")).reset_index()
    spk = spk.merge(kact, left_on=["game_pk", "pitcher"], right_on=["game_pk", "opp_sp"], how="inner").sort_values("game_date")
    spk["k_prior"] = spk.groupby("pitcher")["K"].transform(lambda x: (x.shift(1).expanding().sum().fillna(0) + 25) / (x.shift(1).expanding().count().fillna(0) + 5))
    # team-games
    games = []
    for (gpk, team), g in st.groupby(["game_pk", "bat_team"]):
        if len(g) < 9: continue
        g = g.sort_values("slot"); idx = g.index
        share_sp = float(np.clip(g["bf_exp"].iat[0] / 38.5, 0.2, 0.95))
        rates = (g["e_pa"].to_numpy()[:, None] * (share_sp * P_sp[idx] + (1 - share_sp) * P_bp[idx])).sum(0) / g["e_pa"].sum()
        games.append(dict(game_pk=gpk, team=team, is_home=int(g["is_home"].iat[0]), game_date=g["game_date"].iat[0], rpg=G.team_runs_rate(rates)))
    tg = pd.DataFrame(games)
    s2 = sch.set_index("game_pk")
    tg["runs"] = [s2.at[p, "home_runs"] if h else s2.at[p, "away_runs"] for p, h in zip(tg["game_pk"], tg["is_home"])]
    tg = tg.dropna(subset=["runs"])
    print(f"  games {test}: {len(st):,} batter-games, {len(spk):,} starts, {len(tg):,} team-games ({time.time()-t0:.0f}s)")
    return dict(st=st, spk=spk, tg=tg, q=q, curve=curve, resid=resid)


def main():
    from .backtest import fit_pa
    d = D.load_pa(SEASONS)
    mj = M.load()
    out = run_games(d, mj, 2026, [2024, 2025])
    st = out["st"]
    for name, p, y in [("HR", st["p_hr"], st["y_hr"]), ("HR naive", st["p_hr_naive"], st["y_hr"])]:
        print(name, round(log_loss(y, p), 4), round(roc_auc_score(y, p), 4))


if __name__ == "__main__":
    main()

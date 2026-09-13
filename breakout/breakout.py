"""Breakout detection: (a) an interpretable, expert-inspired skills score and (b) a trained next-season model.

Skills score (z-scores within a season, hitters with >= MIN_PA_SKILLS):
    quality_of_contact : barrel%, hard-hit%, EV50 (avg_best_speed), xwOBAcon          (Sarris / Savant "red ink")
    bat_speed          : avg bat speed, fast-swing rate, ideal attack-angle rate       (Sarris; "ideal fast swing")
    approach           : K% (-), BB%, chase (-), whiff (-), swing/take run value       (PLV "Decision Value")
    speed              : sprint speed  (SB = 3 pts in this league)
    luck               : xwOBA - wOBA  (+ = unlucky -> regression UP)                   (Scott White "should have hit")
    trend              : yoy change in barrel%, hard-hit%, K% (-), bat speed             (Athlon "Barrel Rate Test")
    age                : younger = more room to grow
    market_gap         : how much better the skills say he is than where the market ranked him

Model: HistGradientBoosting on season-t Statcast/age/production -> season-(t+1) points per PA, trained on
2021->2022 ... 2025->2026 pairs, evaluated leave-one-season-out, then applied to 2026 to project 2027.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from . import config as C

QOC = ["barrel_batted_rate", "hard_hit_percent", "avg_best_speed", "xwobacon"]
BAT = ["avg_swing_speed", "fast_swing_rate", "ideal_angle_rate"]
APPR_POS = ["bb_percent", "swing_take_run_value", "iz_contact_percent"]
APPR_NEG = ["k_percent", "oz_swing_percent", "whiff_percent"]
MODEL_FEATURES = ["xwoba", "xiso", "xba", "barrel_batted_rate", "hard_hit_percent", "avg_best_speed", "xwobacon",
                  "k_percent", "bb_percent", "oz_swing_percent", "whiff_percent", "iz_contact_percent",
                  "swing_take_run_value", "sprint_speed", "sb_per600", "pull_percent", "flyballs_percent",
                  "avg_swing_speed", "fast_swing_rate", "ideal_angle_rate", "age", "pts_pa", "PA", "woba", "wobadiff",
                  "hr_per600", "bb_per600"]


def _z(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=0)


def add_rates(ps: pd.DataFrame) -> pd.DataFrame:
    d = ps.copy()
    pa = d["PA"].replace(0, np.nan)
    d["sb_per600"] = d["SB"] / pa * 600
    d["hr_per600"] = d["HR"] / pa * 600
    d["bb_per600"] = d["BB"] / pa * 600
    d["xwoba_minus_woba"] = d["xwoba"] - d["woba"]
    return d


def skills_scores(ps: pd.DataFrame, season: int) -> pd.DataFrame:
    """Interpretable breakout score for every hitter with enough PA in `season`, using season-1 for trends."""
    d = add_rates(ps)
    cur = d[(d["season"] == season) & (d["PA"] >= C.MIN_PA_SKILLS)].copy()
    prev = d[(d["season"] == season - 1)].set_index("mlbam_id")
    z = pd.DataFrame(index=cur.index)
    z["quality_of_contact"] = pd.concat([_z(cur[c]) for c in QOC], axis=1).mean(axis=1)
    z["bat_speed"] = pd.concat([_z(cur[c]) for c in BAT], axis=1).mean(axis=1)
    z["approach"] = pd.concat([_z(cur[c]) for c in APPR_POS] + [-_z(cur[c]) for c in APPR_NEG], axis=1).mean(axis=1)
    z["speed"] = _z(cur["sprint_speed"])
    z["luck"] = _z(cur["xwoba_minus_woba"])
    # trend vs previous season (only where he had >= 150 PA the year before)
    pv = prev.reindex(cur["mlbam_id"])
    ok = (pv["PA"] >= 150).values
    dbar = (cur["barrel_batted_rate"].values - pv["barrel_batted_rate"].values)
    dhh = (cur["hard_hit_percent"].values - pv["hard_hit_percent"].values)
    dk = -(cur["k_percent"].values - pv["k_percent"].values)
    dbs = (cur["avg_swing_speed"].values - pv["avg_swing_speed"].values)
    tr = pd.DataFrame({"dbar": dbar, "dhh": dhh, "dk": dk, "dbs": dbs}, index=cur.index)
    tr[~ok] = np.nan
    z["trend"] = pd.concat([_z(tr[c]) for c in tr.columns], axis=1).mean(axis=1).fillna(0)
    z["age_factor"] = ((27.5 - cur["age"]) * 0.2).clip(-1.2, 1.2)
    z["skills_score"] = (1.0 * z["quality_of_contact"].fillna(0) + 0.75 * z["bat_speed"].fillna(0)
                         + 0.75 * z["approach"].fillna(0) + 0.5 * z["speed"].fillna(0) + 0.5 * z["luck"].fillna(0)
                         + 0.75 * z["trend"] + z["age_factor"])
    out = pd.concat([cur, z], axis=1)
    # market: where did the market/results put him? (final rank this season, ADP this season)
    out["skills_rank"] = out["skills_score"].rank(ascending=False).astype(int)
    out["market_rank"] = out[["final_hitter_rank", "adp_hitter_rank"]].min(axis=1)
    out["market_gap"] = out["market_rank"] - out["skills_rank"]      # + = skills say better than market
    out["yoy_pts_change"] = out["pts"].values - pv["pts"].values
    out["prev_pa"] = pv["PA"].values
    out["prev_pts"] = pv["pts"].values
    out["prev_k_percent"] = pv["k_percent"].values
    out["prev_barrel"] = pv["barrel_batted_rate"].values
    out["prev_swing_speed"] = pv["avg_swing_speed"].values
    return out.sort_values("skills_score", ascending=False)


# ----------------------------------------------------------------------------- trained next-season model
def _pairs(ps: pd.DataFrame, min_pa=200, min_pa_next=200) -> pd.DataFrame:
    d = add_rates(ps)
    d = d[d["PA"] >= min_pa]
    nxt = d[["mlbam_id", "season", "pts_pa", "pts", "PA", "pts_g", "final_hitter_rank"]].copy()
    nxt["season"] -= 1
    nxt = nxt.rename(columns={"pts_pa": "next_pts_pa", "pts": "next_pts", "PA": "next_PA", "pts_g": "next_pts_g",
                              "final_hitter_rank": "next_rank"})
    m = d.merge(nxt, on=["mlbam_id", "season"], how="inner")
    return m[m["next_PA"] >= min_pa_next]


def fit_model(ps: pd.DataFrame, target="next_pts_pa"):
    pr = _pairs(ps)
    X, y = pr[MODEL_FEATURES], pr[target]
    model = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=400, min_samples_leaf=20,
                                          l2_regularization=1.0, random_state=7)
    model.fit(X, y)
    return model, pr


def cross_validate(ps: pd.DataFrame, target="next_pts_pa") -> pd.DataFrame:
    """Leave-one-season-out: train on other season pairs, predict the held-out pair. Compare vs naive carry-over."""
    pr = _pairs(ps)
    rows = []
    for s in sorted(pr["season"].unique()):
        tr, te = pr[pr["season"] != s], pr[pr["season"] == s]
        m = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=400, min_samples_leaf=20,
                                          l2_regularization=1.0, random_state=7).fit(tr[MODEL_FEATURES], tr[target])
        pred = m.predict(te[MODEL_FEATURES])
        r_model = np.corrcoef(pred, te[target])[0, 1]
        r_naive = np.corrcoef(te["pts_pa"], te[target])[0, 1]
        r_xwoba = np.corrcoef(te["xwoba"].fillna(te["xwoba"].mean()), te[target])[0, 1]
        rows.append(dict(train_on=f"{s}->{s+1} held out", n=len(te), r_model=round(r_model, 3),
                         r_naive_last_year=round(r_naive, 3), r_xwoba_only=round(r_xwoba, 3),
                         mae_model=round(float(np.mean(np.abs(pred - te[target]))), 3),
                         mae_naive=round(float(np.mean(np.abs(te["pts_pa"] - te[target]))), 3)))
    return pd.DataFrame(rows)


def project_next(ps: pd.DataFrame, season: int) -> pd.DataFrame:
    """Project season+1 points per PA (and total points) for every hitter with >= 150 PA in `season`."""
    model, _ = fit_model(ps)
    d = add_rates(ps)
    cur = d[(d["season"] == season) & (d["PA"] >= 150)].copy()
    cur["proj_pts_pa"] = model.predict(cur[MODEL_FEATURES])
    hist = d[d["season"].isin([season - 1, season])].groupby("mlbam_id")["PA"].mean()
    cur["proj_PA"] = (0.6 * cur["PA"] + 0.4 * cur["mlbam_id"].map(hist)).clip(250, 640).round(0)
    # health/role upside: if he only got a partial season, note the full-time pace
    cur["proj_pts"] = (cur["proj_pts_pa"] * cur["proj_PA"]).round(0)
    cur["proj_pts_600pa"] = (cur["proj_pts_pa"] * 600).round(0)
    cur["proj_rank"] = cur["proj_pts"].rank(ascending=False).astype(int)
    cur["proj_rank_600pa"] = cur["proj_pts_600pa"].rank(ascending=False).astype(int)
    return cur


def breakout_candidates(ps: pd.DataFrame, season: int) -> pd.DataFrame:
    """Combine skills score + model projection + market rank into a ranked candidate list for season+1."""
    sk = skills_scores(ps, season)
    pj = project_next(ps, season)[["mlbam_id", "proj_pts_pa", "proj_PA", "proj_pts", "proj_pts_600pa", "proj_rank", "proj_rank_600pa"]]
    m = sk.merge(pj, on="mlbam_id", how="left")
    # where the market has him: this season's preseason ADP rank if drafted, else his final rank (what he showed)
    # backtested 2022->23 ... 2025->26: skills_score + 0.02*(market rank - projected rank) picked next-year ADP
    # beaters at ~1.7x the base rate (see cross_validate / verify.py); heavier market weights did worse.
    m["model_gap"] = m["market_rank"] - m["proj_rank"]
    m["breakout_score"] = m["skills_score"].fillna(0) + 0.02 * m["model_gap"].clip(-150, 150).fillna(0)
    m["pool"] = np.where(m["PA"] >= 400, "full-time", "playing-time dependent")
    m["tag"] = m.apply(_tag, axis=1)
    return m.sort_values("breakout_score", ascending=False)


def _tag(r) -> str:
    tags = []
    if r.get("luck", 0) >= 0.8: tags.append("unlucky (xwOBA >> wOBA)")
    if r.get("bat_speed", 0) >= 0.8: tags.append("elite bat speed")
    if r.get("quality_of_contact", 0) >= 0.8: tags.append("elite contact quality")
    if r.get("trend", 0) >= 0.8: tags.append("skills trending up")
    if r.get("approach", 0) >= 0.8: tags.append("plus approach")
    if r.get("speed", 0) >= 0.8: tags.append("plus speed")
    if r.get("PA", 600) < 450: tags.append("needs playing time")
    if r.get("age", 30) <= 25: tags.append("young")
    return "; ".join(tags)


def retro_breakouts(scored: pd.DataFrame, season: int, min_jump=100) -> pd.DataFrame:
    """Who actually broke out in `season`: career-best by a wide margin, top-75 finish, beat the ADP curve."""
    d = scored.copy()
    prev_best = (d[d["season"] < season].groupby("mlbam_id")["pts"].max()).rename("prev_best_pts")
    cur = d[d["season"] == season].merge(prev_best, on="mlbam_id", how="left")
    cur["prev_best_pts"] = cur["prev_best_pts"].fillna(0)
    cur["jump"] = cur["pts"] - cur["prev_best_pts"]
    out = cur[(cur["jump"] >= min_jump) & (cur["final_hitter_rank"] <= C.BIG_RELEVANT_RANK) & (cur["pts_over_exp"] >= 100)]
    return out.sort_values("jump", ascending=False)

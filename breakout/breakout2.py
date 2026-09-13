"""Breakout Index v2 and projections v2.

The v1 mistake: "skills look better than the market" flagged aging stars (Trout) because it never asked whether next
season would be *new territory for that player*. v2 defines a breakout relative to the player's own career:

    breakout(t+1) = pts/PA in t+1 exceeds his best prior 200+ PA season by >= JUMP (0.08 pts/PA ~ 50 pts per 600)
                    AND he gets >= 350 PA AND finishes inside the top-90 hitters in this league's points.

A HistGradientBoosting classifier learns P(breakout) from season-t Statcast skills, the gap between what he showed
and what he deserved (xLP), his career track (best rate, seasons, age), durability (IL days), and pedigree.
Trained on 2021->22 ... 2025->26 pairs, evaluated leave-one-season-out.

Projection v2 = v1 model rate  +  empirical aging step  ->  x  durability-aware PA estimate.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import DATA
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

from . import config as C
from .breakout import add_rates, MODEL_FEATURES
from .formulas import aging_curve, age_adjustment

JUMP = 0.08
BI_FEATURES = ["xwoba", "xiso", "xba", "barrel_batted_rate", "hard_hit_percent", "avg_best_speed", "xwobacon",
               "k_percent", "bb_percent", "oz_swing_percent", "whiff_percent", "iz_contact_percent", "swing_take_run_value",
               "sprint_speed", "sb_per600", "avg_swing_speed", "fast_swing_rate", "ideal_angle_rate", "age", "pts_pa", "PA",
               "xLP_pa", "luck_pa", "career_best_pa", "gap_to_best", "seasons_200", "career_PA", "il_days", "il_days_3yr",
               "yoy_k_change", "yoy_barrel_change", "yoy_bat_speed_change"]


def career_context(d: pd.DataFrame) -> pd.DataFrame:
    """Career-best pts/PA (200+ PA seasons) through season t, seasons played, career PA, yoy skill changes."""
    d = d.sort_values(["mlbam_id", "season"]).copy()
    d["_rate200"] = np.where(d["PA"] >= 200, d["pts_pa"], np.nan)
    g = d.groupby("mlbam_id")
    d["career_best_pa"] = g["_rate200"].cummax()
    d["seasons_200"] = g["_rate200"].transform(lambda s: s.notna().cumsum())
    d["career_PA"] = g["PA"].cumsum()
    d["gap_to_best"] = d["pts_pa"] - d["career_best_pa"]
    for col, new in [("k_percent", "yoy_k_change"), ("barrel_batted_rate", "yoy_barrel_change"), ("avg_swing_speed", "yoy_bat_speed_change")]:
        prev = g[col].shift(1); prev_pa = g["PA"].shift(1)
        d[new] = np.where(prev_pa >= 150, d[col] - prev, np.nan)
    return d.drop(columns=["_rate200"])


def _pairs(d: pd.DataFrame, min_pa=150) -> pd.DataFrame:
    cur = d[d["PA"] >= min_pa].copy()
    nxt = d[["mlbam_id", "season", "pts_pa", "PA", "final_hitter_rank", "pts"]].copy(); nxt["season"] -= 1
    nxt = nxt.rename(columns={"pts_pa": "next_pts_pa", "PA": "next_PA", "final_hitter_rank": "next_rank", "pts": "next_pts"})
    m = cur.merge(nxt, on=["mlbam_id", "season"], how="inner")
    base = m["career_best_pa"].fillna(m["pts_pa"])
    base = np.where(m["seasons_200"] == 0, np.minimum(m["pts_pa"], 0.35), base)   # rookies: their small sample or a low prior
    m["breakout_next"] = ((m["next_pts_pa"] >= base + JUMP) & (m["next_PA"] >= 350) & (m["next_rank"] <= 90)).astype(int)
    return m


def prepare(ps_x: pd.DataFrame, prospects: pd.DataFrame | None = None, injuries: pd.DataFrame | None = None) -> pd.DataFrame:
    d = add_rates(ps_x)
    d = career_context(d)
    if injuries is not None:
        d = d.merge(injuries[["mlbam_id", "season", "il_days", "il_stints", "il_days_3yr", "il_60", "il_reasons"] + (["il_days_w"] if "il_days_w" in injuries.columns else [])], on=["mlbam_id", "season"], how="left")
        if "il_days_w" not in d.columns: d["il_days_w"] = d["il_days_3yr"]
        for c in ("il_days", "il_stints", "il_days_3yr", "il_60", "il_days_w"):
            d[c] = d[c].fillna(0)
    else:
        d["il_days"] = 0; d["il_days_3yr"] = 0; d["il_stints"] = 0; d["il_60"] = 0; d["il_days_w"] = 0
    # playing-time availability (game logs): PA pace per 162 while on the roster, late call-ups, games share
    ap = DATA / "availability.parquet"
    if ap.exists():
        av = pd.read_parquet(ap)[["mlbam_id", "season", "first_game", "last_game", "games", "window_games", "avail_games", "il_games_in_window", "pa_pace_162", "g_share", "days_late", "late_callup"]]
        d = d.merge(av, on=["mlbam_id", "season"], how="left")
    else:
        for c in ("first_game", "last_game", "pa_pace_162", "g_share", "days_late", "late_callup", "avail_games", "window_games", "il_games_in_window", "games"): d[c] = np.nan
    if prospects is not None and "pedigree" in prospects:
        d = d.merge(prospects[["mlbam_id", "pedigree", "pipeline_rank", "prospect_status", "PAS"]], on="mlbam_id", how="left")
        d["pedigree"] = d["pedigree"].fillna(0.15)
    else:
        d["pedigree"] = 0.15
    return d


def _clf():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=25,
                                          l2_regularization=1.0, random_state=11)


def cross_validate_bi(d: pd.DataFrame) -> pd.DataFrame:
    pr = _pairs(d)
    rows = []
    for s in sorted(pr["season"].unique()):
        tr, te = pr[pr["season"] != s], pr[pr["season"] == s]
        m = _clf().fit(tr[BI_FEATURES], tr["breakout_next"])
        p = m.predict_proba(te[BI_FEATURES])[:, 1]
        auc = roc_auc_score(te["breakout_next"], p) if te["breakout_next"].nunique() > 1 else np.nan
        top = te.assign(p=p).sort_values("p", ascending=False).head(40)
        rows.append(dict(held_out=f"{s}->{s+1}", n=len(te), breakouts=int(te["breakout_next"].sum()),
                         base_rate=round(te["breakout_next"].mean(), 3), auc=round(auc, 3),
                         top40_hit_rate=round(top["breakout_next"].mean(), 3),
                         top40_avg_next_pts=round(top["next_pts"].mean(), 0)))
    return pd.DataFrame(rows)


def breakout_index(d: pd.DataFrame, season: int) -> pd.DataFrame:
    pr = _pairs(d)
    m = _clf().fit(pr[BI_FEATURES], pr["breakout_next"])
    cur = d[(d["season"] == season) & (d["PA"] >= 100)].copy()
    cur["BI"] = (100 * m.predict_proba(cur[BI_FEATURES])[:, 1]).round(1)
    return cur


def project_v2(d: pd.DataFrame, season: int) -> pd.DataFrame:
    """Rate model (v1 features) + aging step, times a durability-aware PA estimate."""
    from .breakout import fit_model
    model, _ = fit_model(d)
    curve = aging_curve(d)
    cur = d[(d["season"] == season) & (d["PA"] >= 100)].copy()
    cur["proj_rate_raw"] = model.predict(cur[MODEL_FEATURES])
    # the skills model only sees this season; a hitter's own recent track record adds real signal (leave-one-season-out
    # r 0.49 -> 0.52, MAE 0.0935 -> 0.0909 at a 25% weight). Track = PA-weighted pts/PA over the last three seasons
    # (150+ PA each), this year counted in full, last year 60%, two years ago 30%.
    hist3 = d[d["season"].between(season - 2, season) & (d["PA"] >= 150)].copy()
    hist3["w"] = hist3["season"].map({season: 1.0, season - 1: 0.6, season - 2: 0.3}) * hist3["PA"]
    track = (hist3["pts_pa"] * hist3["w"]).groupby(hist3["mlbam_id"]).sum() / hist3["w"].groupby(hist3["mlbam_id"]).sum()
    cur["track_rate"] = cur["mlbam_id"].map(track).fillna(cur["pts_pa"])
    cur["age_step"] = age_adjustment(curve, cur["age"] + 1)         # step from next-season age (what the year does to him)
    cur["proj_rate"] = 0.75 * cur["proj_rate_raw"] + 0.25 * cur["track_rate"] + 0.5 * cur["age_step"]  # half weight: the boosted model already sees age
    # PA expectation from pace while on the roster (late call-ups and IL time do not count as "didn't play"), falling back to raw PA
    pace_col = "pa_pace_162" if "pa_pace_162" in d.columns else None
    if pace_col:
        cur_pace = cur[pace_col].fillna(cur["PA"]).clip(upper=720)
        hist = d[d["season"].isin([season - 2, season - 1]) & (d["PA"] >= 150)].assign(p=lambda x: x[pace_col].fillna(x["PA"]).clip(upper=720)).groupby("mlbam_id")["p"].mean()
        base_pa = 0.6 * cur_pace + 0.4 * cur["mlbam_id"].map(hist).fillna(cur_pace)
        # a rookie or part-timer with few games up: pull toward his actual PA so 30-game paces are not taken at face value
        w = (cur["avail_games"].fillna(cur["G"]).clip(0, 100) / 100.0)
        base_pa = w * base_pa + (1 - w) * (0.5 * base_pa + 0.5 * cur["PA"].clip(upper=720))
    else:
        hist = d[d["season"].isin([season - 2, season - 1, season]) & (d["PA"] >= 150)].groupby("mlbam_id")["PA"].mean()
        base_pa = 0.6 * cur["PA"] + 0.4 * cur["mlbam_id"].map(hist).fillna(cur["PA"])
    # durability: heavy recent IL history trims expected PA; age over 33 trims a bit more
    ilw = cur["il_days_w"] if "il_days_w" in cur.columns else cur["il_days_3yr"]   # recency-weighted IL days: an ACL two years ago counts 30%, this year's hamstring in full
    dur = (1 - 0.0008 * ilw.clip(0, 250)) * np.where(cur["age"] >= 33, 0.95, 1.0)
    cur["proj_PA"] = (base_pa * dur).clip(200, 700).round(0)
    cur["durability"] = dur.round(3)
    cur["proj_pts"] = (cur["proj_rate"] * cur["proj_PA"]).round(0)
    cur["proj_pts_600"] = (cur["proj_rate"] * 600).round(0)
    cur["proj_rank"] = cur["proj_pts"].rank(ascending=False).astype(int)
    cur["proj_rank_600"] = cur["proj_pts_600"].rank(ascending=False).astype(int)
    return cur

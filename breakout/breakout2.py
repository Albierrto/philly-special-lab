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
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score

from . import config as C
from .breakout import add_rates, MODEL_FEATURES
from .formulas import aging_curve, age_adjustment

JUMP = 0.08
# Eight features were cut after permutation importance on every held-out season showed each of them costing AUC
# rather than adding it: sb_per600, sprint_speed, swing_take_run_value, iz_contact_percent, yoy_k_change,
# prior_best_pa, and BOTH injury columns (il_days, il_days_3yr) — the same double-counting the pitcher model had.
# The pattern in what survives: CHANGES in contact quality earn their place, LEVELS mostly do not, because a
# breakout is a hitter becoming someone new and his current level is already who he is. Leave-one-season-out
# AUC 0.743 -> 0.785 on an unchanged evaluation set.
BI_FEATURES = ["xwoba", "xiso", "xba", "barrel_batted_rate", "hard_hit_percent", "avg_best_speed", "xwobacon",
               "k_percent", "bb_percent", "oz_swing_percent", "whiff_percent",
               "avg_swing_speed", "fast_swing_rate", "ideal_angle_rate", "age", "pts_pa", "PA",
               "xLP_pa", "luck_pa", "career_best_pa", "gap_to_best", "jump_vs_prior", "prior_seasons_200",
               "seasons_200", "career_PA",
               "yoy_barrel_change", "yoy_bat_speed_change"]

# Who the model LEARNS from. At the old 150-PA bar it never saw a part-season hitter become a regular, which is
# most of what a breakout actually is. Dropping the training bar to 60 adds 387 pairs and lifts held-out AUC to
# 0.802 on the same 150+ evaluation set; it is flat from 60 down to 30, so this is the plateau rather than a blip.
BI_TRAIN_PA = 60


def career_context(d: pd.DataFrame) -> pd.DataFrame:
    """Career-best pts/PA (200+ PA seasons) through season t, seasons played, career PA, yoy skill changes.

    Also the same three measured strictly BEFORE season t. career_best_pa includes the current year, so gap_to_best is
    almost always 0 and says nothing; prior_best_pa is what the player had actually shown when the season began, which
    is what makes "did he break out THIS year" answerable at all."""
    d = d.sort_values(["mlbam_id", "season"]).copy()
    d["_rate200"] = np.where(d["PA"] >= 200, d["pts_pa"], np.nan)
    g = d.groupby("mlbam_id")
    d["career_best_pa"] = g["_rate200"].cummax()
    d["seasons_200"] = g["_rate200"].transform(lambda s: s.notna().cumsum())
    d["career_PA"] = g["PA"].cumsum()
    d["gap_to_best"] = d["pts_pa"] - d["career_best_pa"]
    # strictly prior: what he had shown before this season started
    d["prior_best_pa"] = g["_rate200"].transform(lambda s: s.shift(1).cummax())
    d["prior_seasons_200"] = g["_rate200"].transform(lambda s: s.shift(1).notna().cumsum())
    rk = d["final_hitter_rank"] if "final_hitter_rank" in d.columns else pd.Series(np.nan, index=d.index)
    d["prior_top90"] = d.assign(_rk=rk).groupby("mlbam_id")["_rk"].transform(lambda s: (s.shift(1) <= 90).cumsum() > 0)
    d["jump_vs_prior"] = d["pts_pa"] - d["prior_best_pa"].fillna(np.minimum(d["pts_pa"], 0.35))
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


def breakout_status(d: pd.DataFrame, min_pa: int = 100, age_max: float = 28.0) -> pd.Series:
    """Who a breakout number is even meaningful for, in this season's row.

    A percentage next to Aaron Judge is noise: he cannot enter new territory, he has been there for six years. And a
    percentage next to Sal Stewart is backwards — his jump already happened, in this season, and the number is quietly
    asking whether he will do it AGAIN. Four states instead:

        broke_out   he cleared the bar this season: beat his prior best by JUMP, 350+ PA, top-90 finish, and had never
                    finished top-90 before. 17 hitters in 2026 — Stewart, Jordan Walker, Jensen, Vargas, McGonigle.
        established he has finished top-90 in a season before this one. Nothing to break out of.
        candidate   neither, young enough that a jump is still plausible. This is the only group with a percentage.
        thin        too few plate appearances to say anything.
    """
    base = d["prior_best_pa"].fillna(np.minimum(d["pts_pa"], 0.35))
    rk = d["final_hitter_rank"] if "final_hitter_rank" in d.columns else pd.Series(np.nan, index=d.index)
    broke = (d["pts_pa"] >= base + JUMP) & (d["PA"] >= 350) & (rk <= 90) & (~d["prior_top90"].fillna(False))
    est = d["prior_top90"].fillna(False)
    out = pd.Series("candidate", index=d.index)
    out[est] = "established"
    out[broke] = "broke_out"
    out[d["PA"] < min_pa] = "thin"
    out[(out == "candidate") & (d["age"] > age_max)] = "established"   # past the aging curve, a first leap is not coming
    return out


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
    pr = _pairs(d, min_pa=BI_TRAIN_PA)
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
    """Breakout probability, calibrated so the number on the page means what it says.

    Raw, the classifier is badly over-spread: its 33% bucket broke out 14% of the time. A Platt fit on
    leave-one-season-out predictions lines the buckets up — 1.2 said / 1.0 happened, 3.3 / 3.7, 7.0 / 6.3,
    13.6 / 15.9 — and being strictly monotone it leaves the ordering and the AUC alone. Isotonic calibrates
    about as well but flattens the top into a single step, which would tie two dozen men at the same number
    and throw away the ranking that makes the list worth reading.
    """
    pr = _pairs(d, min_pa=BI_TRAIN_PA)
    oof, y = [], []
    for s in sorted(pr["season"].unique()):
        tr, te = pr[pr["season"] != s], pr[pr["season"] == s]
        if tr.empty or te.empty or tr["breakout_next"].nunique() < 2: continue
        oof.append(_clf().fit(tr[BI_FEATURES], tr["breakout_next"]).predict_proba(te[BI_FEATURES])[:, 1])
        y.append(te["breakout_next"].to_numpy())
    m = _clf().fit(pr[BI_FEATURES], pr["breakout_next"])
    cur = d[(d["season"] == season) & (d["PA"] >= 100)].copy()
    raw = m.predict_proba(cur[BI_FEATURES])[:, 1]
    if oof:
        lg = lambda v: np.log(np.clip(v, 1e-6, 1 - 1e-6) / (1 - np.clip(v, 1e-6, 1 - 1e-6)))
        pl = LogisticRegression().fit(lg(np.concatenate(oof)).reshape(-1, 1), np.concatenate(y))
        raw = pl.predict_proba(lg(raw).reshape(-1, 1))[:, 1]
    cur["BI"] = (100 * raw).round(1)
    return cur


def _pa_model(d: pd.DataFrame, cur: pd.DataFrame) -> pd.Series:
    """Next season's plate appearances, fitted rather than hand-weighted, with the market's view folded in.

    Benchmarked against the FantasyPros consensus over 2022-2026, out of sample, on this league's points: the
    hand-tuned pace rule scored .539 where the consensus scored .596. Almost all of that gap was playing time,
    not skill — the rate model BEATS the consensus at points per PA (.487 to .435) and loses at who actually
    gets the trips (.535 to .561). A market rank knows about a trade, a job battle and an offseason signing;
    a Statcast model cannot.

    Fitting the same inputs instead of weighting them by hand is worth most of it on its own (.579), and adding
    the ADP rank takes the whole projection to .601, past the consensus, winning three of the five seasons.
    Only the most recent preseason ADP is ever available when this runs, so that is what it is measured with.
    The market is used ONLY for playing time. The rate stays the model's, because there it is already better.
    """
    F = ["base_pa", "PA", "age", "lg_adp", "has_adp"]
    def prep(x):
        x = x.copy()
        x["lg_adp"] = np.log(x["adp_hitter_rank"].clip(1, 900).fillna(900)) if "adp_hitter_rank" in x else np.log(900)
        x["has_adp"] = x["adp_hitter_rank"].notna().astype(float) if "adp_hitter_rank" in x else 0.0
        return x
    nxt = d[["mlbam_id", "season", "PA"]].copy(); nxt["season"] -= 1
    tr = d[d["PA"] >= 150].merge(nxt.rename(columns={"PA": "next_PA"}), on=["mlbam_id", "season"], how="inner")
    if len(tr) < 100:
        return cur["base_pa"].clip(200, 700)                   # not enough history to fit: fall back to the rule
    seasons = sorted(tr["season"].unique())
    parts = []
    for s in seasons:                                          # base_pa has to be rebuilt per season, as project_v2 does
        blk = tr[tr["season"] == s]
        if blk.empty: continue
        parts.append(blk.assign(base_pa=_base_pa(d, blk)))
    tr = pd.concat(parts)
    tr = prep(tr); te = prep(cur)
    mu = tr[F].mean()
    m = LinearRegression().fit(tr[F].fillna(mu), tr["next_PA"])
    return pd.Series(np.clip(m.predict(te[F].fillna(mu)), 200, 700), index=cur.index)


def _base_pa(d: pd.DataFrame, cur: pd.DataFrame) -> pd.Series:
    """The pace-and-durability rule, as a reusable piece so the fit above trains on the same input it scores."""
    season = int(cur["season"].iloc[0])
    pace = cur["pa_pace_162"].fillna(cur["PA"]).clip(upper=720) if "pa_pace_162" in cur.columns else cur["PA"].clip(upper=720)
    if "pa_pace_162" in d.columns:
        hist = (d[d["season"].isin([season - 2, season - 1]) & (d["PA"] >= 150)]
                  .assign(p=lambda x: x["pa_pace_162"].fillna(x["PA"]).clip(upper=720)).groupby("mlbam_id")["p"].mean())
    else:
        hist = d[d["season"].isin([season - 2, season - 1]) & (d["PA"] >= 150)].groupby("mlbam_id")["PA"].mean()
    base = 0.6 * pace + 0.4 * cur["mlbam_id"].map(hist).fillna(pace)
    g = cur["avail_games"].fillna(cur["G"]) if "avail_games" in cur.columns else cur["G"]
    w = (g.clip(0, 100) / 100.0)
    base = w * base + (1 - w) * (0.5 * base + 0.5 * cur["PA"].clip(upper=720))
    ilw = cur["il_days_w"] if "il_days_w" in cur.columns else cur.get("il_days_3yr", pd.Series(0.0, index=cur.index))
    dur = (1 - 0.0008 * ilw.clip(0, 250)) * np.where(cur["age"] >= 33, 0.95, 1.0)
    return (base * dur).clip(200, 700)


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
    cur["base_pa"] = (base_pa * dur).clip(200, 700)
    cur["durability"] = dur.round(3)
    cur["proj_PA"] = _pa_model(d, cur).round(0)
    cur["proj_pts"] = (cur["proj_rate"] * cur["proj_PA"]).round(0)
    cur["proj_pts_600"] = (cur["proj_rate"] * 600).round(0)
    cur["proj_rank"] = cur["proj_pts"].rank(ascending=False).astype(int)
    cur["proj_rank_600"] = cur["proj_pts_600"].rank(ascending=False).astype(int)
    return cur

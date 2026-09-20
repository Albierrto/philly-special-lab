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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
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


# The compact set: the eight features a nested forward selection kept picking when it could not see the season it was
# scored on (age and PA in all five runs, gap to his own best and the luck gap in four, sprint speed and exit velocity
# in three). Nothing here is a level of contact quality, which is the pattern the ablations kept showing: a breakout
# is a hitter becoming someone new, so what he already is tends to be priced in and what has moved is what matters.
BI_COMPACT = ["age", "PA", "gap_to_best", "luck_pa", "sprint_speed", "exit_velocity_avg", "career_PA", "k_minus_bb"]
# counts, not rates: these stay raw when everything else is turned into a within-season percentile
BI_KEEP_RAW = {"age", "PA", "seasons_200", "prior_seasons_200", "career_PA"}


def _bi_frame(x: pd.DataFrame) -> pd.DataFrame:
    """Every feature as a within-season percentile. The league's offensive level drifts year to year and a raw xwOBA
    of .330 meant something different in 2021 than in 2026; ranking within the season takes that out and lifted the
    boosted model .767 -> .775 out of sample on its own.

    Applied AFTER the pairs and their labels are built, never before: the breakout label is defined on raw points per
    PA against a raw career best, and ranking those first would quietly redefine what a breakout is."""
    x = x.copy()
    x["k_minus_bb"] = x["k_percent"] - x["bb_percent"]
    for c in set(BI_FEATURES) | set(BI_COMPACT):
        if c not in BI_KEEP_RAW:
            x[c] = x.groupby("season")[c].rank(pct=True)
    return x


def _clf():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=25,
                                          l2_regularization=1.0, random_state=11)


def _lr():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.3, max_iter=3000))


_LOGIT = lambda v: np.log(np.clip(v, 1e-6, 1 - 1e-6) / (1 - np.clip(v, 1e-6, 1 - 1e-6)))


def _bi_fit_predict(tr: pd.DataFrame, te: pd.DataFrame) -> np.ndarray:
    """Three learners, averaged on the logit scale, every one on the same percentile features.

    The boosted tree alone is .767 out of sample and could not be improved by any single added feature: with 86
    breakouts to learn from it is saturated, and more columns made it worse. A logistic on the same 26 columns is
    .786, a logistic on the compact eight is .785, and averaging the three is .810 with a top-40 hit rate of 17.5%
    against 14.5%. The tree and the linear models disagree in useful ways (the tree over-reads sprint speed, the
    linear ones cannot bend the age curve), which is what an ensemble is for.
    """
    y = tr["breakout_next"]
    parts = [_clf().fit(tr[BI_FEATURES], y).predict_proba(te[BI_FEATURES])[:, 1],
             _lr().fit(tr[BI_FEATURES], y).predict_proba(te[BI_FEATURES])[:, 1],
             _lr().fit(tr[BI_COMPACT], y).predict_proba(te[BI_COMPACT])[:, 1]]
    z = np.mean([_LOGIT(p) for p in parts], axis=0)
    return 1 / (1 + np.exp(-z))


_BI_CACHE: dict = {}


def _bi_oof(d: pd.DataFrame):
    """Held-out predictions for every training pair, computed once per table and shared by the CV report and the
    calibration, which used to run the same five fits twice."""
    pr = _bi_frame(_pairs(d, min_pa=BI_TRAIN_PA))
    key = (len(pr), tuple(sorted(pr["season"].unique())), int(pr["breakout_next"].sum()), round(float(pr["PA"].sum())))
    if key in _BI_CACHE:
        return _BI_CACHE[key]
    oof = pd.Series(np.nan, index=pr.index)
    for s in sorted(pr["season"].unique()):
        tr, te = pr[pr["season"] != s], pr[pr["season"] == s]
        if tr.empty or te.empty or tr["breakout_next"].nunique() < 2:
            continue
        oof.loc[te.index] = _bi_fit_predict(tr, te)
    _BI_CACHE.clear(); _BI_CACHE[key] = (pr, oof)
    return pr, oof


def cross_validate_bi(d: pd.DataFrame) -> pd.DataFrame:
    pr, oof = _bi_oof(d)
    rows = []
    for s in sorted(pr["season"].unique()):
        te = pr[(pr["season"] == s) & (pr["PA"] >= 150)]; p = oof.loc[te.index]
        if te.empty or p.isna().all():
            continue
        auc = roc_auc_score(te["breakout_next"], p) if te["breakout_next"].nunique() > 1 else np.nan
        top = te.assign(p=p).sort_values("p", ascending=False).head(40)
        rows.append(dict(held_out=f"{s}->{s+1}", n=len(te), breakouts=int(te["breakout_next"].sum()),
                         base_rate=round(te["breakout_next"].mean(), 3), auc=round(auc, 3),
                         top40_hit_rate=round(top["breakout_next"].mean(), 3),
                         top40_avg_next_pts=round(top["next_pts"].mean(), 0)))
    return pd.DataFrame(rows)


def breakout_index(d: pd.DataFrame, season: int) -> pd.DataFrame:
    """Breakout probability, calibrated so the number on the page means what it says.

    Raw, a classifier is badly over-spread: the old model's 33% bucket broke out 14% of the time. A Platt fit on the
    leave-one-season-out predictions lines the buckets up, and being strictly monotone it leaves the ordering and the
    AUC alone. Isotonic calibrates about as well but flattens the top into a single step, which would tie two dozen
    men at the same number and throw away the ranking that makes the list worth reading.
    """
    pr, oof = _bi_oof(d)
    # the current season is ranked within the same population the model learned on (everyone at the training PA bar)
    cur = _bi_frame(d[(d["season"] == season) & (d["PA"] >= BI_TRAIN_PA)])
    cur = cur[cur["PA"] >= 100].copy()
    raw = _bi_fit_predict(pr, cur)
    ok = oof.notna()
    if ok.any():
        pl = LogisticRegression().fit(_LOGIT(oof[ok].to_numpy()).reshape(-1, 1), pr.loc[ok, "breakout_next"])
        raw = pl.predict_proba(_LOGIT(raw).reshape(-1, 1))[:, 1]
        # the calibration is a straight line on the logit scale, so it will happily extrapolate to a number no
        # held-out player has ever been given. The page never claims more than the model has actually earned: the
        # ceiling is the highest calibrated probability any training pair received out of sample (about 52%).
        raw = np.minimum(raw, float(pl.predict_proba(_LOGIT(oof[ok].to_numpy()).reshape(-1, 1))[:, 1].max()))
    out = d[(d["season"] == season) & (d["PA"] >= 100)].copy()
    out.attrs["raw_pairs"] = _pairs(d, min_pa=BI_TRAIN_PA)         # real numbers, for the comparables
    out["BI"] = (100 * raw).round(1)
    # what the number is a chance OF, and why he has it
    base = out["career_best_pa"].fillna(out["pts_pa"])
    base = np.where(out["seasons_200"] == 0, np.minimum(out["pts_pa"], 0.35), base)
    out["bi_line"] = (base + JUMP).round(3)                      # the rate that would count as a breakout
    out["bi_why"] = _bi_reasons(pr, cur, out)
    out["bi_skills"] = _bi_skills(d, out, season)
    out["bi_comps"] = _bi_comps(pr, out)
    return out


# how each compact-model input is said in words. The sign is the sign of its contribution for THIS player: the
# same column can be a reason for or against, and the wording follows. Every template gets the player's raw row, the
# within-season percentile of the input, and a context dict (the empirical jump rate at his age, from the training pairs).
_WHY = {
    "age":               (lambda r, pct, cx: f"{r['age']:.0f} years old, an age that jumps {cx['age_rate']:.0%} of the time",
                          lambda r, pct, cx: f"already {r['age']:.0f}, an age that jumps {cx['age_rate']:.0%} of the time"),
    "PA":                (lambda r, pct, cx: f"{int(r['PA'])} PA{' in his first season' if r.get('career_PA', 0) <= r['PA'] + 1 else ''}, already has the job",
                          lambda r, pct, cx: f"only {int(r['PA'])} PA{' in his first season' if r.get('career_PA', 0) <= r['PA'] + 1 else ''}, and a breakout needs 350 next year"),
    "gap_to_best":       (lambda r, pct, cx: f"{r['pts_pa']:.2f}/PA is his best rate yet, so the bar is one step up",
                          lambda r, pct, cx: f"{r['pts_pa']:.2f}/PA this year against {r['career_best_pa']:.2f} at his best, two steps to clear"),
    "luck_pa":           (lambda r, pct, cx: f"deserved {r['xLP_pa']:.2f}/PA on contact quality, got {r['pts_pa']:.2f}",
                          lambda r, pct, cx: f"got {r['pts_pa']:.2f}/PA but deserved {r['xLP_pa']:.2f}"),
    "sprint_speed":      (lambda r, pct, cx: f"sprint {r['sprint_speed']:.1f} ft/s ({_ord(pct)} pct)",
                          lambda r, pct, cx: f"sprint {r['sprint_speed']:.1f} ft/s ({_ord(pct)} pct)"),
    "exit_velocity_avg": (lambda r, pct, cx: f"avg EV {r['exit_velocity_avg']:.1f} ({_ord(pct)} pct)",
                          lambda r, pct, cx: f"avg EV {r['exit_velocity_avg']:.1f} ({_ord(pct)} pct)"),
    "career_PA":         (lambda r, pct, cx: f"{int(r['career_PA']):,} career PA, still new",
                          lambda r, pct, cx: f"{int(r['career_PA']):,} career PA, a known quantity"),
    "k_minus_bb":        (lambda r, pct, cx: f"K {r['k_percent']:.0f}% / BB {r['bb_percent']:.0f}%, controls the zone",
                          lambda r, pct, cx: f"K {r['k_percent']:.0f}% / BB {r['bb_percent']:.0f}%, swing and miss"),
}
# every template pair is (positive contribution, negative contribution). The coefficient signs the compact model
# learned, for the record: age -, PA +, gap to best + (being AT your best means the bar is one step up rather than
# two), luck - (luck is actual minus deserved, so negative is unlucky), sprint +, exit velocity +, career PA -,
# K minus BB -. So "PA positive" means MORE plate appearances, which reads as odd for a breakout until you remember
# the label needs 350 next year: a man who already has the job is far likelier to clear it.


def _age_rates(pr: pd.DataFrame) -> dict:
    """How often each age actually jumped, from the training pairs at 150+ PA, smoothed over the neighbouring ages.

    22-year-olds cleared the bar 18% of the time, 24-year-olds 7%, anyone 26 or older 3 to 5%. Saying the number next to
    the age turns 'he is young' from a platitude into the evidence it is."""
    x = pr[pr["PA"] >= 150]
    a = x["age"].round().clip(21, 31)
    return {int(k): float(x.loc[(a - k).abs() <= 1, "breakout_next"].mean()) for k in range(21, 32)}


def _bi_reasons(pr: pd.DataFrame, cur_pct: pd.DataFrame, cur_raw: pd.DataFrame) -> pd.Series:
    """Three reasons per player, in words, from the compact model's own arithmetic.

    The compact logistic is one of the three votes and the only one a person can read: each input's contribution is
    its coefficient times how far this player sits from the average, so the biggest positive terms ARE the reasons.
    The tree's reasons would need a Shapley pass and would mostly say the same things. Two changes the compact model
    does not see but the other two do, barrel rate and bat speed against last year, are appended when they moved.
    """
    m = _lr().fit(pr[BI_COMPACT], pr["breakout_next"])
    imp, sc, lr = m.named_steps["simpleimputer"], m.named_steps["standardscaler"], m.named_steps["logisticregression"]
    X = sc.transform(imp.transform(cur_pct[BI_COMPACT]))
    contrib = X * lr.coef_[0]                                      # (players x features), positive pushes the % up
    raw = cur_raw.set_index("mlbam_id"); pct = cur_pct.set_index("mlbam_id")
    ages = _age_rates(pr)
    out = []
    for i, pid in enumerate(cur_pct["mlbam_id"]):
        r, q = raw.loc[pid], pct.loc[pid]
        cx = {"age_rate": ages.get(int(min(max(round(float(r["age"])), 21), 31)), np.nan) if pd.notna(r.get("age")) else np.nan}
        order = np.argsort(-contrib[i])
        bits = []
        for j in order:
            if len(bits) >= 3:
                break
            f = BI_COMPACT[j]
            if abs(contrib[i, j]) < 0.05 or pd.isna(r.get(f)):
                continue
            # in a first full season the gap to his best is zero by construction, and the sentence says nothing
            if f == "gap_to_best" and (r.get("prior_seasons_200") or 0) == 0:
                continue
            # first season: career PA IS this season's PA, and the PA sentence already says so
            if f == "career_PA" and r.get("career_PA", 0) <= r["PA"] + 1:
                continue
            pos, neg = _WHY[f]
            pcv = 100 * float(q[f]) if f not in BI_KEEP_RAW and pd.notna(q[f]) else np.nan
            bits.append((pos if contrib[i, j] > 0 else neg)(r, pcv, cx))
        bc, bs = r.get("yoy_barrel_change"), r.get("yoy_bat_speed_change")
        if pd.notna(bc) and bc >= 2: bits.append(f"barrel rate up {bc:.1f} pts on last year")
        if pd.notna(bs) and bs >= 0.8: bits.append(f"bat speed up {bs:.1f} mph")
        out.append(" \u00b7 ".join(bits))
    return pd.Series(out, index=cur_pct.index)


# the Statcast line under a candidate: the numbers themselves, with where each one sits among this season's hitters,
# so a reader can see whether the skills back the percentage or whether it is age and playing time doing the work
_SKILL_COLS = [("xwoba", "xwOBA", "{:.3f}", True), ("barrel_batted_rate", "barrels", "{:.1f}%", True),
               ("exit_velocity_avg", "EV", "{:.1f}", True), ("hard_hit_percent", "hard-hit", "{:.0f}%", True),
               ("k_percent", "K", "{:.0f}%", False), ("bb_percent", "BB", "{:.0f}%", True),
               ("oz_swing_percent", "chase", "{:.0f}%", False), ("avg_swing_speed", "bat speed", "{:.1f} mph", True),
               ("sprint_speed", "sprint", "{:.1f}", True)]


def _ord(n: int) -> str:
    n = int(n); return f"{n}{'th' if 10 <= n % 100 <= 20 else {1:'st', 2:'nd', 3:'rd'}.get(n % 10, 'th')}"


def _bi_skills(d: pd.DataFrame, out: pd.DataFrame, season: int) -> pd.Series:
    pool = d[(d["season"] == season) & (d["PA"] >= 100)]
    lines = []
    for _, r in out.iterrows():
        bits = []
        for col, label, fmt, higher_good in _SKILL_COLS:
            v = r.get(col)
            if pd.isna(v) or col not in pool.columns:
                continue
            pct = (pool[col] < v).mean() if higher_good else (pool[col] > v).mean()
            bits.append(f"{label} {fmt.format(v)} ({_ord(round(100 * pct))})")
        lines.append(" · ".join(bits))
    return pd.Series(lines, index=out.index)


# who he looks like, and what happened to them. The twenty nearest player-seasons in the training pairs on age,
# playing time, this year's rate, contact quality, strikeouts, speed and how much career he has, with next year's
# result attached to each. This is the check on the model: if the men who looked like him mostly did not jump, a
# high number needs a better reason than the ones listed.
# The change columns are in the distance so the lookalikes share his REASONS, not only his level: a man whose number
# comes from a barrel-rate jump is matched with men whose barrel rate had just jumped. As a predictor on its own the
# nearest-20 rate scores .70 AUC leave-one-season-out against the model's .81, so it is the check, not the verdict.
_COMP_COLS = ["age", "PA", "pts_pa", "xwoba", "k_percent", "exit_velocity_avg", "barrel_batted_rate", "sprint_speed", "career_PA", "prior_best_pa",
              "luck_pa", "yoy_barrel_change", "yoy_bat_speed_change", "gap_to_best"]
_COMP_W = np.array([2.0, 1.2, 1.5, 1.0, 1.0, 1.0, 0.8, 0.6, 1.0, 0.8, 1.0, 0.8, 0.8, 1.0])


def _bi_comps(pr: pd.DataFrame, out: pd.DataFrame, k: int = 20) -> pd.Series:
    """JSON per player: how many of his nearest neighbours broke out, and the closest few by name with what they did."""
    import json
    # pr's feature columns are percentiles by now; comparables want the real numbers, which _pairs kept on the raw table
    src = out.attrs.get("raw_pairs")
    if src is None:
        return pd.Series([""] * len(out), index=out.index)
    X = src[_COMP_COLS].copy(); X["prior_best_pa"] = X["prior_best_pa"].fillna(X["pts_pa"])
    mu, sd = X.mean(), X.std(ddof=0).replace(0, 1)
    Z = ((X - mu) / sd).fillna(0).to_numpy() * _COMP_W
    res = []
    for _, r in out.iterrows():
        v = pd.Series({c: r.get(c) for c in _COMP_COLS}); v["prior_best_pa"] = v["prior_best_pa"] if pd.notna(v["prior_best_pa"]) else v["pts_pa"]
        z = (((v - mu) / sd).fillna(0).to_numpy() * _COMP_W).astype(float)
        dist = np.sqrt(((Z - z) ** 2).sum(axis=1))
        # never his own earlier seasons: the question is what happened to people LIKE him
        dist = np.where(src["mlbam_id"].to_numpy() == r["mlbam_id"], np.inf, dist)
        idx = np.argsort(dist)[:k]
        nb = src.iloc[idx]
        # compact on purpose, it rides in the page data for every hitter: [name, season, age, rate, next rate, next pts, broke out]
        top = [[str(x["name"]), int(x["season"]), int(round(float(x["age"]))), round(float(x["pts_pa"]), 2), round(float(x["next_pts_pa"]), 2),
                int(round(float(x["next_pts"]))), int(x["breakout_next"])] for _, x in nb.head(5).iterrows()]
        res.append(json.dumps(dict(n=int(len(nb)), b=int(nb["breakout_next"].sum()), c=top), separators=(",", ":")))
    return pd.Series(res, index=out.index)


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
    """Rate model times a durability-aware PA estimate.

    The rate is a ridge regression on this season's Statcast and production plus his own two prior seasons (see
    breakout.add_track). It replaced a gradient-boosted tree in September 2026: out of sample the tree managed
    Spearman .451 with a hand-weighted track blend on top, the ridge gets .500 with the track inside it, and it wins
    every held-out season. The post-hoc aging step is gone too, because the model's own age term already carries
    it: adding half a step back moved rho by .001 and made MAE worse.
    """
    from .breakout import fit_model, add_track, RATE_FEATURES
    d = add_track(d) if "rate_m1" not in d.columns else d
    model, _ = fit_model(d)
    curve = aging_curve(d)
    cur = d[(d["season"] == season) & (d["PA"] >= 100)].copy()
    cur["proj_rate_raw"] = model.predict(cur[RATE_FEATURES])
    # kept for the card: what a hitter's own last three seasons say, PA-weighted, this year in full
    hist3 = d[d["season"].between(season - 2, season) & (d["PA"] >= 150)].copy()
    hist3["w"] = hist3["season"].map({season: 1.0, season - 1: 0.6, season - 2: 0.3}) * hist3["PA"]
    track = (hist3["pts_pa"] * hist3["w"]).groupby(hist3["mlbam_id"]).sum() / hist3["w"].groupby(hist3["mlbam_id"]).sum()
    cur["track_rate"] = cur["mlbam_id"].map(track).fillna(cur["pts_pa"])
    cur["age_step"] = age_adjustment(curve, cur["age"] + 1)         # shown on the card; the multi-year view uses the curve itself
    cur["proj_rate"] = cur["proj_rate_raw"]
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

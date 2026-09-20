"""Starting pitchers for an SP-only points league (5 SP slots, 10 starts a week cap).

League points (verified 116/116 against the Fantrax 2026 table):
    IP + K − ER + 4·QS + 10·CG + 8·SHO − 3·BS   (+10 no-hitter, +25 perfect game: not in public feeds)

Because starts are capped, the currency is **points per start**; season value is pts/GS × starts.

Public stand-ins for the pitch-modelling metrics Bort follows (PitcherList Pitching+/PLV, Sarris' Stuff+/Location+):
    Stuff proxy     usage-weighted z of: whiff%, in-zone whiff%, put-away%, fastball velocity, movement uniqueness
                    (usage-weighted |break vs league| on the pitcher's arsenal), per-100 run value of his pitches
    Location proxy  z of: edge%, first-strike%, zone% (mild), −meatball%, −BB%, chase% induced
    Pitching proxy  ridge-fitted blend of both plus xERA / K−BB%, trained to predict NEXT-year points per start
All three are scaled like the originals: 100 = league average, 10 = one standard deviation among starters.
They are not PLV or Stuff+ (those need pitch-level models); they are the public ingredients arranged the same way.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import roc_auc_score

from .config import DATA
from .names import key


def _f(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default

_S = requests.Session(); _S.headers.update({"User-Agent": "Mozilla/5.0"})
PSCORE = {"IP": 1, "K": 1, "ER": -1, "QS": 4, "CG": 10, "SHO": 8, "BS": -3}


# ------------------------------------------------------------------------------------------------ sources
def mlb_pitching(year: int) -> pd.DataFrame:
    def load(stats):
        p = DATA / "mlb" / f"pitching_{stats}_{year}.json"
        if not p.exists():
            out, offset = [], 0
            while True:
                u = (f"https://statsapi.mlb.com/api/v1/stats?stats={stats}&group=pitching&season={year}&sportId=1"
                     f"&playerPool=all&limit=1000&offset={offset}&hydrate=person")
                st = _S.get(u, timeout=120).json()["stats"][0]; sp = st.get("splits", []); out += sp; offset += len(sp)
                if not sp or offset >= st.get("totalSplits", 0): break
            p.parent.mkdir(parents=True, exist_ok=True)     # a fresh CI runner has no data/mlb cache at all
            p.write_text(json.dumps(out))
        return json.loads(p.read_text())
    rows = {}
    for s in load("season"):
        p, st = s["player"], s["stat"]
        ip_str = str(st.get("inningsPitched", "0"))
        a, _, b = ip_str.partition("."); ip = int(a) + int(b or 0) / 3
        rows[p["id"]] = dict(mlbam_id=p["id"], name=p["fullName"], season=year, birth_date=p.get("birthDate"),
                             throws=(p.get("pitchHand") or {}).get("code"),
                             team=(s.get("team") or {}).get("name", "multi") if s.get("numTeams", 1) == 1 else "multi",
                             G=st.get("gamesPitched", 0), GS=st.get("gamesStarted", 0), IP=ip, K=st.get("strikeOuts", 0),
                             BB=st.get("baseOnBalls", 0), ER=st.get("earnedRuns", 0), H=st.get("hits", 0), HR=st.get("homeRuns", 0),
                             CG=st.get("completeGames", 0), SHO=st.get("shutouts", 0), BS=st.get("blownSaves", 0), SV=st.get("saves", 0),
                             HLD=st.get("holds", 0), W=st.get("wins", 0), L=st.get("losses", 0), ERA=_f(st.get("era")),
                             WHIP=_f(st.get("whip")), BF=st.get("battersFaced", 0), pitches=st.get("numberOfPitches", 0))
    for s in load("seasonAdvanced"):
        st = s["stat"]; pid = s["player"]["id"]
        if pid in rows:
            rows[pid].update(QS=st.get("qualityStarts", 0), k_minus_bb=_f(st.get("strikeoutsMinusWalksPercentage")),
                             whiff_pct_adv=_f(st.get("whiffPercentage")), pitches_per_ip=_f(st.get("pitchesPerInning")),
                             gb_pct=_f(st.get("groundOutsToAirouts")), babip=_f(st.get("babip")))
    df = pd.DataFrame(list(rows.values()))
    df["pts"] = sum(w * df[c] for c, w in PSCORE.items())
    df["pts_gs"] = np.where(df["GS"] > 0, df["pts"] / df["GS"].replace(0, np.nan), np.nan)
    df["pts_ip"] = df["pts"] / df["IP"].replace(0, np.nan)
    df["age"] = ((pd.Timestamp(f"{year}-06-30") - pd.to_datetime(df["birth_date"])).dt.days / 365.25).round(1)
    df["is_sp"] = (df["GS"] >= 5) | ((df["GS"] >= 3) & (df["GS"] / df["G"].replace(0, np.nan) >= 0.5))
    df["nkey"] = df["name"].apply(key)
    return df


SAV_P = ["player_age", "pa", "k_percent", "bb_percent", "xba", "xslg", "woba", "xwoba", "xera", "xobp", "xiso", "wobacon", "xwobacon",
         "exit_velocity_avg", "launch_angle_avg", "sweet_spot_percent", "barrel_batted_rate", "hard_hit_percent", "avg_best_speed",
         "z_swing_percent", "z_swing_miss_percent", "oz_swing_percent", "oz_swing_miss_percent", "oz_contact_percent", "iz_contact_percent",
         "in_zone_percent", "edge_percent", "meatball_percent", "meatball_swing_percent", "whiff_percent", "swing_percent",
         "f_strike_percent", "groundballs_percent", "flyballs_percent", "linedrives_percent", "popups_percent", "pull_percent",
         "swing_take_run_value", "pitch_run_value_fastball", "pitch_run_value_breaking", "pitch_run_value_offspeed", "pitch_count",
         "pitch_count_fastball", "pitch_count_breaking", "pitch_count_offspeed", "p_quality_start", "p_era", "pitch_hand"]


def savant_pitchers(year: int) -> pd.DataFrame:
    p = DATA / "savant" / f"custom_pitcher_{year}.json"
    if not p.exists():
        u = (f"https://baseballsavant.mlb.com/leaderboard/custom?year={year}&type=pitcher&filter=&min=50&selections=xwoba"
             f"&chart=false&x=xwoba&y=xwoba&r=no&chartType=beeswarm&sort=xwoba&sortDir=asc")
        h = _S.get(u, timeout=120).text; i = h.find("var data = ["); j = h.find("];", i)
        p.parent.mkdir(parents=True, exist_ok=True); p.write_text(h[i + 11:j + 1])
    df = pd.DataFrame(json.loads(p.read_text()))
    keep = [c for c in SAV_P if c in df.columns]
    df = df[["player_id"] + keep].rename(columns={"player_id": "mlbam_id", "pa": "pa_savant", "pitch_hand": "throws_sv"})
    for c in df.columns:
        if c not in ("mlbam_id", "throws_sv"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def arsenal(year: int) -> pd.DataFrame:
    """Per pitcher: arsenal-level aggregates from Savant pitch-arsenal-stats, velocity, and movement leaderboards."""
    a = pd.read_csv(DATA / "savant" / "arsenal" / f"arsenal_stats_{year}.csv")
    a = a[a["pitches"] >= 25].copy()
    a["w"] = a["pitch_usage"] / 100
    g = a.groupby("player_id")
    out = pd.DataFrame({
        "n_pitches_10": g.apply(lambda d: int((d["pitch_usage"] >= 10).sum())),
        "arsenal_whiff": g.apply(lambda d: np.average(d["whiff_percent"].fillna(0), weights=d["w"] + 1e-9)),
        "arsenal_putaway": g.apply(lambda d: np.average(d["put_away"].fillna(0), weights=d["w"] + 1e-9)),
        "arsenal_xwoba": g.apply(lambda d: np.average(d["est_woba"].fillna(d["est_woba"].mean()), weights=d["w"] + 1e-9)),
        "arsenal_rv100": g.apply(lambda d: np.average(d["run_value_per_100"].fillna(0), weights=d["w"] + 1e-9)),
        "best_pitch_whiff": g["whiff_percent"].max(),
        "best_pitch_rv100": g["run_value_per_100"].max(),
        "n_plus_pitches": g.apply(lambda d: int(((d["run_value_per_100"] > 0) & (d["pitch_usage"] >= 8)).sum())),
        "best_pitch": g.apply(lambda d: d.sort_values("run_value_per_100", ascending=False)["pitch_name"].iloc[0]),
        "arsenal_desc": g.apply(lambda d: ", ".join(f"{r.pitch_name} {r.pitch_usage:.0f}% ({r.whiff_percent:.0f}% whiff)" for r in d.sort_values("pitch_usage", ascending=False).itertuples())),
    }).reset_index().rename(columns={"player_id": "mlbam_id"})
    v = pd.read_csv(DATA / "savant" / "arsenal" / f"arsenal_avg_speed_{year}.csv").rename(columns={"pitcher": "mlbam_id"})
    v["fb_velo"] = v[["ff_avg_speed", "si_avg_speed"]].max(axis=1)
    out = out.merge(v[["mlbam_id", "fb_velo", "ff_avg_speed", "si_avg_speed"]], on="mlbam_id", how="left")
    m = pd.read_csv(DATA / "savant" / "arsenal" / f"movement_{year}.csv").rename(columns={"pitcher_id": "mlbam_id"})
    m["uniq"] = (m["diff_z"].abs() + m["diff_x"].abs())
    mv = m.groupby("mlbam_id").apply(lambda d: np.average(d["uniq"], weights=d["pitch_per"].fillna(1) + 1e-9)).rename("movement_uniqueness").reset_index()
    out = out.merge(mv, on="mlbam_id", how="left")
    return out


# ------------------------------------------------------------------------------------------------ table
def pitcher_seasons(seasons) -> pd.DataFrame:
    frames = []
    for y in seasons:
        d = mlb_pitching(y).merge(savant_pitchers(y), on="mlbam_id", how="left").merge(arsenal(y), on="mlbam_id", how="left")
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["IP"] >= 10)]
    df["sp_rank"] = df[df["is_sp"]].groupby("season")["pts"].rank(ascending=False, method="min")
    return df


def _z(s): return (s - s.mean()) / (s.std(ddof=0) or 1)


def proxies(ps: pd.DataFrame, min_ip=50) -> pd.DataFrame:
    """Stuff / Location / Pitching proxies on the 100 ± 10 scale, computed within season among starters >= min_ip."""
    d = ps.copy()
    for col in ("stuff_plus", "location_plus", "pitching_plus"):
        d[col] = np.nan
    for s, g in d.groupby("season"):
        q = g[(g["is_sp"]) & (g["IP"] >= min_ip)]
        if len(q) < 30: continue
        stuff = (_z(q["whiff_percent"]) + _z(q["z_swing_miss_percent"]) + _z(q["arsenal_putaway"]) + _z(q["fb_velo"])
                 + 0.5 * _z(q["movement_uniqueness"]) + _z(q["arsenal_rv100"])) / 5.5
        loc = (_z(q["edge_percent"]) + _z(q["f_strike_percent"]) + 0.5 * _z(q["in_zone_percent"]) - _z(q["meatball_percent"])
               - _z(q["bb_percent"]) + _z(q["oz_swing_percent"])) / 5.5
        d.loc[q.index, "stuff_plus"] = (100 + 10 * stuff / stuff.std(ddof=0)).round(1)
        d.loc[q.index, "location_plus"] = (100 + 10 * loc / loc.std(ddof=0)).round(1)
    return d


PITCH_FEATURES = ["stuff_plus", "location_plus", "xera", "k_percent", "bb_percent", "whiff_percent", "z_swing_miss_percent",
                  "oz_swing_percent", "edge_percent", "meatball_percent", "f_strike_percent", "barrel_batted_rate", "hard_hit_percent",
                  "xwoba", "groundballs_percent", "fb_velo", "movement_uniqueness", "arsenal_whiff", "arsenal_putaway", "arsenal_rv100",
                  "n_plus_pitches", "n_pitches_10", "age", "pts_gs", "pts_ip", "IP", "GS", "pitches_per_ip", "k_minus_bb", "il_days", "il_days_3yr"]


def _pairs(d: pd.DataFrame, min_gs=10):
    cur = d[(d["is_sp"]) & (d["GS"] >= min_gs)].copy()
    nxt = d[["mlbam_id", "season", "pts_gs", "pts", "GS", "sp_rank"]].copy(); nxt["season"] -= 1
    nxt = nxt.rename(columns={"pts_gs": "next_pts_gs", "pts": "next_pts", "GS": "next_GS", "sp_rank": "next_rank"})
    m = cur.merge(nxt, on=["mlbam_id", "season"], how="inner")
    return m[m["next_GS"] >= min_gs]


def fit_pitching_plus(d: pd.DataFrame) -> pd.DataFrame:
    """Pitching proxy = ridge blend of stuff, location, xERA, K−BB% fitted to NEXT-year points per start."""
    pr = _pairs(d).dropna(subset=["stuff_plus", "location_plus", "xera"])
    X = pr[["stuff_plus", "location_plus", "xera", "k_minus_bb"]].fillna(pr[["stuff_plus", "location_plus", "xera", "k_minus_bb"]].median())
    r = Ridge(alpha=5.0).fit(X, pr["next_pts_gs"])
    out = d.copy()
    ok = out["stuff_plus"].notna()
    Xo = out.loc[ok, ["stuff_plus", "location_plus", "xera", "k_minus_bb"]].fillna(X.median())
    raw = pd.Series(r.predict(Xo), index=Xo.index)
    out.loc[ok, "pitching_plus_raw"] = raw
    for s, g in out[ok].groupby("season"):
        z = _z(g["pitching_plus_raw"]); out.loc[g.index, "pitching_plus"] = (100 + 10 * z).round(1)
    out.attrs["pitching_plus_coef"] = dict(zip(["stuff_plus", "location_plus", "xera", "k_minus_bb"], r.coef_.round(4)))
    return out


def cross_validate(d: pd.DataFrame) -> pd.DataFrame:
    """Scores the model the projection actually uses: the three-term ridge in project() (skills, own track, age).

    Until September 2026 this table scored a boosted tree the projection had already stopped using, so the Methods
    page was grading a model nobody was reading. Same fit as project(), held out a season at a time."""
    pr = _pairs(d)
    pr["track3"] = track3(d, pr)
    pr["skills"] = pr["pitching_plus_raw"].fillna(pr["track3"]) if "pitching_plus_raw" in pr.columns else pr["track3"]
    rows = []
    for s in sorted(pr["season"].unique()):
        tr, te = pr[pr["season"] != s], pr[pr["season"] == s]
        rr = Ridge(alpha=1.0).fit(np.c_[tr["skills"], tr["track3"], tr["age"]], tr["next_pts_gs"])
        p = rr.predict(np.c_[te["skills"], te["track3"], te["age"]])
        rows.append(dict(held_out=f"{s}->{s+1}", n=len(te), r_model=round(np.corrcoef(p, te["next_pts_gs"])[0, 1], 3),
                         r_carry_over=round(np.corrcoef(te["pts_gs"], te["next_pts_gs"])[0, 1], 3),
                         r_pitching_plus=round(np.corrcoef(te["pitching_plus"].fillna(100), te["next_pts_gs"])[0, 1], 3),
                         r_xera=round(-np.corrcoef(te["xera"].fillna(te["xera"].mean()), te["next_pts_gs"])[0, 1], 3)))
    return pd.DataFrame(rows)


def career_context(d: pd.DataFrame) -> pd.DataFrame:
    d = d.sort_values(["mlbam_id", "season"]).copy()
    d["_r"] = np.where((d["GS"] >= 10), d["pts_gs"], np.nan)
    g = d.groupby("mlbam_id")
    d["career_best_gs"] = g["_r"].cummax(); d["seasons_10gs"] = g["_r"].transform(lambda s: s.notna().cumsum())
    d["career_GS"] = g["GS"].cumsum(); d["gap_to_best"] = d["pts_gs"] - d["career_best_gs"]
    d["yoy_stuff_change"] = np.where(g["GS"].shift(1) >= 10, d["stuff_plus"] - g["stuff_plus"].shift(1), np.nan)
    d["yoy_velo_change"] = np.where(g["GS"].shift(1) >= 10, d["fb_velo"] - g["fb_velo"].shift(1), np.nan)
    return d.drop(columns=["_r"])


BI_P = PITCH_FEATURES + ["career_best_gs", "gap_to_best", "seasons_10gs", "career_GS", "yoy_stuff_change", "yoy_velo_change"]
JUMP_GS = 1.5


def breakout_index(d: pd.DataFrame, season: int):
    pr = _pairs(d)
    base = pr["career_best_gs"].fillna(pr["pts_gs"]); base = np.where(pr["seasons_10gs"] == 0, np.minimum(pr["pts_gs"], 8.0), base)
    pr["breakout_next"] = ((pr["next_pts_gs"] >= base + JUMP_GS) & (pr["next_GS"] >= 20) & (pr["next_rank"] <= 40)).astype(int)
    clf = lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=250, min_samples_leaf=20, l2_regularization=1.0, random_state=11)
    cv = []
    for s in sorted(pr["season"].unique()):
        tr, te = pr[pr["season"] != s], pr[pr["season"] == s]
        if te["breakout_next"].nunique() < 2 or tr["breakout_next"].sum() < 5: continue
        m = clf().fit(tr[BI_P], tr["breakout_next"]); p = m.predict_proba(te[BI_P])[:, 1]
        top = te.assign(p=p).sort_values("p", ascending=False).head(20)
        cv.append(dict(held_out=f"{s}->{s+1}", n=len(te), breakouts=int(te["breakout_next"].sum()), base_rate=round(te["breakout_next"].mean(), 3),
                       auc=round(roc_auc_score(te["breakout_next"], p), 3), top20_hit_rate=round(top["breakout_next"].mean(), 3)))
    m = clf().fit(pr[BI_P], pr["breakout_next"])
    cur = d[(d["season"] == season) & (d["is_sp"]) & (d["GS"] >= 5)].copy()
    cur["BI"] = (100 * m.predict_proba(cur[BI_P])[:, 1]).round(1)
    return cur, pd.DataFrame(cv)


NEUTRAL_AGE = 27       # the rate projection is quoted "as a 27-year-old"; age_step is the difference from that
# Extra points per start lost past LATE_FROM, on top of the fitted linear age term, for survivor bias: the pairs the
# ridge sees are pitchers who were still starting the following year, so it cannot see the ones who fell off a cliff.
# It was hand-set at 0.15/yr from 34 and never checked. Checked on the 72 age-34+ pairs in this data it was WORSE THAN
# APPLYING NOTHING (MAE 1.269 against 1.252, bias +0.490 against +0.013), because 34-35 year-olds who are still
# starting do not decline at all - the model is already 0.33 points per start too PESSIMISTIC on them before the
# penalty, and the penalty pushed that to +0.62. The decline only shows up at 36+ (bias -0.56 with no penalty), and
# 0.10/yr from 36 corrects it without overshooting: MAE 1.216, bias +0.151, the best of every setting tried.
# n=72 is thin, so this is a smaller correction applied later rather than a confident one.
LATE_DECLINE = 0.10
LATE_FROM = 36
# Same story on projected STARTS: cutting 7% off every 34-year-old made the age-34+ projection worse (MAE 7.616,
# bias +0.613 starts) than not cutting at all (7.417, -0.094). Moving the cut to 36 is the best of the three: 7.413
# and +0.101. Note the ridge is fitted on next_GS / durability and then multiplied back, so this is not double-counted.
DUR_AGE_FROM = 36
DUR_AGE_MULT = 0.93


def _clean_leash(d: pd.DataFrame) -> pd.Series:
    """Innings per start, but ONLY for seasons he was a full-time starter.

    IP/GS across every row is garbage: IP counts relief innings and GS counts only starts, so Jesse Chavez 2022 (60
    games, 1 start, 69.3 innings) reads as 69 innings per start, and the whole column tops out at 88. Restricting to
    10+ starts with at most 3 relief appearances gives a sane distribution: min 3.36, median 5.44, max 7.15.
    """
    relief = d["G"] - d["GS"]
    return pd.Series(np.where((d["GS"] >= 10) & (relief <= 3), d["IP"] / d["GS"].replace(0, np.nan), np.nan), index=d.index)


def _leash_model(d: pd.DataFrame, t3):
    """Fit next season's innings per start. It is predicted by how GOOD he is, not by what he got this year.

    Regressed on itself, a clean leash carries a slope of 0.269 - so about 73% of a short leash comes off, not the 90%
    the contaminated column suggested. But in a multivariate fit, this year's leash has NO independent signal left
    (t -0.42) once quality is in: what predicts next year's leash is points per inning (+0.081 per unit, t +2.62).
    Managers hand innings to pitchers who are getting outs, which is why regressing everyone toward the league average
    was wrong - it quietly took innings off the aces and handed them to the innings-eaters.

    Age moves it too, and not the way the folklore says. Leash rises for the young (+0.18 an outing under 25, +0.17 at
    25-26), is FLAT right through the supposed peak (-0.03 to -0.05 from 27 to 34), and only falls at 35+ (-0.21).
    So it is fitted as quality plus a young flag and a 35+ flag, with no term for his own leash at all - which also
    means it is defined for a swingman who has no clean leash season to measure.
    """
    leash = _clean_leash(d)
    s = d.assign(_leash=leash)
    nxt = s.set_index(["mlbam_id", "season"])["_leash"]
    fit = s[s["_leash"].notna()].copy()
    fit["next_leash"] = [nxt.get((r.mlbam_id, r.season + 1), np.nan) for r in fit.itertuples()]
    fit["rate3"] = t3(fit, "pts_ip")
    fit = fit.dropna(subset=["next_leash", "rate3"])
    X = np.c_[fit["rate3"], (fit["age"] < 27).astype(float), (fit["age"] >= 35).astype(float)]
    lr = LinearRegression().fit(X, fit["next_leash"])
    lo, hi = float(leash.quantile(0.02)), float(leash.quantile(0.98))
    def proj(rate3, age):
        v = lr.intercept_ + lr.coef_[0] * rate3 + lr.coef_[1] * (age < 27) + lr.coef_[2] * (age >= 35)
        return np.clip(v, lo, hi)
    proj.coef = dict(intercept=round(float(lr.intercept_), 3), rate=round(float(lr.coef_[0]), 3),
                     young=round(float(lr.coef_[1]), 3), age35=round(float(lr.coef_[2]), 3), n=int(len(fit)))
    return proj


def _t3(d: pd.DataFrame, rows: pd.DataFrame, col: str) -> pd.Series:
    """GS-weighted value of `col` over the last three seasons (this year in full, last year 60%, two ago 30%)."""
    dd = d[(d["is_sp"]) & (d["GS"] >= 1)][["mlbam_id", "season", "GS", col]].rename(columns={"season": "s_h"})
    m = rows[["mlbam_id", "season"]].reset_index().merge(dd, on="mlbam_id")
    m = m[(m["s_h"] <= m["season"]) & (m["s_h"] >= m["season"] - 2)]
    m["w"] = m["GS"] * (m["season"] - m["s_h"]).map({0: 1.0, 1: 0.6, 2: 0.3})
    t = (m[col] * m["w"]).groupby(m["index"]).sum() / m["w"].groupby(m["index"]).sum()
    return pd.Series(rows.index.map(t), index=rows.index)


def track3(d: pd.DataFrame, rows: pd.DataFrame) -> pd.Series:
    """Track record in points per start, re-expressed at the leash he is projected to get next year.

    Points per start conflates two different things: how good a pitcher is per inning, and how long they let him go.
    A man back from surgery on a five-inning limit posts a low points-per-start that says little about next season.
    Drew Rasmussen 2025: 150 innings over 31 starts, 4.84 an outing, 8.48 points a start. His points per INNING that
    year were fine - the leash was the story, and the leash came off (5.73 and 12.08 in 2026).

    So the history is carried as points per inning, which is skill and persists, times a projected leash from
    _leash_model. Cross-validated the rate projection goes from MAE 1.5106 on plain points per start to 1.4948.
    """
    rate3 = _t3(d, rows, "pts_ip")
    proj = _leash_model(d, lambda rws, c: _t3(d, rws, c))
    out = rate3 * proj(rate3, rows["age"])
    return out.fillna(_t3(d, rows, "pts_gs")).fillna(rows["pts_gs"])


def comeback_year(d: pd.DataFrame, rows: pd.DataFrame) -> pd.Series:
    """Flag: the season BEFORE this one was lost (<=8 starts) and the one before that was real (>=15).

    Not a model term - a label. On the 11 such pairs in this data the rate model came in 1.52 points per start too
    LOW (se 0.82, t +1.85), and those pitchers went 8.73 -> 10.96 the following year (+2.22) while everyone else
    went 10.24 -> 9.94 (-0.30). The effect is large and the mechanism is not mysterious, but n=11 is far too thin to
    fit a coefficient on, and most of it is the leash, which track3 now handles. So it is surfaced and left to the
    reader rather than baked into everybody's projection.
    """
    gs = d.set_index(["mlbam_id", "season"])["GS"]
    prev = pd.Series([gs.get((r.mlbam_id, r.season - 1), 0.0) for r in rows.itertuples()], index=rows.index)
    prev2 = pd.Series([gs.get((r.mlbam_id, r.season - 2), 0.0) for r in rows.itertuples()], index=rows.index)
    return (prev <= 8) & (prev2 >= 15)


def project(d: pd.DataFrame, season: int) -> pd.DataFrame:
    """Points per start = ridge(skills, track record, age) fitted to next-year points per start, times projected starts.

    skills   = the Pitching+ inputs (Stuff+, Location+, xERA, K−BB%) already fitted to next-year pts/GS in fit_pitching_plus
    track    = GS-weighted pts/GS over the last three seasons (1 / 0.6 / 0.3)
    age      = linear (about −0.05 pts/GS per year) plus LATE_DECLINE per year past LATE_FROM (see the note on those
               constants: both late-career penalties were hand-set from 34 and, measured, were worse than nothing)
    Leave-one-season-out this beats the boosted model in four folds of five (r 0.50–0.67 vs 0.42–0.63) and, being three
    terms, cannot rate a bad year above a good one the way the boosted model did (Anthony Kay over Shota Imanaga).
    Projected starts = ridge(GS, last year's GS, age, 3-year points per start) fitted to next-year GS, times an
    age-only durability factor. IL history is deliberately NOT in it - see the note in durab()."""
    pr = _pairs(d)
    pr["track3"] = track3(d, pr); pr["skills"] = pr["pitching_plus_raw"].fillna(pr["track3"]) if "pitching_plus_raw" in pr.columns else pr["track3"]
    X = np.c_[pr["skills"], pr["track3"], pr["age"]]
    rr = Ridge(alpha=1.0).fit(X, pr["next_pts_gs"]); b_sk, b_tr, b_age = rr.coef_; b0 = rr.intercept_
    # descriptive aging curve for the chart: delta method on pts/GS
    pp = pr.copy(); pp["age_i"] = pp["age"].round().astype(int); pp["delta"] = pp["next_pts_gs"] - pp["pts_gs"]
    ag = pp.groupby("age_i")["delta"].mean().rolling(3, center=True, min_periods=1).mean()
    cur = d[(d["season"] == season) & (d["is_sp"]) & (d["GS"] >= 5)].copy()
    cur["track3"] = track3(d, cur); cur["skills"] = cur["pitching_plus_raw"].fillna(cur["track3"]) if "pitching_plus_raw" in cur.columns else cur["track3"]
    cur["ip_gs_last"] = _clean_leash(cur).round(2)     # blank unless he was a full-time starter; see _clean_leash
    cur["comeback"] = comeback_year(d, cur)                                       # first year back from a lost season
    cur["proj_pts_gs_raw"] = b0 + b_sk * cur["skills"] + b_tr * cur["track3"] + b_age * NEUTRAL_AGE
    cur["age_step"] = b_age * (cur["age"] - NEUTRAL_AGE) - LATE_DECLINE * (cur["age"] + 1 - LATE_FROM).clip(lower=0)
    cur["proj_pts_gs"] = cur["proj_pts_gs_raw"] + cur["age_step"]
    # starts: ridge on this year's GS, last year's GS (0 if he had no MLB season) and age, target next-year GS / durability
    gs_prev = d.set_index(["mlbam_id", "season"])["GS"]
    def prev(rows): return pd.Series([gs_prev.get((r.mlbam_id, r.season - 1), 0.0) for r in rows.itertuples()], index=rows.index)
    def durab(rows):
        # NO IL TERM. It used to cut 0.1% of a pitcher's starts per recency-weighted IL day, and that was charging him
        # twice for the same injury: a pitcher who missed time already shows fewer GS, and GS is the strongest input to
        # the ridge below. Fitted on 801 pairs, IL history adds nothing once GS, last year's GS and age are known -
        # il_days_w +0.0049 starts per day (se 0.0061, t +0.80, and POSITIVE), il_days_3yr -0.0013 (t -0.28),
        # il_stints_3yr -0.118 (t -0.45), il_60 +0.09 (t +0.06). The shipped 0.001/day is about -0.021 starts per day,
        # roughly four standard errors the wrong side of zero. On the 190 pairs with 90+ weighted IL days - Wheeler and
        # Rasmussen's cohort - it projected 1.40 starts too FEW (MAE 9.194); dropping it leaves bias +0.15 (MAE 8.983),
        # and overall MAE goes 8.580 -> 8.532. The age factor survives because it was checked separately.
        return np.where(rows["age"] >= DUR_AGE_FROM, DUR_AGE_MULT, 1.0)
    gp = d[(d["is_sp"]) & (d["GS"] >= 10)].merge(d[["mlbam_id", "season", "GS"]].assign(season=lambda x: x["season"] - 1).rename(columns={"GS": "next_GS"}), on=["mlbam_id", "season"])
    # QUALITY BELONGS IN THE STARTS MODEL. Without it, a short season means the same thing whoever threw it, and that is
    # plainly false: a rotation spot is earned. Residual next-year starts against the ridge on (GS, prev GS, age) rise
    # +0.855 per point of points-per-start (se 0.140, t +6.10), +0.224 per point of Pitching+ (t +6.36), +0.476 per
    # point of K% (t +6.69) and fall -2.38 per point of xERA (t -6.02) - four independent measures all past t 6.
    # Among pitchers whose season was SHORT (GS <= 26) the old model missed the best quartile by +3.12 starts and the
    # worst by -3.32: it handed the good ones too few and the bad ones too many. 5-fold CV MAE 8.571 -> 8.373.
    gp["track3"] = track3(d, gp)
    rg = Ridge(alpha=1.0).fit(np.c_[gp["GS"], prev(gp), gp["age"], gp["track3"]], gp["next_GS"] / durab(gp))
    g_gs, g_prev, g_age, g_q = rg.coef_; g0 = rg.intercept_
    dur = durab(cur)
    cur["proj_GS"] = ((g0 + g_gs * cur["GS"] + g_prev * prev(cur) + g_age * cur["age"] + g_q * cur["track3"]) * dur).clip(10, 32).round(0); cur["durability"] = pd.Series(dur, index=cur.index).round(3)
    cur["proj_pts"] = (cur["proj_pts_gs"] * cur["proj_GS"]).round(0)
    cur["proj_rank"] = cur["proj_pts"].rank(ascending=False).astype(int)
    cur["proj_rank_gs"] = cur["proj_pts_gs"].rank(ascending=False).astype(int)
    cur.attrs["proj_coef"] = dict(intercept=round(float(b0), 3), skills=round(float(b_sk), 3), track=round(float(b_tr), 3), age=round(float(b_age), 4),
                                  gs_intercept=round(float(g0), 2), gs=round(float(g_gs), 3), gs_prev=round(float(g_prev), 3), gs_age=round(float(g_age), 3), gs_quality=round(float(g_q), 3))
    return cur, ag.reset_index().rename(columns={"age_i": "age", "delta": "yoy_delta"})

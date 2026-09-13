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
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import Ridge
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
        h = _S.get(u, timeout=120).text; i = h.find("var data = ["); j = h.find("];", i); p.write_text(h[i + 11:j + 1])
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
    pr = _pairs(d)
    rows = []
    for s in sorted(pr["season"].unique()):
        tr, te = pr[pr["season"] != s], pr[pr["season"] == s]
        m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=15, l2_regularization=1.0, random_state=7)
        m.fit(tr[PITCH_FEATURES], tr["next_pts_gs"]); p = m.predict(te[PITCH_FEATURES])
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


def project(d: pd.DataFrame, season: int) -> pd.DataFrame:
    pr = _pairs(d)
    m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=15, l2_regularization=1.0, random_state=7)
    m.fit(pr[PITCH_FEATURES], pr["next_pts_gs"])
    # pitcher aging: delta method on pts/GS
    pp = pr.copy(); pp["age_i"] = pp["age"].round().astype(int); pp["delta"] = pp["next_pts_gs"] - pp["pts_gs"]
    ag = pp.groupby("age_i")["delta"].mean().rolling(3, center=True, min_periods=1).mean()
    cur = d[(d["season"] == season) & (d["is_sp"]) & (d["GS"] >= 5)].copy()
    cur["proj_pts_gs_raw"] = m.predict(cur[PITCH_FEATURES])
    cur["age_step"] = (cur["age"] + 1).round().clip(ag.index.min(), ag.index.max()).astype(int).map(ag).fillna(0)
    cur["proj_pts_gs"] = cur["proj_pts_gs_raw"] + 0.5 * cur["age_step"]
    hist = d[d["season"].isin([season - 2, season - 1, season])].groupby("mlbam_id")["GS"].mean()
    base_gs = 0.6 * cur["GS"] + 0.4 * cur["mlbam_id"].map(hist).fillna(cur["GS"])
    dur = (1 - 0.001 * cur["il_days_3yr"].clip(0, 250)) * np.where(cur["age"] >= 34, 0.93, 1.0)
    cur["proj_GS"] = (base_gs * dur).clip(12, 32).round(0); cur["durability"] = dur.round(3)
    cur["proj_pts"] = (cur["proj_pts_gs"] * cur["proj_GS"]).round(0)
    cur["proj_rank"] = cur["proj_pts"].rank(ascending=False).astype(int)
    cur["proj_rank_gs"] = cur["proj_pts_gs"].rank(ascending=False).astype(int)
    return cur, ag.reset_index().rename(columns={"age_i": "age", "delta": "yoy_delta"})

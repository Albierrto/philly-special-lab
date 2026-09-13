"""Prospect status and minor-league performance.

Sources
  * MLB Pipeline Top 100 (current list, embedded JSON on mlb.com/prospects/stats/top-prospects) -> pedigree
  * MLB Stats API minor-league season hitting (AAA=11, AA=12, A+=13, A=14) -> age-vs-level production
  * Career MLB at-bats (from our own player-season table) -> rookie eligibility (<= 130 AB)

Prospect Arrival Score (PAS): a 0-100 read on how ready/impactful a bat with < 300 MLB PA looks for next season.
    PAS = 35 * pedigree + 25 * level_age + 25 * milb_bat + 15 * mlb_sample
    pedigree  : Pipeline rank -> 1 - (rank-1)/100 ; unranked = 0.15
    level_age : how young for the highest level with 100+ PA (AAA avg age 26.5, AA 24.5, A+ 22.5, A 21.0):
                clip(0.5 + (avg_level_age - age) * 0.15, 0, 1), scaled by level (AAA 1.0, AA 0.85, A+ 0.6, A 0.4)
    milb_bat  : league-weighted production at highest level: z of (ISO, BB%-K%, SB rate, OPS) averaged, mapped to 0-1
    mlb_sample: if any MLB PA: percentile of xwOBA-ish quality (xwoba, avg_best_speed, avg_swing_speed) else 0.5
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import requests

from .config import DATA
from .names import key

_S = requests.Session(); _S.headers.update({"User-Agent": "Mozilla/5.0"})
LEVELS = {11: "AAA", 12: "AA", 13: "A+", 14: "A"}
LEVEL_AGE = {"AAA": 26.5, "AA": 24.5, "A+": 22.5, "A": 21.0}
LEVEL_W = {"AAA": 1.0, "AA": 0.85, "A+": 0.6, "A": 0.4}


def pipeline_top100(refresh=False) -> pd.DataFrame:
    p = DATA / "prospects" / "pipeline_current.json"
    if not p.exists() or refresh:
        h = _S.get("https://www.mlb.com/prospects/stats/top-prospects", timeout=90).text
        s = h.find('[{"name":"'); depth = 0; j = s
        while j < len(h):
            depth += h[j] == "["; depth -= h[j] == "]"
            if depth == 0: break
            j += 1
        p.parent.mkdir(parents=True, exist_ok=True); p.write_text(h[s:j + 1])
    d = json.loads(p.read_text())
    rows = [dict(mlbam_id=x["playerId"], prospect_name=x["name"], pipeline_rank=x["rank"], prospect_pos=x.get("position"),
                 prospect_age=x.get("age"), prospect_level=x.get("sportAbbrev"), prospect_org=x.get("team"))
            for x in d if x.get("position") not in ("RHP", "LHP", "P")]
    df = pd.DataFrame(rows); df["nkey"] = df["prospect_name"].apply(key)
    return df


def milb_hitting(year: int, sport: int, refresh=False) -> list[dict]:
    p = DATA / "milb" / f"hitting_{LEVELS[sport]}_{year}.json"
    if p.exists() and not refresh:
        return json.loads(p.read_text())
    out, offset = [], 0
    while True:
        u = (f"https://statsapi.mlb.com/api/v1/stats?stats=season&group=hitting&season={year}&sportId={sport}"
             f"&playerPool=all&limit=2000&offset={offset}&hydrate=person")
        st = _S.get(u, timeout=120).json()["stats"][0]; sp = st.get("splits", []); out += sp; offset += len(sp)
        if not sp or offset >= st.get("totalSplits", 0): break
    p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(out))
    return out


def milb_table(years=(2024, 2025, 2026)) -> pd.DataFrame:
    rows = []
    for y in years:
        for sport, lvl in LEVELS.items():
            for s in milb_hitting(y, sport):
                p, st = s["player"], s["stat"]
                pa = st.get("plateAppearances", 0) or 0
                if pa < 60: continue
                ab = st.get("atBats", 0) or 1
                h = st.get("hits", 0); hr = st.get("homeRuns", 0); d2 = st.get("doubles", 0); d3 = st.get("triples", 0)
                rows.append(dict(mlbam_id=p["id"], name=p["fullName"], birth_date=p.get("birthDate"), season=y, level=lvl,
                                 milb_PA=pa, milb_AVG=h / ab, milb_OBP=float(st.get("obp", 0) or 0), milb_SLG=float(st.get("slg", 0) or 0),
                                 milb_OPS=float(st.get("ops", 0) or 0), milb_ISO=(d2 + 2 * d3 + 3 * hr) / ab,
                                 milb_K=st.get("strikeOuts", 0) / pa, milb_BB=st.get("baseOnBalls", 0) / pa,
                                 milb_HR=hr, milb_SB=st.get("stolenBases", 0), milb_SB600=st.get("stolenBases", 0) / pa * 600))
    df = pd.DataFrame(rows)
    df["age"] = ((pd.to_datetime(f"{df['season'].iloc[0]}-06-30") - pd.to_datetime(df["birth_date"])).dt.days / 365.25)
    df["age"] = df.apply(lambda r: (pd.Timestamp(f"{r.season}-06-30") - pd.Timestamp(r.birth_date)).days / 365.25 if pd.notna(r.birth_date) else np.nan, axis=1).round(1)
    df["nkey"] = df["name"].apply(key)
    return df


def _z(s): return (s - s.mean()) / (s.std(ddof=0) or 1)


def milb_summary(year: int) -> pd.DataFrame:
    """Highest level with 100+ PA in `year`, with a league-weighted bat score (z within level)."""
    t = milb_table((year,))
    t = t[t["milb_PA"] >= 100].copy()
    order = {"AAA": 4, "AA": 3, "A+": 2, "A": 1}
    t["lvl_order"] = t["level"].map(order)
    # z-scores within level for a points-league bat: ISO, OBP, BB-K, SB rate
    parts = []
    for lvl, d in t.groupby("level"):
        d = d.copy()
        d["milb_bat_z"] = (_z(d["milb_ISO"]) + _z(d["milb_OBP"]) + _z(d["milb_BB"] - d["milb_K"]) + 0.5 * _z(d["milb_SB600"])) / 3.5
        parts.append(d)
    t = pd.concat(parts)
    top = t.sort_values(["mlbam_id", "lvl_order", "milb_PA"], ascending=[True, False, False]).drop_duplicates("mlbam_id")
    top["level_age_score"] = ((0.5 + (top["level"].map(LEVEL_AGE) - top["age"]) * 0.15).clip(0, 1) * top["level"].map(LEVEL_W))
    top["milb_bat_score"] = (0.5 + 0.2 * top["milb_bat_z"]).clip(0, 1)
    return top[["mlbam_id", "nkey", "name", "age", "level", "milb_PA", "milb_AVG", "milb_OBP", "milb_SLG", "milb_OPS", "milb_ISO",
                "milb_K", "milb_BB", "milb_HR", "milb_SB", "milb_bat_z", "level_age_score", "milb_bat_score"]].rename(
        columns={"age": "milb_age", "level": "milb_level"})


def prospect_features(ps: pd.DataFrame, season: int) -> pd.DataFrame:
    """Rookie status + pedigree + PAS for every hitter in `season` plus unrostered top prospects."""
    career_ab = ps[ps["season"] <= season].groupby("mlbam_id")["AB"].sum().rename("career_AB")
    debut_pa = ps[ps["season"] <= season].groupby("mlbam_id")["PA"].sum().rename("career_PA")
    cur = ps[ps["season"] == season][["mlbam_id", "nkey", "name", "age", "PA", "xwoba", "avg_best_speed", "avg_swing_speed", "pts_pa"]].copy()
    top = pipeline_top100()
    ms = milb_summary(season)
    # union of current MLB hitters, ranked prospects, and MiLB bats
    base = cur.merge(top.drop(columns=["nkey"]), on="mlbam_id", how="outer")
    base = base.merge(ms.drop(columns=["nkey", "name"]), on="mlbam_id", how="left")
    base["name"] = base["name"].fillna(base["prospect_name"])
    base["nkey"] = base["name"].apply(key)
    base = base.merge(career_ab, on="mlbam_id", how="left").merge(debut_pa, on="mlbam_id", how="left")
    base["career_AB"] = base["career_AB"].fillna(0); base["career_PA"] = base["career_PA"].fillna(0)
    base["age"] = base["age"].fillna(base["prospect_age"]).fillna(base["milb_age"])
    base["rookie_eligible"] = base["career_AB"] <= 130
    base["pedigree"] = np.where(base["pipeline_rank"].notna(), 1 - (base["pipeline_rank"].fillna(100) - 1) / 100, 0.15)
    # MLB sample quality percentile (only meaningful with some PA)
    q = cur.dropna(subset=["xwoba"])
    pct = lambda s, v: float((s < v).mean()) if pd.notna(v) else np.nan
    base["mlb_sample"] = base.apply(lambda r: np.nanmean([pct(q["xwoba"], r.get("xwoba")), pct(q["avg_best_speed"], r.get("avg_best_speed")),
                                                          pct(q["avg_swing_speed"], r.get("avg_swing_speed"))]) if pd.notna(r.get("xwoba")) else np.nan, axis=1)
    base["PAS"] = (35 * base["pedigree"] + 25 * base["level_age_score"].fillna(0.3) + 25 * base["milb_bat_score"].fillna(0.4)
                   + 15 * base["mlb_sample"].fillna(0.5)).round(1)
    base["prospect_status"] = np.select(
        [base["pipeline_rank"].notna() & (base["career_PA"] < 300), base["rookie_eligible"] & (base["career_PA"] > 0),
         base["pipeline_rank"].notna(), (base["career_PA"] < 1200) & (base["age"] <= 25)],
        ["top-100 prospect", "rookie", "graduating prospect", "young / establishing"], default="established")
    return base

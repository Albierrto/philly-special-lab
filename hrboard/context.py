"""Game context: schedule (venue, recorded or forecast weather, scores), Savant park factors by batter side, venues."""
from __future__ import annotations
import json, re, time
from pathlib import Path
import numpy as np
import pandas as pd
import requests

from breakout.config import DATA

API = "https://statsapi.mlb.com/api/v1"
CTX = DATA / "hrboard" / "context"
_S = requests.Session(); _S.headers.update({"User-Agent": "Mozilla/5.0"})


def _get(url, **params):
    for i in range(4):
        try:
            r = _S.get(url, params=params, timeout=90); r.raise_for_status(); return r.json()
        except Exception:
            time.sleep(1 + 2 * i)
    return {}


WIND_RE = re.compile(r"(\d+)\s*mph,?\s*(.*)", re.I)
# direction text MLB records at first pitch -> component blowing out toward the outfield (+1 out to CF, -1 in from CF)
WIND_DIR = {"out to cf": 1.0, "out to lf": 0.7, "out to rf": 0.7, "in from cf": -1.0, "in from lf": -0.7, "in from rf": -0.7,
            "l to r": 0.0, "r to l": 0.0, "calm": 0.0, "varies": 0.0, "none": 0.0}
INDOOR = ("dome", "roof closed")


def parse_weather(w: dict | None) -> dict:
    w = w or {}
    cond = str(w.get("condition") or "").strip()
    try: temp = float(w.get("temp"))
    except (TypeError, ValueError): temp = np.nan
    mph, comp = np.nan, np.nan
    m = WIND_RE.match(str(w.get("wind") or ""))
    if m:
        mph = float(m.group(1)); comp = WIND_DIR.get(m.group(2).strip().lower(), 0.0)
    indoor = cond.lower() in INDOOR
    return dict(wx_cond=cond or None, temp_f=temp, wind_mph=mph, wind_out=(0.0 if indoor else (mph * comp if mph == mph else np.nan)), indoor=indoor)


def schedule(start: str, end: str, hydrate: str = "weather,venue,probablePitcher,team") -> pd.DataFrame:
    j = _get(f"{API}/schedule", sportId=1, startDate=start, endDate=end, hydrate=hydrate, gameType="R")
    rows = []
    for d in j.get("dates", []):
        for g in d["games"]:
            if g.get("gameType") != "R": continue
            h, a = g["teams"]["home"], g["teams"]["away"]
            rows.append(dict(game_pk=g["gamePk"], date=g.get("officialDate") or d["date"], game_time=g["gameDate"], status=g["status"]["detailedState"],
                             abstract=g["status"].get("abstractGameState"), venue_id=g["venue"]["id"], venue=g["venue"]["name"],
                             home_id=h["team"]["id"], away_id=a["team"]["id"], home_name=h["team"]["name"], away_name=a["team"]["name"],
                             home_abbr=h["team"].get("abbreviation"), away_abbr=a["team"].get("abbreviation"),
                             home_runs=h.get("score"), away_runs=a.get("score"), day_night=g.get("dayNight"), dh=g.get("doubleHeader"), game_no=g.get("gameNumber"),
                             home_sp=(h.get("probablePitcher") or {}).get("id"), away_sp=(a.get("probablePitcher") or {}).get("id"),
                             home_sp_name=(h.get("probablePitcher") or {}).get("fullName"), away_sp_name=(a.get("probablePitcher") or {}).get("fullName"),
                             **parse_weather(g.get("weather"))))
    return pd.DataFrame(rows)


def season_schedule(season: int, refresh: bool = False) -> pd.DataFrame:
    CTX.mkdir(parents=True, exist_ok=True)
    f = CTX / f"schedule_{season}.parquet"
    if f.exists() and not refresh:
        return pd.read_parquet(f)
    parts = []
    for m0, m1 in [("03-15", "04-30"), ("05-01", "06-15"), ("06-16", "07-31"), ("08-01", "09-15"), ("09-16", "10-05")]:
        parts.append(schedule(f"{season}-{m0}", f"{season}-{m1}"))
    d = pd.concat(parts, ignore_index=True).drop_duplicates("game_pk", keep="last")
    d.to_parquet(f, index=False)
    return d


def park_factors(year: int, rolling: int = 3) -> pd.DataFrame:
    """Savant Statcast park factors (100 = neutral) for each venue and batter side, rolling `rolling` years ending `year`."""
    CTX.mkdir(parents=True, exist_ok=True)
    f = CTX / f"parks_{year}_r{rolling}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    out = []
    for side in ("L", "R"):
        r = _S.get(f"https://baseballsavant.mlb.com/leaderboard/statcast-park-factors?type=year&year={year}&batSide={side}"
                   f"&stat=index_wOBA&condition=All&rolling={rolling}", timeout=90)
        m = re.search(r"var data = (\[.*?\]);", r.text, re.S)
        if not m: continue
        for d in json.loads(m.group(1)):
            out.append(dict(venue_id=int(d["venue_id"]), stand=side, n_pa=int(d.get("n_pa") or 0),
                            **{f"pf_{k}": float(d[f"index_{k}"]) / 100 for k in ("hr", "1b", "2b", "3b", "so", "bb", "runs", "woba")}))
    d = pd.DataFrame(out)
    if len(d): d.to_parquet(f, index=False)
    return d


def venues(season: int = 2026) -> pd.DataFrame:
    j = _get(f"{API}/venues", hydrate="location,fieldInfo,timezone", season=season, sportId=1)
    rows = []
    for v in j.get("venues", []):
        loc = v.get("location", {}); co = loc.get("defaultCoordinates", {}); fi = v.get("fieldInfo", {}); tz = v.get("timeZone", {})
        rows.append(dict(venue_id=v["id"], venue=v["name"], lat=co.get("latitude"), lon=co.get("longitude"), azimuth=loc.get("azimuthAngle"),
                         elevation=loc.get("elevation"), roof=fi.get("roofType"), tz=tz.get("id"), city=loc.get("city")))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    for s in (2023, 2024, 2025, 2026):
        x = season_schedule(s, refresh=(s == 2026)); print(s, len(x), x["temp_f"].notna().mean().round(3), x["wind_out"].notna().mean().round(3))
    for y in (2022, 2023, 2024, 2025, 2026):
        p = park_factors(y); print("parks", y, len(p))

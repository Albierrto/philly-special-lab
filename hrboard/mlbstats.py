"""Season stat lines for every hitter and pitcher from the MLB Stats API, in one call per group. Used by the postseason
board for the counting stats the plate-appearance cache does not carry (runs, RBI, steals, saves)."""
from __future__ import annotations
import json, time, urllib.request
import pandas as pd

API = "https://statsapi.mlb.com/api/v1"


def _get(url):
    for i in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as r: return json.load(r)
        except Exception:
            time.sleep(2 + 2 * i)
    return {}


def ip_float(s):
    if s in (None, ""): return 0.0
    w, _, f = str(s).partition("."); return int(w) + int(f or 0) / 3


def season_hitting(season: int) -> pd.DataFrame:
    rows = []
    for offset in (0, 1000, 2000):
        j = _get(f"{API}/stats?stats=season&group=hitting&season={season}&sportId=1&limit=1000&offset={offset}&playerPool=ALL")
        splits = (j.get("stats") or [{}])[0].get("splits", [])
        for s in splits:
            st = s["stat"]; p = s["player"]; t = s.get("team") or {}
            rows.append(dict(pid=p["id"], name=p["fullName"], team=t.get("abbreviation") or t.get("name"), pos=(s.get("position") or {}).get("abbreviation"),
                             G=st.get("gamesPlayed", 0), PA=st.get("plateAppearances", 0), AB=st.get("atBats", 0), H=st.get("hits", 0),
                             X2=st.get("doubles", 0), X3=st.get("triples", 0), HR=st.get("homeRuns", 0), R=st.get("runs", 0), RBI=st.get("rbi", 0),
                             SB=st.get("stolenBases", 0), BB=st.get("baseOnBalls", 0), HBP=st.get("hitByPitch", 0), SO=st.get("strikeOuts", 0)))
        if len(splits) < 1000: break
    d = pd.DataFrame(rows).drop_duplicates("pid", keep="last")
    return d


def season_pitching(season: int) -> pd.DataFrame:
    rows = []
    for offset in (0, 1000, 2000):
        j = _get(f"{API}/stats?stats=season&group=pitching&season={season}&sportId=1&limit=1000&offset={offset}&playerPool=ALL")
        splits = (j.get("stats") or [{}])[0].get("splits", [])
        for s in splits:
            st = s["stat"]; p = s["player"]; t = s.get("team") or {}
            rows.append(dict(pid=p["id"], name=p["fullName"], team=t.get("abbreviation") or t.get("name"),
                             G=st.get("gamesPlayed", 0), GS=st.get("gamesStarted", 0), IP=ip_float(st.get("inningsPitched")), K=st.get("strikeOuts", 0),
                             W=st.get("wins", 0), L=st.get("losses", 0), SV=st.get("saves", 0), HLD=st.get("holds", 0), ER=st.get("earnedRuns", 0),
                             BF=st.get("battersFaced", 0), H=st.get("hits", 0), BB=st.get("baseOnBalls", 0), HR=st.get("homeRuns", 0)))
        if len(splits) < 1000: break
    return pd.DataFrame(rows).drop_duplicates("pid", keep="last")


def season_fielding(season: int, min_games: int = 10) -> dict:
    """Positions each player has played `min_games` or more games at this season: {pid: ["OF", "1B", ...]}."""
    out = {}
    for offset in (0, 1000, 2000, 3000):
        j = _get(f"{API}/stats?stats=season&group=fielding&season={season}&sportId=1&limit=1000&offset={offset}&playerPool=ALL")
        splits = (j.get("stats") or [{}])[0].get("splits", [])
        for s in splits:
            g = (s.get("stat") or {}).get("gamesPlayed", 0); pos = (s.get("position") or {}).get("abbreviation")
            if pos and g >= min_games:
                slot = {"LF": "OF", "CF": "OF", "RF": "OF"}.get(pos, pos)
                out.setdefault(s["player"]["id"], set()).add(slot)
        if len(splits) < 1000: break
    return {k: sorted(v) for k, v in out.items()}


HOLDEM_H = dict(X1=1, X2=2, X3=3, HR=4, R=1, RBI=1, SB=1, BB=1, HBP=1, OUT=-0.25)
HOLDEM_P = dict(IP=1, K=1, W=4, SV=4, ER=-1)


def holdem_hit_points(d: pd.DataFrame) -> pd.Series:
    x1 = d["H"] - d["X2"] - d["X3"] - d["HR"]
    return (x1 + 2 * d["X2"] + 3 * d["X3"] + 4 * d["HR"] + d["R"] + d["RBI"] + d["SB"] + d["BB"] + d["HBP"] - 0.25 * (d["AB"] - d["H"]))


def holdem_pit_points(d: pd.DataFrame) -> pd.Series:
    return d["IP"] + d["K"] + 4 * d["W"] + 4 * d["SV"] - d["ER"]

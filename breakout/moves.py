"""Offseason moves: MLB's transaction feed, cut down to the players this league cares about (every rostered player, the
top of the next draft pool, and the prospects on the minors board). Trades, free-agent signings, players electing free
agency, waiver claims, designations, releases, retirements and injured-list moves, each with MLB's own sentence and, when
a player changes clubs, his old and new home park so the page can say what the move does to his numbers.

    python -m breakout.moves            (runs inside the rehab step of the refresh, so it never fails the build)

Writes data/news/moves_<season>.json: {"asof", "since", "teams": {abbr: [name, venue_id]}, "moves": [...]}, newest first.
"""
from __future__ import annotations
import datetime as dt, json, sys
import pandas as pd
import requests

from . import config as C
from .config import league_today
from .names import key

API = "https://statsapi.mlb.com/api/v1"
DIR = C.DATA / "news"
DAYS = 75                        # how far back the feed reaches; the winter's moves stay on the page into spring
KEEP = {"TR": "trade", "SFA": "signed", "DFA": "free agent", "CLW": "claimed", "DES": "designated", "OUT": "outrighted",
        "REL": "released", "RET": "retired", "SC": "injury", "R5": "Rule 5", "SE": "selected"}
_S = requests.Session(); _S.headers.update({"User-Agent": "Mozilla/5.0"})


def _get(path: str, **params) -> dict:
    for i in range(3):
        try:
            r = _S.get(f"{API}/{path}", params=params, timeout=60)
            if r.ok: return r.json()
        except Exception:
            pass
    return {}


def interest(season: int) -> dict[int, str]:
    """MLBAM id -> why he matters: 'rostered', 'pool' (top of the next draft) or 'prospect'."""
    out: dict[int, str] = {}
    for f, n in ((C.OUT / "v3" / "hitter_projections_2027.csv", 300), (C.OUT / "v3" / "pitcher_projections_2027.csv", 150)):
        if f.exists():
            d = pd.read_csv(f, usecols=lambda c: c in ("mlbam_id", "proj_pts", "owner"))
            for r in d.sort_values("proj_pts", ascending=False).head(n).itertuples(): out[int(r.mlbam_id)] = "pool"
            if "owner" in d:
                for r in d[d["owner"].notna() & (d["owner"].astype(str) != "FA")].itertuples(): out[int(r.mlbam_id)] = "rostered"
    pv = C.OUT / "v3" / "prospect_values.csv"
    if pv.exists():
        d = pd.read_csv(pv, usecols=lambda c: c in ("mlbam_id", "eligible"))
        for r in d.itertuples():
            out.setdefault(int(r.mlbam_id), "prospect")
    ro = C.DATA / "fantrax" / f"rosters_{season}.csv"
    if ro.exists():            # rostered by name: a stashed prospect or an injured man the boards do not carry
        names = {key(n) for n in pd.read_csv(ro)["player"].dropna()}
        for f in (C.OUT / "v3" / "hitter_projections_2027.csv", C.OUT / "v3" / "pitcher_projections_2027.csv", C.OUT / "v3" / "prospect_values.csv"):
            if f.exists():
                d = pd.read_csv(f, usecols=lambda c: c in ("mlbam_id", "name"))
                for r in d.itertuples():
                    if key(r.name) in names: out[int(r.mlbam_id)] = "rostered"
    return out


def build(season: int) -> dict:
    today = league_today(); since = today - dt.timedelta(days=DAYS)
    who = interest(season)
    tm = _get("teams", sportId=1, season=season)
    teams = {t["id"]: (t.get("abbreviation"), t.get("name"), (t.get("venue") or {}).get("id")) for t in tm.get("teams", [])}
    j = _get("transactions", sportId=1, startDate=since.isoformat(), endDate=today.isoformat())
    moves = []
    for t in j.get("transactions", []):
        code = t.get("typeCode"); pid = (t.get("person") or {}).get("id")
        if code not in KEEP or pid not in who: continue
        desc = t.get("description") or ""
        low = desc.lower()
        if code == "SC" and not any(w in low for w in ("injured list", "activated", "reinstated", "retired", "restricted", "suspended")): continue
        fr = teams.get((t.get("fromTeam") or {}).get("id")); to = teams.get((t.get("toTeam") or {}).get("id"))
        moves.append(dict(date=t.get("date") or t.get("effectiveDate"), id=int(pid), name=(t.get("person") or {}).get("fullName"),
                          code=code, kind=KEEP[code], why=who[pid], desc=desc,
                          fr=fr[0] if fr else None, to=to[0] if to else None,
                          fv=fr[2] if fr else None, tv=to[2] if to else None))
    moves.sort(key=lambda m: (m["date"] or "", m["id"]), reverse=True)
    seen, out = set(), []
    for m in moves:                                  # one line per player per move and day
        k = (m["id"], m["code"], m["date"])
        if k in seen: continue
        seen.add(k); out.append(m)
    return {"asof": today.isoformat(), "since": since.isoformat(), "moves": out[:600],
            "teams": {a: [n, v] for (a, n, v) in teams.values() if a}}


def write(season: int) -> dict:
    d = build(season)
    DIR.mkdir(parents=True, exist_ok=True)
    (DIR / f"moves_{season}.json").write_text(json.dumps(d, separators=(",", ":")))
    n = len(d["moves"]); r = sum(1 for m in d["moves"] if m["why"] == "rostered")
    print(f"moves: {n} since {d['since']} ({r} involving rostered players)")
    return d


if __name__ == "__main__":
    sys.exit(0 if write(C.CURRENT_SEASON) else 1)

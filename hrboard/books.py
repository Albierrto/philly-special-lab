"""Game lines from ten sportsbooks and exchanges (The Odds API), on a strict credit budget.

The key lives only in the ODDS_API_KEY repository secret and is never written to the site or the logs. One request
covers every upcoming game for ten books and three markets (moneyline, run line, total) and costs 3 credits. The free
plan has 500 a month, so the build calls it at most a few times a day, spread over the hours that matter, and never
after the credits for the rest of the month are spoken for. Without a key this step prints one line and does nothing.

Snapshots: data/hrboard/books/<date>.json (a game's prices stop updating once it has started, so the page always shows
pre-game lines next to pre-game forecasts). Usage: data/hrboard/books/usage.json.
"""
from __future__ import annotations
import calendar, json, os
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from breakout.config import DATA
from . import odds as O, context as C

DIR = DATA / "hrboard" / "books"
USAGE = DIR / "usage.json"
ET = ZoneInfo("America/New_York")
COST = 3
MIN_GAP = timedelta(minutes=150)
WINDOW_ET = (9, 21)          # first call after 9 am ET, last before 9 pm ET
ALIASES = {"Oakland Athletics": "ATH", "Athletics": "ATH", "Arizona Diamondbacks": "AZ", "Chicago White Sox": "CWS",
           "Washington Nationals": "WSH", "Cleveland Guardians": "CLE"}


def _load(f, default):
    try: return json.loads(f.read_text())
    except (OSError, ValueError): return default


def _key() -> str:
    return (os.environ.get("ODDS_API_KEY") or "").strip()


def status() -> dict:
    u = _load(USAGE, {})
    return dict(configured=bool(_key()) or bool(u.get("last_call")), remaining=u.get("remaining"), used=u.get("used"),
                last_call=u.get("last_call"), last_status=u.get("status"))


def _allowed_today(u: dict, today: date) -> bool:
    rem = u.get("remaining")
    if rem is None: return True
    try: rem = float(rem)
    except ValueError: return True
    if rem < COST: return False
    days_left = calendar.monthrange(today.year, today.month)[1] - today.day + 1
    per_day = int(rem // COST // days_left)
    return u.get("calls", {}).get(today.isoformat(), 0) < max(per_day, 1 if rem >= COST * days_left else 0)


def maybe_fetch(today: str, force: bool = False) -> bool:
    key = _key()
    if not key:
        print("  odds api: no ODDS_API_KEY secret; multi-book lines skipped")
        return False
    DIR.mkdir(parents=True, exist_ok=True)
    u = _load(USAGE, {})
    now = datetime.now(timezone.utc); et = now.astimezone(ET)
    if not force:
        if not (WINDOW_ET[0] <= et.hour < WINDOW_ET[1]):
            print("  odds api: outside the call window"); return False
        last = u.get("last_call")
        if last and now - datetime.fromisoformat(last) < MIN_GAP:
            print("  odds api: called recently, reusing the last snapshot"); return False
        if not _allowed_today(u, et.date()):
            print(f"  odds api: daily budget used (credits left {u.get('remaining')})"); return False
    days = [today, (date.fromisoformat(today) + timedelta(days=1)).isoformat()]
    scheds = {d: C.schedule(d, d) for d in days}
    if not any(len(s) and (s["abstract"] == "Preview").any() for s in scheds.values()):
        print("  odds api: nothing left to price"); return False
    events, usage = O.odds_api_lines(key)
    u.update(remaining=usage.get("remaining"), used=usage.get("used"), status=usage.get("status"), last_call=now.isoformat(timespec="seconds"))
    u.setdefault("calls", {})[et.date().isoformat()] = u.get("calls", {}).get(et.date().isoformat(), 0) + 1
    u["calls"] = {k: v for k, v in u["calls"].items() if k >= (et.date() - timedelta(days=40)).isoformat()}
    USAGE.write_text(json.dumps(u, indent=1))
    if usage.get("status") != 200:
        print(f"  odds api: HTTP {usage.get('status')} (credits left {usage.get('remaining')})"); return False
    for d, sch in scheds.items():
        if not len(sch): continue
        O.TEAM_FULL.update({n: a for n, a in zip(sch["home_name"], sch["home_abbr"])})
        O.TEAM_FULL.update({n: a for n, a in zip(sch["away_name"], sch["away_abbr"])})
        O.TEAM_FULL.update(ALIASES)
        g = sch.rename(columns={"game_time": "time"})
        by = O.odds_api_by_game(events, g)
        f = DIR / f"{d}.json"
        snap = _load(f, {"games": {}})
        start = dict(zip(g["game_pk"].astype(int), g["time"]))
        n = 0
        for pk, books in by.items():
            if datetime.fromisoformat(str(start[pk]).replace("Z", "+00:00")) <= now:
                continue                                    # live odds would not be comparable to the pre-game forecast
            snap["games"][str(pk)] = books; n += 1
        if n:
            snap["as_of"] = now.isoformat(timespec="seconds")
            f.write_text(json.dumps(snap, separators=(",", ":")))
        print(f"  odds api: {d} {n} games priced; credits left {usage.get('remaining')}")
    cut = (date.fromisoformat(today) - timedelta(days=2)).isoformat()
    for old in DIR.glob("20*.json"):
        if old.stem < cut: old.unlink()
    return True


def snapshot(day: str, games=None) -> dict:
    return _load(DIR / f"{day}.json", {})


if __name__ == "__main__":
    import sys
    maybe_fetch(sys.argv[1] if len(sys.argv) > 1 else datetime.now(ET).date().isoformat(), force="--force" in sys.argv)

"""Who is on a rehab assignment right now, how the outings have gone, and when the window runs out.

This comes from the same MLB Stats API feed the IL history already uses, which is the primary record rather than a
news site writing about it: every assignment is a transaction ("Kansas City Royals sent 3B Maikel Garcia on a rehab
assignment to Omaha Storm Chasers"), and the outings themselves are that player's game log at the affiliate. So the
board is not scraped prose, it is the same rows the team filed, and it is current the moment a club files one.

Three things make it useful rather than just a list:

  the clock      MLB caps a rehab assignment at 20 days for a position player and 30 for a pitcher. That is a hard
                 deadline, so "day 6 of 20" says more about when a man is back than any report does.
  the outings    An arm building up shows it in the pitch count, not the line: 1 IP on 10 pitches is a man two
                 outings from ready. A bat shows it in at-bats per game climbing to nine innings' worth.
  the ending     A rehab ends when the club activates, recalls or options him, and each of those is a transaction
                 too, so an assignment closes itself without anyone deciding it looks stale.

Cost: one transactions call for the recent window, five cached calls to map affiliates to levels, and one game-log
call per player actually out there. Nothing is scraped and nothing needs a key.
"""
from __future__ import annotations
import datetime as dt
import json
import re

import pandas as pd
import requests

from .config import DATA
from .injuries import transactions

_S = requests.Session(); _S.headers.update({"User-Agent": "Mozilla/5.0"})
DIR = DATA / "rehab"
LEVEL = {11: "AAA", 12: "AA", 13: "A+", 14: "A", 16: "Rk"}
PITCHER_POS = {"P", "SP", "RP", "RHP", "LHP"}
CAP = {"pitcher": 30, "hitter": 20}          # MLB's limit on the length of a rehab assignment
STINT_MAX = 75                               # older than this and it is history, not a live assignment

REHAB_RE = re.compile(r"\bsent\s+([A-Z0-9]{1,3})\s+(.+?)\s+on a rehab assignment to\s+(.+?)\.?$", re.I)
BACK_RE = re.compile(r"\b(activated|recalled|selected the contract of|optioned|released|designated|reassigned|sent .* outright)\b", re.I)
IL_RE = re.compile(r"\bplaced\b.*\b(\d+)-day injured list", re.I)
STALE_OUTING = 10                            # no game in this many days and the clock has stopped, whatever it says


def _recent(season: int, days: int = STINT_MAX + 15) -> list[dict]:
    """The last few weeks of transactions, fetched fresh every build.

    injuries.transactions() caches the whole season once and never looks again, which is right for IL history and
    useless for this: an assignment filed this morning has to show up this morning. So the season file is the base
    and a short rolling window is laid on top of it.
    """
    end = dt.date.today()
    start = end - dt.timedelta(days=days)
    u = (f"https://statsapi.mlb.com/api/v1/transactions?startDate={start:%Y-%m-%d}"
         f"&endDate={end:%Y-%m-%d}&sportId=1")
    try:
        return _S.get(u, timeout=60).json().get("transactions", [])
    except Exception:
        return []


def affiliates(refresh: bool = False) -> dict[int, dict]:
    """Minor-league team id -> level, cached, because it changes about once a decade."""
    p = DIR / "affiliates.json"
    if p.exists() and not refresh:
        try:
            return {int(k): v for k, v in json.loads(p.read_text()).items()}
        except Exception:
            pass
    out = {}
    for sid in LEVEL:
        try:
            r = _S.get(f"https://statsapi.mlb.com/api/v1/teams?sportId={sid}", timeout=30).json()
        except Exception:
            continue
        for t in r.get("teams", []):
            out[int(t["id"])] = {"sportId": sid, "level": LEVEL[sid], "name": t.get("name")}
    if out:
        DIR.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out))
    return out


def outings(pid: int, sport_ids, season: int, pitcher: bool, since: str) -> list[dict]:
    """His game log at the affiliate since the assignment, which is what the rehab actually looked like."""
    grp = "pitching" if pitcher else "hitting"
    rows = []
    for sid in dict.fromkeys(sport_ids):                     # usually one level, two if he moved up
        u = (f"https://statsapi.mlb.com/api/v1/people/{pid}/stats?stats=gameLog&group={grp}"
             f"&season={season}&sportId={sid}")
        try:
            blocks = _S.get(u, timeout=30).json().get("stats", [])
        except Exception:
            continue
        for b in blocks:
            for s in b.get("splits", []):
                d = s.get("date")
                if not d or d < since:
                    continue
                st = s.get("stat", {})
                row = {"date": d, "team": (s.get("team") or {}).get("name"), "level": LEVEL.get(sid, "")}
                if pitcher:
                    row |= {"ip": st.get("inningsPitched"), "h": st.get("hits"), "er": st.get("earnedRuns"),
                            "k": st.get("strikeOuts"), "bb": st.get("baseOnBalls"), "pitches": st.get("numberOfPitches"),
                            "hr": st.get("homeRuns")}
                else:
                    row |= {"ab": st.get("atBats"), "h": st.get("hits"), "hr": st.get("homeRuns"),
                            "bb": st.get("baseOnBalls"), "k": st.get("strikeOuts"), "rbi": st.get("rbi"),
                            "pa": st.get("plateAppearances")}
                rows.append(row)
    return sorted(rows, key=lambda r: r["date"])


def _mlb_return(pid: int, season: int, pitcher: bool, since: str) -> str | None:
    """The date he next appeared in a major-league game, which closes an assignment no transaction closed."""
    grp = "pitching" if pitcher else "hitting"
    u = f"https://statsapi.mlb.com/api/v1/people/{pid}/stats?stats=gameLog&group={grp}&season={season}"
    try:
        blocks = _S.get(u, timeout=30).json().get("stats", [])
    except Exception:
        return None
    dates = [s.get("date") for b in blocks for s in b.get("splits", []) if s.get("date") and s["date"] >= since]
    return min(dates) if dates else None


def statuses(refresh: bool = True) -> dict[int, str]:
    """Every 40-man player's current roster status, which is the only thing that actually says a rehab is over.

    A club does not always file something a regex can read when it ends an assignment: it can reassign him to the
    minors, option him, or simply activate him in a way the transaction text words differently. But the roster
    status is unambiguous and it is thirty calls that take under two seconds. D7/D10/D15/D60 means still on the
    injured list; anything else means he is not rehabbing any more, whatever the last transaction said.
    """
    p = DIR / "status.json"
    if not refresh and p.exists():
        try:
            return {int(k): v for k, v in json.loads(p.read_text()).items()}
        except Exception:
            pass
    out = {}
    try:
        teams = _S.get("https://statsapi.mlb.com/api/v1/teams?sportId=1", timeout=30).json().get("teams", [])
    except Exception:
        return out
    for t in teams:
        try:
            r = _S.get(f"https://statsapi.mlb.com/api/v1/teams/{t['id']}/roster?rosterType=40Man", timeout=30).json()
        except Exception:
            continue
        for pl in r.get("roster", []):
            out[int(pl["person"]["id"])] = ((pl.get("status") or {}).get("code") or "")
    if out:
        DIR.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(out))
    return out


def build(season: int, with_outings: bool = True) -> dict:
    """The live board: open assignments with their outings, plus who came back in the last fortnight."""
    today = dt.date.today()
    seen, tx = set(), []
    for t in list(transactions(season)) + _recent(season):
        i = t.get("id")
        if i in seen:
            continue
        seen.add(i); tx.append(t)
    tx.sort(key=lambda t: (t.get("date") or "", t.get("id") or 0))

    aff = affiliates()
    assigns, enders, placements, reasons = {}, {}, {}, {}
    for t in tx:
        d, desc = t.get("date"), (t.get("description") or "")
        pid = (t.get("person") or {}).get("id")
        if not pid or not d:
            continue
        m = REHAB_RE.search(desc)
        if m:
            a = aff.get(int((t.get("toTeam") or {}).get("id") or 0), {})
            assigns.setdefault(pid, []).append({
                "date": d, "pos": m.group(1).upper(), "name": m.group(2).strip(),
                "club": (t.get("fromTeam") or {}).get("name"),
                "to": (t.get("toTeam") or {}).get("name") or m.group(3).strip(),
                "sportId": a.get("sportId"), "level": a.get("level", "")})
            continue
        il = IL_RE.search(desc)
        if il:
            placements.setdefault(pid, []).append(d)
            reasons[pid] = {"date": d, "days": int(il.group(1)),
                            "text": desc.split("injured list.")[-1].strip().rstrip(".") if "injured list." in desc else ""}
        elif BACK_RE.search(desc) and "rehab" not in desc.lower():
            enders.setdefault(pid, []).append({"date": d, "desc": desc})

    stat = statuses()
    on_il = lambda pid: (stat.get(pid) or "").startswith("D")

    open_rows, back_rows = [], []
    for pid, runs in assigns.items():
        runs.sort(key=lambda r: r["date"])
        ends = sorted(enders.get(pid, []), key=lambda r: r["date"])
        last_end = max([e["date"] for e in ends] or [""])
        # a man can be moved from the 10-day to the 60-day mid-rehab, which restarts the assignment; the stint that
        # matters is the run of assignments after the later of his last IL placement and whatever last ended one
        last_il = max(placements.get(pid, []) or [""])
        floor = max(last_end, last_il)
        stint = [r for r in runs if r["date"] >= floor] or [r for r in runs if r["date"] > last_end]
        if not stint or not on_il(pid):
            # not on the injured list any more, so whatever he was doing in the minors, it is not a rehab now
            r = runs[-1]
            e = [x for x in ends if x["date"] >= r["date"]]
            when = e[0]["desc"] if e else None
            back = e[0]["date"] if e else None
            if back and (today - dt.date.fromisoformat(back)).days <= 14:
                back_rows.append({"mlbam_id": pid, "name": r["name"], "pos": r["pos"], "club": r["club"],
                                  "started": r["date"], "back": back, "how": when})
            continue
        start = stint[0]
        age = (today - dt.date.fromisoformat(start["date"])).days
        if age > STINT_MAX:
            continue
        pitcher = start["pos"].upper() in PITCHER_POS
        cap = CAP["pitcher" if pitcher else "hitter"]
        back = _mlb_return(pid, season, pitcher, start["date"]) if with_outings else None
        if back:                                             # playing in the majors again, so it is over
            if (today - dt.date.fromisoformat(back)).days <= 14:
                back_rows.append({"mlbam_id": pid, "name": start["name"], "pos": start["pos"], "club": start["club"],
                                  "started": start["date"], "back": back, "how": "back in a major-league game"})
            continue
        og = outings(pid, [s["sportId"] for s in stint if s.get("sportId")], season, pitcher, start["date"]) if with_outings else []
        rs = reasons.get(pid) or {}
        last = og[-1]["date"] if og else None
        since = (today - dt.date.fromisoformat(last)).days if last else None
        # What says a rehab has stopped is that he has stopped playing, not that a calendar ran out. A 60-day arm is
        # routinely out there past the nominal window on a second assignment or an agreed extension, and Gavin Stone
        # throwing 54 pitches four days ago is plainly still rehabbing. So the setback flag is about the games, and
        # being past the window is reported as its own thing rather than a negative countdown.
        stalled = (since is None and age >= 5) or (since is not None and since >= STALE_OUTING)
        over = age > cap
        open_rows.append({
            "mlbam_id": pid, "name": start["name"], "pos": start["pos"], "pitcher": pitcher,
            "club": start["club"], "to": stint[-1]["to"], "level": stint[-1]["level"],
            "started": start["date"], "day": age + 1, "cap": cap,
            "deadline": (dt.date.fromisoformat(start["date"]) + dt.timedelta(days=cap)).isoformat(),
            "left": max(0, cap - age), "stalled": stalled, "over": over, "since_outing": since,
            "moved": len({s["to"] for s in stint}) > 1, "il": stat.get(pid, ""),
            "reason": rs.get("text") or "", "il_date": rs.get("date") or "",
            "outings": og, "last_outing": last, "n_outings": len(og)})
    # the ones actually playing come first, soonest to run out of window at the top; setbacks sit below
    open_rows.sort(key=lambda r: (r["stalled"], r["over"], r["left"], -r["day"]))
    back_rows.sort(key=lambda r: r["back"], reverse=True)
    return {"asof": today.isoformat(), "open": open_rows, "returned": back_rows}


def write(season: int) -> dict:
    d = build(season)
    DIR.mkdir(parents=True, exist_ok=True)
    (DIR / f"rehab_{season}.json").write_text(json.dumps(d))
    print(f"rehab: {len(d['open'])} on assignment, {len(d['returned'])} back in the last two weeks")
    return d


if __name__ == "__main__":
    import sys
    from .config import CURRENT_SEASON
    sys.exit(0 if write(CURRENT_SEASON) else 1)

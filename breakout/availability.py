"""Playing-time availability from MLB game logs: how many of his team's games a hitter was actually available for
(from his first MLB game of the season to his last, minus IL stints), and his PA pace over a 162-game season.

A late call-up (Kurtz in 2025) or a September debut (Joshua Baez in 2026) is not a durability problem; this separates
"was not up yet" from "was hurt" so projections can use a full-season pace instead of raw PA.
"""
from __future__ import annotations
import json, time
from datetime import date
import numpy as np
import pandas as pd
import requests

from .config import DATA
from .injuries import il_table, SEASON_END

_S = requests.Session(); _S.headers.update({"User-Agent": "Mozilla/5.0"})
API = "https://statsapi.mlb.com/api/v1"


def season_schedule(season: int, refresh: bool = False) -> pd.DataFrame:
    p = DATA / "availability" / f"schedule_{season}.parquet"
    if p.exists() and not refresh:
        return pd.read_parquet(p)
    j = _S.get(f"{API}/schedule", params=dict(sportId=1, season=season, gameType="R", startDate=f"{season}-03-01", endDate=f"{season}-10-05"), timeout=120).json()
    rows = []
    for d in j.get("dates", []):
        for g in d["games"]:
            if g["status"].get("codedGameState") not in ("F", "O") and g["status"].get("abstractGameState") != "Final":
                continue
            for side in ("home", "away"):
                rows.append(dict(season=season, date=d["date"], team_id=g["teams"][side]["team"]["id"], gamePk=g["gamePk"]))
    df = pd.DataFrame(rows); p.parent.mkdir(parents=True, exist_ok=True); df.to_parquet(p, index=False)
    return df


def _one_log(pid: int, season: int) -> dict | None:
    for attempt in range(3):
        try:
            j = _S.get(f"{API}/people/{pid}/stats", params=dict(stats="gameLog", group="hitting", season=season), timeout=60).json()
            break
        except Exception:
            time.sleep(1 + attempt)
    else:
        return None
    sp = (j.get("stats") or [{}])[0].get("splits", [])
    sp = [s for s in sp if s.get("game", {}).get("gameType", "R") == "R"]
    if not sp:
        return dict(mlbam_id=pid, season=season, first_game=None, last_game=None, games=0, segments="[]")
    dates = sorted(s["date"] for s in sp)
    # contiguous team segments (traded players): (team_id, first date with that team, last date with that team)
    segs = []
    for s in sorted(sp, key=lambda x: x["date"]):
        tid = (s.get("team") or {}).get("id")
        if segs and segs[-1][0] == tid:
            segs[-1][2] = s["date"]
        else:
            segs.append([tid, s["date"], s["date"]])
    return dict(mlbam_id=pid, season=season, first_game=dates[0], last_game=dates[-1], games=len(sp), segments=json.dumps(segs))


def game_logs(ids, season: int, refresh=False) -> pd.DataFrame:
    """First/last game, games played and teams for each hitter-season, cached. refresh=True re-reads every id (the
    season in progress, whose last game moves); otherwise only ids not yet cached are fetched. Eight at a time."""
    from concurrent.futures import ThreadPoolExecutor
    p = DATA / "availability" / f"gamelogs_{season}.parquet"
    have = pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=["mlbam_id"])
    want = [int(i) for i in ids]
    todo = want if refresh else [i for i in want if i not in set(have["mlbam_id"].astype(int))]
    with ThreadPoolExecutor(8) as ex:
        rows = [r for r in ex.map(lambda i: _one_log(i, season), todo) if r is not None]
    if rows:
        new = pd.DataFrame(rows)
        have = have[~have["mlbam_id"].astype(int).isin(new["mlbam_id"])] if len(have) else have
        df = pd.concat([have, new], ignore_index=True)
        p.parent.mkdir(parents=True, exist_ok=True); df.to_parquet(p, index=False)
    else:
        df = have
    return df[df["mlbam_id"].astype(int).isin(want)]


def availability(ps: pd.DataFrame, seasons=(2024, 2025, 2026), asof: str | None = None, refresh_current: bool = False) -> pd.DataFrame:
    """Per hitter-season: team games available (window minus IL), PA pace per 162, games share, call-up flag.

    Built for EVERY season the models train on. It used to exist for 2024-26 only, so the playing-time model's main
    input (base_pa, which uses pace when it has it) meant raw PA in three of its five training seasons and pace in the
    other two, averaging 40-57 PA apart for the same players. refresh_current re-reads the latest season's logs, which
    otherwise froze at the date they were first fetched."""
    out = []
    for season in seasons:
        sub = ps[(ps["season"] == season) & (ps["PA"] >= 30)]
        if not len(sub):
            continue
        live = refresh_current and season == max(seasons)
        sched = season_schedule(season, refresh=live); logs = game_logs(sub["mlbam_id"].tolist(), season, refresh=live)
        il = il_table(season)
        end_cap = pd.Timestamp(min(SEASON_END[season], asof or SEASON_END[season])) if season == max(seasons) else pd.Timestamp(SEASON_END[season])
        team_dates = {tid: sorted(pd.to_datetime(g["date"]).tolist()) for tid, g in sched.groupby("team_id")}
        season_len = sched.groupby("team_id")["gamePk"].nunique().median()
        first_team_game = {tid: d[0] for tid, d in team_dates.items()}
        for _, r in logs.iterrows():
            if not r["games"]:
                continue
            segs = json.loads(r["segments"]); pid = int(r["mlbam_id"])
            stints = il[il["mlbam_id"] == pid]
            il_ranges = [(pd.Timestamp(s["il_start"]), pd.Timestamp(s["il_start"]) + pd.Timedelta(days=int(s["il_days"]))) for _, s in stints.iterrows()]
            avail = 0; window_games = 0
            for k, (tid, d0, d1) in enumerate(segs):
                lo = pd.Timestamp(d0); hi = pd.Timestamp(d1)
                if k == len(segs) - 1:
                    # still on the club after his last game: count through the as-of date unless he is on the IL / sent down
                    hi = max(hi, min(end_cap, hi + pd.Timedelta(days=7)))
                for gd in team_dates.get(tid, []):
                    if lo <= gd <= hi:
                        window_games += 1
                        if not any(a <= gd <= b for a, b in il_ranges):
                            avail += 1
            pa = float(sub.loc[sub["mlbam_id"] == pid, "PA"].iloc[0]); g = int(r["games"])
            avail = max(avail, g)
            late = (pd.Timestamp(r["first_game"]) - first_team_game.get(segs[0][0], pd.Timestamp(f"{season}-03-25"))).days
            out.append(dict(mlbam_id=pid, season=season, first_game=r["first_game"], last_game=r["last_game"], games=g, window_games=window_games, avail_games=avail,
                            il_games_in_window=window_games - avail, pa_pace_162=round(min(pa / avail * season_len, 740), 0) if avail else np.nan,
                            g_share=round(g / avail, 3) if avail else np.nan, days_late=int(late), late_callup=int(late > 21)))
    return pd.DataFrame(out)

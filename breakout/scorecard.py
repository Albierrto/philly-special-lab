"""Keep score of the streamer forecasts: archive each day's numbers before first pitch, grade them after.

    python -m breakout.scorecard --archive     # snapshot today's projections while the games are still ahead
    python -m breakout.scorecard --grade       # score every archived day whose games have finished
    python -m breakout.scorecard               # both, which is what the refresh workflow runs

Why archiving is the whole trick: output/streamers/streamers.json is overwritten on every build, so by the time a
game is over the number that was on the page beforehand is gone. Grading the rebuilt file would be marking your own
homework after seeing the answer. So each day's projections are written once, to data/forecasts/<date>.csv, and only
while `now` is still earlier than that day's first scheduled pitch - a later run of the same day will refresh it
(probables firm up through the morning), and any run after the first game has started leaves it alone.

Actuals come from the MLB box scores and are scored with the league's own rules, the same ones the model projects in.
"""
from __future__ import annotations
import argparse, json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import requests

from . import config as C

FC = C.DATA / "forecasts"
SC = C.DATA / "scorecard"
API = "https://statsapi.mlb.com/api/v1"
_S = requests.Session(); _S.headers.update({"User-Agent": "philly-special-lab/1.0 (forecast scorecard)"})

PSCORE = {"IP": 1, "K": 1, "ER": -1, "QS": 4, "CG": 10, "SHO": 8}


def _box(pk: int) -> dict | None:
    for attempt in range(3):
        try:
            r = _S.get(f"{API}/game/{int(pk)}/boxscore", timeout=60)
            if r.ok: return r.json()
        except Exception:
            pass
    return None


def hitter_points(s: dict) -> float:
    """The league's hitting formula, verified against Fantrax's own table for all 500 hitters (see league_config)."""
    h, d2, t3, hr = s.get("hits", 0), s.get("doubles", 0), s.get("triples", 0), s.get("homeRuns", 0)
    b1 = h - d2 - t3 - hr
    return float(2 * b1 + 3 * d2 + 4 * t3 + 5 * hr + s.get("runs", 0) + s.get("rbi", 0)
                 + s.get("baseOnBalls", 0) + s.get("hitByPitch", 0) + 3 * s.get("stolenBases", 0))


def pitcher_points(s: dict) -> tuple[float, float]:
    ip_raw = str(s.get("inningsPitched", "0") or "0")
    whole, _, frac = ip_raw.partition(".")
    ip = float(whole or 0) + (float(frac or 0) / 3.0)
    er, k = s.get("earnedRuns", 0), s.get("strikeOuts", 0)
    pts = ip + k - er
    if ip >= 6 and er <= 3 and s.get("gamesStarted", 0): pts += PSCORE["QS"]
    if s.get("completeGames", 0): pts += PSCORE["CG"]
    if s.get("shutouts", 0): pts += PSCORE["SHO"]
    return float(pts), round(ip, 1)


def _streamers() -> dict:
    p = C.OUT / "streamers" / "streamers.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _rows(block):
    return block if isinstance(block, list) else [dict(zip(block["cols"], r)) for r in block["rows"]]


def archive(day: str | None = None) -> str:
    """Write today's projections, but only while the first game of the day is still ahead of us."""
    s = _streamers()
    if not s: print("scorecard: no streamers build to archive"); return ""
    day = day or pd.Timestamp.today().strftime("%Y-%m-%d")
    games = _rows(s.get("games", []))
    starts = [g.get("game_utc") or g.get("gameDate") for g in games if g.get("date") == day]
    hd = [r for r in _rows(s["hitter_days"]) if r["date"] == day]
    ps = [r for r in _rows(s["pitcher_starts"]) if r["date"] == day and r.get("sp_source") != "replaced"]
    if not hd and not ps: print(f"scorecard: nothing scheduled for {day}"); return ""
    out = FC / f"{day}.csv"
    if out.exists():
        # a later run may refresh it (probables firm up through the morning) but only before anything has started
        first = _first_pitch(day)
        if first is not None and _now() >= first:
            print(f"scorecard: {day} is already under way - keeping the forecast that was on the page before it started")
            return str(out)
    rows = ([dict(kind="H", mlbam_id=r["mlbam_id"], name=r["name"], team=r.get("team"), gamePk=r["gamePk"],
                  opp=r.get("opp"), exp_pts=r["exp_pts"], src=r.get("opp_sp_source")) for r in hd]
            + [dict(kind="P", mlbam_id=r["mlbam_id"], name=r["name"], team=r.get("team"), gamePk=r["gamePk"],
                    opp=r.get("opp"), exp_pts=r["exp_pts"], src=r.get("sp_source")) for r in ps])
    FC.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows); df.insert(0, "date", day)
    df["made_utc"] = _now().strftime("%Y-%m-%dT%H:%M:%SZ")
    df.to_csv(out, index=False)
    print(f"scorecard: archived {len(df)} projections for {day} ({(df.kind=='P').sum()} starts)")
    return str(out)


def _now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").tz_localize(None)


def _first_pitch(day: str):
    """When the earliest game of the day starts, in naive UTC. None if the schedule cannot be read."""
    try:
        j = _S.get(f"{API}/schedule", params=dict(sportId=1, date=day), timeout=60).json()
        ts = [g["gameDate"] for d in j.get("dates", []) for g in d.get("games", [])]
        if not ts: return None
        return pd.to_datetime(ts, utc=True).min().tz_localize(None)
    except Exception:
        return None


def grade(day: str) -> pd.DataFrame | None:
    """Score one archived day against the box scores. Returns the per-player frame, or None if it is not gradeable."""
    src = FC / f"{day}.csv"
    if not src.exists(): return None
    fc = pd.read_csv(src)
    pks = sorted(set(int(x) for x in fc["gamePk"].dropna()))
    with ThreadPoolExecutor(10) as ex:
        boxes = [b for b in ex.map(_box, pks) if b]
    if len(boxes) < max(1, 0.8 * len(pks)):
        print(f"scorecard: {day} only {len(boxes)}/{len(pks)} box scores back, leaving it for next time"); return None
    hit, pit, ip = {}, {}, {}
    final = 0
    for b in boxes:
        for side in ("away", "home"):
            for _, pl in b.get("teams", {}).get(side, {}).get("players", {}).items():
                mid = pl.get("person", {}).get("id"); st = pl.get("stats", {})
                bt = st.get("batting", {})
                if bt and bt.get("plateAppearances") is not None: hit[mid] = hitter_points(bt)
                pt = st.get("pitching", {})
                if pt and pt.get("gamesStarted"): pit[mid], ip[mid] = pitcher_points(pt)
        final += 1
    fc["actual"] = [(hit if k == "H" else pit).get(int(m), np.nan) for k, m in zip(fc["kind"], fc["mlbam_id"])]
    fc["ip"] = [ip.get(int(m), np.nan) if k == "P" else np.nan for k, m in zip(fc["kind"], fc["mlbam_id"])]
    fc["played"] = fc["actual"].notna()
    SC.mkdir(parents=True, exist_ok=True)
    fc.to_csv(SC / f"{day}.csv", index=False)
    return fc


def summarise() -> pd.DataFrame:
    """One row per day per population, plus a rolling all-time row, written to data/scorecard/summary.csv."""
    rows = []
    for f in sorted(SC.glob("20*.csv")):
        d = pd.read_csv(f)
        for kind, lab in (("H", "hitters"), ("P", "starters")):
            x = d[(d["kind"] == kind) & d["played"]].dropna(subset=["actual", "exp_pts"])
            if len(x) < 5: continue
            e = x["actual"] - x["exp_pts"]
            rows.append(dict(date=f.stem, who=lab, n=len(x), predicted=round(x["exp_pts"].mean(), 2),
                             actual=round(x["actual"].mean(), 2), bias=round(e.mean(), 2),
                             mae=round(e.abs().mean(), 2),
                             r=round(float(np.corrcoef(x["exp_pts"], x["actual"])[0, 1]), 3) if len(x) > 3 else np.nan,
                             zero_pct=round(100 * float((x["actual"] == 0).mean()), 1)))
    out = pd.DataFrame(rows)
    if len(out):
        SC.mkdir(parents=True, exist_ok=True)
        out.sort_values(["date", "who"]).to_csv(SC / "summary.csv", index=False)
    return out


def main(argv=None):
    a = argparse.ArgumentParser(); a.add_argument("--archive", action="store_true"); a.add_argument("--grade", action="store_true")
    a.add_argument("--day"); ns = a.parse_args(argv)
    do_all = not (ns.archive or ns.grade)
    if ns.archive or do_all: archive(ns.day)
    if ns.grade or do_all:
        done = {f.stem for f in SC.glob("20*.csv")}
        today = pd.Timestamp.today().strftime("%Y-%m-%d")
        for f in sorted(FC.glob("20*.csv")):
            if f.stem in done or f.stem >= today: continue
            g = grade(f.stem)
            if g is not None:
                x = g[g["played"]].dropna(subset=["actual"])
                print(f"scorecard: graded {f.stem} - {len(x)} players, MAE {(x['actual']-x['exp_pts']).abs().mean():.2f}")
    s = summarise()
    if len(s): print(s.tail(6).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

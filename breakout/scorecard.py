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
    day = day or C.league_today().isoformat()
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
                  opp=r.get("opp"), exp_pts=r["exp_pts"], src=r.get("opp_sp_source"),
                  owner=r.get("owner"), elig=r.get("elig"), active=r.get("active")) for r in hd]
            + [dict(kind="P", mlbam_id=r["mlbam_id"], name=r["name"], team=r.get("team"), gamePk=r["gamePk"],
                    opp=r.get("opp"), exp_pts=r["exp_pts"], src=r.get("sp_source"),
                    owner=r.get("owner"), elig="SP", active=r.get("active")) for r in ps])
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
        today = C.league_today().isoformat()
        for f in sorted(FC.glob("20*.csv")):
            if f.stem in done or f.stem >= today: continue
            g = grade(f.stem)
            if g is not None:
                x = g[g["played"]].dropna(subset=["actual"])
                print(f"scorecard: graded {f.stem} - {len(x)} players, MAE {(x['actual']-x['exp_pts']).abs().mean():.2f}")
    s = summarise()
    if len(s): print(s.tail(6).to_string(index=False))
    m = log_current_period()
    if m is not None and len(m):
        tot = m.groupby("team")[["proj", "actual", "perfect"]].sum().round(1)
        print("\nmatchup log, period to date (following the recommended lineup):")
        print(tot.sort_values("actual", ascending=False).head(4).to_string())
    return 0


def log_current_period():
    """Run the matchup log for every team over whichever scoring period today falls in."""
    cfg_p = C.DATA / "fantrax" / "league_config.json"
    if not cfg_p.exists(): return None
    sch = json.loads(cfg_p.read_text()).get("schedule", {})
    today = C.league_today().isoformat()
    per = next((p for p in sch.get("periods", []) if p["start"] <= today <= p["end"]), None)
    if not per: return None
    teams = set()
    for f in sorted(SC.glob("20*.csv")):
        if not (per["start"] <= f.stem <= per["end"]): continue
        d = pd.read_csv(f)
        if "owner" in d.columns:
            teams |= {o for o in d["owner"].dropna().unique() if o and str(o) not in ("FA", "W (Sun)", "W (Mon)")}
    if not teams: return None
    df = matchup_log(per["start"], per["end"], sorted(teams))
    if len(df): df.insert(0, "period", per["n"])
    if len(df): df.to_csv(SC / "matchup_log.csv", index=False)
    return df




# ------------------------------------------------------------------------------------------------------------------
# The matchup log: not "was the projection right" but "would following it have worked".
#
# The lineup is chosen from the PROJECTIONS, exactly as the site recommends it, and then scored with what actually
# happened. That is the number a manager who did what this page said would have put up. Alongside it sits the perfect
# hindsight lineup from the same roster, which is the ceiling nobody reaches, so the gap between the two is the price
# of not knowing the future rather than a flaw in the model.
#
# Fantrax's read-only API does not publish set lineups or per-day team scores, so the real Fantrax number is not
# recoverable here. This is like-for-like instead: both columns are the same ten slots, one priced before the games
# and one after.
SLOTS = ["C", "1B", "2B", "3B", "SS", "OF", "OF", "OF", "UT", "UT"]
CAP = 10          # pitcher starts per period in this league


def _fill(rows, value_key):
    """Best legal ten-man lineup out of `rows`, maximising `value_key`. Returns (total, the chosen rows)."""
    from scipy.optimize import linear_sum_assignment
    rows = [r for r in rows if r.get(value_key) is not None and r[value_key] == r[value_key]]
    if not rows: return 0.0, []
    n, m = len(SLOTS), len(rows)
    C = np.full((m, m + n), 0.0)
    for i, r in enumerate(rows):
        e = str(r.get("elig") or "").split("/")
        for j, s in enumerate(SLOTS):
            C[i, j] = -float(r[value_key]) if (s == "UT" or s in e) else 1e6
    ri, ci = linear_sum_assignment(C)
    tot, used = 0.0, []
    for i, j in zip(ri, ci):
        if j < n and C[i, j] < 1e5:
            tot += float(rows[i][value_key]); used.append(rows[i])
    return tot, used


def matchup_log(period_start: str, period_end: str, teams: list[str]) -> pd.DataFrame:
    """Day by day for each team: what the recommended lineup was projected to score, what it actually scored,
    and what the perfect lineup would have scored. Pitchers are allocated against the ten-start cap best-first."""
    out = []
    for team in teams:
        used = 0
        for f in sorted(SC.glob("20*.csv")):
            day = f.stem
            if not (period_start <= day <= period_end): continue
            d = pd.read_csv(f)
            d = d[(d["owner"] == team) & (d.get("active", 1) != 0)] if "owner" in d.columns else d.iloc[0:0]
            if not len(d): continue
            hit = d[d["kind"] == "H"].to_dict("records")
            pit = sorted(d[d["kind"] == "P"].to_dict("records"), key=lambda r: -r["exp_pts"])
            pit = pit[:max(0, CAP - used)]; used += len(pit)
            pj, chosen = _fill(hit, "exp_pts")
            act = sum(float(r["actual"]) for r in chosen if r.get("actual") == r.get("actual"))
            bj, _ = _fill(hit, "actual")
            pp = sum(float(r["exp_pts"]) for r in pit)
            pa = sum(float(r["actual"]) for r in pit if r.get("actual") == r.get("actual"))
            out.append(dict(date=day, team=team, proj=round(pj + pp, 1), actual=round(act + pa, 1),
                            perfect=round(bj + pa, 1), starts=len(pit)))
    df = pd.DataFrame(out)
    if len(df):
        SC.mkdir(parents=True, exist_ok=True)
        df.sort_values(["team", "date"]).to_csv(SC / "matchup_log.csv", index=False)
    return df


if __name__ == "__main__":
    sys.exit(main())

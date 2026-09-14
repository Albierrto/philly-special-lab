"""Streamers: day-by-day hitter matchups and start-by-start pitcher matchups for a date window.

Inputs (all public): MLB Stats API schedule + probable pitchers + venues (coordinates, orientation, roof),
recent starters (to project rotations where no probable is posted), team and player L/R splits and last-30-day form;
Baseball Savant Statcast park factors (3-year rolling, by batter side); Open-Meteo hourly forecast at first pitch;
the lab's own hitter and pitcher tables (points rates, xLP, Pitching+ proxies) and the Fantrax ownership tables.

Everything is a transparent multiplier on a baseline rate, so the weights can be argued with:
    hitter game pts = base pts/PA x PA/G x SP^0.6 x platoon x park(runs, bat side) x weather x home x form
    pitcher start pts = base pts/GS x opponent(wOBA vs hand) x park x weather x home  + K bonus(opponent K% vs hand)
"""
from __future__ import annotations
import io, json, math, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import requests

from .config import DATA, OUT
from .names import key

_S = requests.Session(); _S.headers.update({"User-Agent": "Mozilla/5.0"})
API = "https://statsapi.mlb.com/api/v1"
LG_WOBA = 0.312; LG_K = 0.222; LG_XWOBA_P = 0.312
WOBA_W = dict(bb=0.69, hbp=0.72, x1b=0.88, x2b=1.25, x3b=1.59, hr=2.05)


def _get(url, **params):
    for i in range(3):
        try:
            r = _S.get(url, params=params, timeout=60); r.raise_for_status(); return r.json()
        except Exception as e:  # noqa
            time.sleep(1 + i)
    return {}


def woba_from_stat(st: dict) -> tuple[float, int, float]:
    """(wOBA, PA, K%) from an MLB API hitting stat block (works for team and player, and for pitching 'against' blocks)."""
    pa = st.get("plateAppearances") or 0
    if not pa:
        return np.nan, 0, np.nan
    h, d, t, hr = st.get("hits", 0), st.get("doubles", 0), st.get("triples", 0), st.get("homeRuns", 0)
    bb, ibb, hbp, sf, ab = st.get("baseOnBalls", 0), st.get("intentionalWalks", 0), st.get("hitByPitch", 0), st.get("sacFlies", 0), st.get("atBats", 0)
    x1b = h - d - t - hr
    num = WOBA_W["bb"] * (bb - ibb) + WOBA_W["hbp"] * hbp + WOBA_W["x1b"] * x1b + WOBA_W["x2b"] * d + WOBA_W["x3b"] * t + WOBA_W["hr"] * hr
    den = ab + bb - ibb + sf + hbp
    return (num / den if den else np.nan), int(pa), (st.get("strikeOuts", 0) / pa)


def shrink(x, n, prior, k):
    x = np.where(pd.isna(x), prior, x); n = np.where(pd.isna(n), 0, n)
    return (x * n + prior * k) / (n + k)


# ---------------------------------------------------------------- schedule, venues, rotations
def teams() -> pd.DataFrame:
    j = _get(f"{API}/teams", sportId=1, season=2026)
    return pd.DataFrame([dict(team_id=t["id"], abbr=t["abbreviation"], name=t["name"], club=t.get("teamName")) for t in j.get("teams", [])])


def schedule(start: str, end: str) -> pd.DataFrame:
    j = _get(f"{API}/schedule", sportId=1, startDate=start, endDate=end, hydrate="probablePitcher,venue,team")
    rows = []
    for d in j.get("dates", []):
        for g in d["games"]:
            if g.get("gameType") not in ("R", "P", "F", "D", "L", "W"):
                continue
            for side, opp in (("home", "away"), ("away", "home")):
                t = g["teams"][side]; o = g["teams"][opp]
                pp = t.get("probablePitcher") or {}
                rows.append(dict(gamePk=g["gamePk"], date=d["date"], gameDate=g["gameDate"], status=g["status"]["detailedState"],
                                 venue_id=g["venue"]["id"], venue=g["venue"]["name"], dayNight=g.get("dayNight"), doubleHeader=g.get("doubleHeader"),
                                 team_id=t["team"]["id"], team=t["team"]["name"], home=(side == "home"), opp_id=o["team"]["id"], opp=o["team"]["name"],
                                 sp_id=pp.get("id"), sp_name=pp.get("fullName"), sp_source="listed" if pp else None))
    return pd.DataFrame(rows)


def venues() -> pd.DataFrame:
    j = _get(f"{API}/venues", hydrate="location,fieldInfo,timezone", season=2026, sportId=1)
    rows = []
    for v in j.get("venues", []):
        loc = v.get("location", {}); co = loc.get("defaultCoordinates", {}); fi = v.get("fieldInfo", {}); tz = v.get("timeZone", {})
        rows.append(dict(venue_id=v["id"], venue=v["name"], lat=co.get("latitude"), lon=co.get("longitude"), azimuth=loc.get("azimuthAngle"),
                         elevation=loc.get("elevation"), roof=fi.get("roofType"), tz=tz.get("id")))
    return pd.DataFrame(rows)


def recent_starters(end: str, days: int = 16) -> pd.DataFrame:
    start = (date.fromisoformat(end) - timedelta(days=days)).isoformat()
    sch = schedule(start, end); sch = sch[sch["status"].str.startswith("Final") | sch["status"].str.contains("Completed")]
    return sch.sort_values("gameDate")[["date", "gameDate", "team_id", "team", "sp_id", "sp_name"]]


ROTATION_MIN = 5    # clubs run five or six starters; a shorter projected cycle manufactures second starts
REST_DAYS = 6       # default days between two starts by the same arm. 93% of 2026 starters turn on a six-day median,
                    # not five — six-man rotations and extra rest are the norm now, and assuming five invents two-start weeks.


def rest_days(logs: pd.DataFrame, default: int = REST_DAYS, lo: int = 5, hi: int = 7) -> dict:
    """Each arm's own turn, from the gaps between his starts this season. Tampa give Griffin Jax six or seven days;
    projecting him on five is what put a second start on his week that was never going to happen."""
    out = {}
    for r in logs.itertuples():
        ds = sorted(getattr(r, "start_dates", None) or [])
        if len(ds) < 5: continue
        gaps = [(date.fromisoformat(b) - date.fromisoformat(a)).days for a, b in zip(ds, ds[1:])]
        gaps = [g for g in gaps if 3 <= g <= 9]          # ignore IL gaps and doubleheader oddities
        if len(gaps) >= 4: out[int(r.mlbam_id)] = int(min(hi, max(lo, round(float(np.median(gaps))))))
    return out


def project_rotations(sch: pd.DataFrame, recent: pd.DataFrame, min_apps: int = 2, regular_ids=None, active_ids=None, exclude_ids=None, rest=None) -> pd.DataFrame:
    """Fill sp_id/sp_name for games without a listed probable by cycling each team's recent rotation order.
    A pitcher counts as a rotation member if he started at least `min_apps` times in the look-back window or is a
    regular starter on the season (regular_ids); this keeps openers and one-off spot starters out of the cycle."""
    regular_ids = set(regular_ids or []); active_ids = set(active_ids) if active_ids is not None else None; exclude_ids = set(exclude_ids or [])
    if active_ids is not None:
        recent = recent[recent["sp_id"].isin(active_ids) | recent["sp_id"].isna()]
    if exclude_ids:
        recent = recent[~recent["sp_id"].isin(exclude_ids)]
    sch = sch.sort_values(["gameDate", "gamePk"]).copy()
    for tid, grp in sch.groupby("team_id"):
        rec = recent[recent["team_id"] == tid].dropna(subset=["sp_id"])
        if not len(rec):
            continue
        # rotation = distinct starters in order of most recent appearance, keeping the last ~5-6 regulars
        order = []
        for pid, nm in zip(rec["sp_id"][::-1], rec["sp_name"][::-1]):
            if pid not in [o[0] for o in order]:
                order.append((pid, nm))
        counts = rec["sp_id"].value_counts()
        regs = [(p, n) for p, n in order if counts.get(p, 0) >= min_apps or p in regular_ids][:6]
        # a club's rotation is five or six deep. If the filters leave fewer than five, top up from the arms that have
        # actually been taking turns, most recent first — otherwise the cycle is too short and it hands somebody a second
        # start he was never going to make (Houston lost Miguel Ullola, one start in the window and no season line, and
        # the four-man cycle that was left gave Cristian Javier a phantom 9/20).
        if len(regs) < ROTATION_MIN:
            have = {p for p, _ in regs}
            for p, n in order:
                if p not in have:
                    regs.append((p, n)); have.add(p)
                if len(regs) >= ROTATION_MIN: break
        rot = list(reversed(regs[:6]))  # oldest -> most recent
        ids = [r[0] for r in rot]
        last = rot[-1][0]
        idx = ids.index(last)
        # last start date per arm (from the look-back window, then from listed/projected starts as we go). A modern
        # rotation turns every five days; projecting a man on four would invent two-start weeks nobody is going to get.
        known = {}
        for pid, d in zip(rec["sp_id"], rec["date"]): known.setdefault(pid, []).append(str(d)[:10])
        for gi in grp.index:   # starts already on the board this week (posted probables, or picks kept from an earlier pass), past AND future
            if pd.notna(sch.at[gi, "sp_id"]): known.setdefault(sch.at[gi, "sp_id"], []).append(str(sch.at[gi, "date"])[:10])
        def rested(pid, day):
            d0 = date.fromisoformat(str(day)[:10]); need = (rest or {}).get(int(pid), REST_DAYS)
            return all(abs((d0 - date.fromisoformat(d)).days) >= need for d in known.get(pid, []))
        for gi in grp.index:
            day = sch.at[gi, "date"]
            if pd.notna(sch.at[gi, "sp_id"]):
                # a listed probable resets the cycle position if he is in the rotation
                pid = sch.at[gi, "sp_id"]
                if pid in ids:
                    idx = ids.index(pid)
                continue
            pick = None
            for k in range(1, len(rot) + 1):
                cand = rot[(idx + k) % len(rot)]
                if rested(cand[0], day):
                    pick, idx = cand, (idx + k) % len(rot); break
            if pick is None:
                continue  # everyone pitched in the last four days (spot start / opener day): leave the game TBD
            known.setdefault(pick[0], []).append(str(day)[:10])
            sch.at[gi, "sp_id"], sch.at[gi, "sp_name"], sch.at[gi, "sp_source"] = pick[0], pick[1], "projected"
    return sch


def active_rosters(team_ids) -> tuple[set, dict]:
    """Ids on each club's active (26-man) roster, plus a status map for everyone on the 40-man (IL, minors, restricted)."""
    active = set(); status = {}
    for tid in team_ids:
        j = _get(f"{API}/teams/{tid}/roster", rosterType="40Man", season=2026)
        for r in j.get("roster", []):
            pid = r["person"]["id"]; st = (r.get("status") or {}).get("code", ""); desc = (r.get("status") or {}).get("description", "")
            status[pid] = desc or st
            if st == "A":
                active.add(pid)
        time.sleep(0.08)
    return active, status


def current_teams(ids) -> pd.DataFrame:
    rows = []
    for chunk in [ids[i:i + 40] for i in range(0, len(ids), 40)]:
        j = _get(f"{API}/people", personIds=",".join(str(int(x)) for x in chunk), hydrate="currentTeam")
        for p in j.get("people", []):
            ct = p.get("currentTeam") or {}
            rows.append(dict(mlbam_id=p["id"], team_id_cur=ct.get("id"), team_cur=ct.get("name")))
        time.sleep(0.1)
    return pd.DataFrame(rows, columns=["mlbam_id", "team_id_cur", "team_cur"])


# ---------------------------------------------------------------- park factors and weather
def park_factors(year: int = 2026) -> pd.DataFrame:
    import re
    out = []
    for side in ("", "L", "R"):
        r = _S.get(f"https://baseballsavant.mlb.com/leaderboard/statcast-park-factors?type=year&year={year}&batSide={side}&stat=index_wOBA&condition=All&rolling=3", timeout=60)
        m = re.search(r"var data = (\[.*?\]);", r.text, re.S)
        if not m:
            continue
        for d in json.loads(m.group(1)):
            out.append(dict(venue_id=int(d["venue_id"]), venue=d["venue_name"], bat_side=side or "All", pf_runs=int(d["index_runs"]), pf_hr=int(d["index_hr"]),
                            pf_woba=int(d["index_woba"]), pf_so=int(d["index_so"]), pf_1b=int(d["index_1b"]), pf_years=d["year_range"]))
    return pd.DataFrame(out)


def weather(games: pd.DataFrame, ven: pd.DataFrame) -> pd.DataFrame:
    """Open-Meteo hourly forecast at first pitch for each unique (venue, gamePk)."""
    g = games.drop_duplicates("gamePk")[["gamePk", "gameDate", "venue_id"]].merge(ven, on="venue_id", how="left")
    rows = []
    for vid, grp in g.groupby("venue_id"):
        v = grp.iloc[0]
        if pd.isna(v.get("lat")):
            continue
        tz = v["tz"] or "America/New_York"
        local = [datetime.fromisoformat(x.replace("Z", "+00:00")).astimezone(ZoneInfo(tz)) for x in grp["gameDate"]]
        d0, d1 = min(local).date().isoformat(), max(local).date().isoformat()
        j = _get("https://api.open-meteo.com/v1/forecast", latitude=v["lat"], longitude=v["lon"], hourly="temperature_2m,wind_speed_10m,wind_direction_10m,precipitation_probability,precipitation",
                 temperature_unit="fahrenheit", wind_speed_unit="mph", timezone=tz, start_date=d0, end_date=d1)
        h = j.get("hourly", {})
        if not h:
            continue
        times = {t: i for i, t in enumerate(h["time"])}
        for pk, lt in zip(grp["gamePk"], local):
            # first-pitch hour and the following 2 hours averaged
            keyt = lt.replace(minute=0).strftime("%Y-%m-%dT%H:00")
            i = times.get(keyt)
            if i is None:
                continue
            sl = slice(i, min(i + 3, len(h["time"])))
            temp = float(np.nanmean(h["temperature_2m"][sl])); ws = float(np.nanmean(h["wind_speed_10m"][sl])); wd = float(h["wind_direction_10m"][i])
            pop = float(np.nanmax(h["precipitation_probability"][sl])) if h.get("precipitation_probability") else np.nan
            rows.append(dict(gamePk=pk, local_start=lt.strftime("%a %m/%d %I:%M %p"), temp_f=round(temp, 0), wind_mph=round(ws, 0), wind_dir=wd, precip_prob=pop))
        time.sleep(0.2)
    cols = ["gamePk", "local_start", "temp_f", "wind_mph", "wind_dir", "precip_prob"]
    if not rows:   # forecast service unreachable: still return the frame shape so the build runs (the site fetches weather live anyway)
        g2 = games.drop_duplicates("gamePk")[["gamePk", "gameDate", "venue_id"]].merge(ven, on="venue_id", how="left")
        for r in g2.itertuples():
            try: lt = datetime.fromisoformat(r.gameDate.replace("Z", "+00:00")).astimezone(ZoneInfo(getattr(r, "tz", None) or "America/New_York")); ls = lt.strftime("%a %m/%d %I:%M %p")
            except Exception: ls = None
            rows.append(dict(gamePk=r.gamePk, local_start=ls, temp_f=np.nan, wind_mph=np.nan, wind_dir=np.nan, precip_prob=np.nan))
    return pd.DataFrame(rows, columns=cols)


def wind_component(wind_from_deg, azimuth_deg):
    """+ = blowing out toward center field, - = blowing in. azimuth = bearing home plate -> center field.
    Meteorological wind direction is where the wind comes FROM; wind blows toward (dir+180)."""
    if pd.isna(wind_from_deg) or pd.isna(azimuth_deg):
        return 0.0
    toward = (wind_from_deg + 180) % 360
    return math.cos(math.radians(toward - azimuth_deg))


# ---------------------------------------------------------------- splits and form
def team_splits(team_ids) -> pd.DataFrame:
    rows = []
    for tid in team_ids:
        j = _get(f"{API}/teams/{tid}/stats", stats="statSplits", group="hitting", season=2026, sitCodes="vl,vr")
        for st in j.get("stats", []):
            for s in st.get("splits", []):
                w, pa, k = woba_from_stat(s["stat"]); rows.append(dict(team_id=tid, vs=s["split"]["code"], t_woba=w, t_pa=pa, t_k=k, t_ops=float(s["stat"].get("ops", 0) or 0)))
        time.sleep(0.1)
    return pd.DataFrame(rows)


def team_form(team_ids, season: int = 2026, days: int = 30, asof=None, workers: int = 10) -> pd.DataFrame:
    """Each club's strikeout rate over the trailing `days`, against its rate over everything before that window.

    Measured over 2,924 real 2026 starts: a club's strikeout rate PERSISTS month to month (r 0.43 between one 25-game
    block and the next) while its wOBA does not at all (r -0.01). That is why recency helps here and did not for the
    bat: a hot month at the plate is noise, but a lineup that has started striking out more usually has actually
    changed - a call-up, an injury to a contact bat, a September roster. Residual on this movement is +2.19 points per
    start per unit of relative move (t 2.11), and refitting on April-July alone gives 2.10, so it is stable.
    It is a small correction - a one-sd move is 0.22 points - and it is applied at half strength for that reason."""
    from concurrent.futures import ThreadPoolExecutor
    end = pd.Timestamp(asof or pd.Timestamp.today().normalize())
    def one(tid):
        j = _get(f"{API}/teams/{int(tid)}/stats", stats="gameLog", group="hitting", season=season)
        rows = []
        for st in j.get("stats", []):
            for g in st.get("splits", []):
                d = g.get("date"); k = g["stat"].get("strikeOuts", 0); pa = g["stat"].get("plateAppearances", 0)
                if d and pa: rows.append((pd.Timestamp(d), float(k), float(pa)))
        if not rows: return dict(team_id=int(tid), k_move=np.nan, k_30=np.nan, pa_30=0.0)
        g = pd.DataFrame(rows, columns=["date", "k", "pa"]).sort_values("date")
        g = g[g["date"] < end]
        recent, prior = g[g["date"] >= end - pd.Timedelta(days=days)], g[g["date"] < end - pd.Timedelta(days=days)]
        if recent["pa"].sum() < 200 or prior["pa"].sum() < 200: return dict(team_id=int(tid), k_move=np.nan, k_30=np.nan, pa_30=float(recent["pa"].sum()))
        kr, kp = recent["k"].sum() / recent["pa"].sum(), prior["k"].sum() / prior["pa"].sum()
        return dict(team_id=int(tid), k_move=float(np.clip(kr / kp - 1, -0.35, 0.35)), k_30=kr, pa_30=float(recent["pa"].sum()))
    with ThreadPoolExecutor(workers) as ex:
        return pd.DataFrame(list(ex.map(one, [int(t) for t in team_ids])))


def _split_row(pid, group, start30, end30) -> dict:
        params = dict(stats="statSplits,byDateRange,season", group=group, season=2026, sitCodes="vl,vr")
        if start30: params.update(startDate=start30, endDate=end30)
        j = _get(f"{API}/people/{int(pid)}/stats", **params)
        rec = dict(mlbam_id=int(pid))
        for st in j.get("stats", []):
            nm = st["type"]["displayName"]
            for s in st.get("splits", []):
                w, pa, k = woba_from_stat(s["stat"])
                if nm == "statSplits":
                    c = s["split"]["code"]; rec[f"woba_{c}"] = w; rec[f"pa_{c}"] = pa; rec[f"k_{c}"] = k
                elif nm == "byDateRange" and "woba_30" not in rec:
                    rec["woba_30"] = w; rec["pa_30"] = pa; rec["g_30"] = s["stat"].get("gamesPlayed", 0)
                    if group == "pitching": rec["gs_30"] = s["stat"].get("gamesStarted", 0); rec["ip_30"] = s["stat"].get("inningsPitched")
                elif nm == "season" and "woba_season" not in rec:
                    rec["woba_season"] = w; rec["pa_season"] = pa; rec["k_season"] = k; rec["g_season"] = s["stat"].get("gamesPlayed", 0)
        return rec


def player_splits(ids, group="hitting", start30=None, end30=None, workers: int = 12) -> pd.DataFrame:
    """L/R splits, last-30-days and season lines for a list of players. Threaded: 500 hitters used to take two and a half
    minutes one at a time, which was most of the pipeline's running time."""
    ids = [int(i) for i in ids]
    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(lambda p: _split_row(p, group, start30, end30), ids))
    return pd.DataFrame(rows)


def _ip(v):
    a, _, b = str(v).partition("."); return int(a or 0) + int(b or 0) / 3


def _log_row(pid: int, season: int, W: dict, asof: str | None) -> dict:
    j = _get(f"{API}/people/{int(pid)}/stats", stats="gameLog", group="pitching", season=season)
    sp = (j.get("stats") or [{}])[0].get("splits", [])
    sp = sorted(sp, key=lambda s: str(s.get("date") or ""))
    def pts_of(st):
        ip = _ip(st.get("inningsPitched", 0)); er = st.get("earnedRuns", 0)
        return (W["IP"] * ip + W["K"] * st.get("strikeOuts", 0) + W["ER"] * er + W["QS"] * (1 if ip >= 6 and er <= 3 else 0)
                + W["CG"] * st.get("completeGames", 0) + W["SHO"] * st.get("shutouts", 0) + W["BS"] * st.get("blownSaves", 0))
    starts = [s for s in sp if s["stat"].get("gamesStarted") == 1]
    pts = [pts_of(s["stat"]) for s in starts]; n = len(starts)
    last_start = str(starts[-1].get("date"))[:10] if n else None
    last_app = str(sp[-1].get("date"))[:10] if sp else None
    # appearances out of the bullpen since his last start: the plain signal that a starter has been moved
    relief_after = sum(1 for s in sp if s["stat"].get("gamesStarted") != 1 and last_start and str(s.get("date"))[:10] > last_start)
    cut = (date.fromisoformat(asof) - timedelta(days=30)).isoformat() if asof else None
    starts_30 = sum(1 for s in starts if cut and str(s.get("date"))[:10] >= cut)
    allg = [dict(date=str(x.get("date"))[:10], opp=((x.get("opponent") or {}).get("abbreviation") or (x.get("opponent") or {}).get("name")),
                 home=(x.get("isHome") if x.get("isHome") is not None else None), ip=round(_ip(x["stat"].get("inningsPitched", 0)), 1),
                 K=x["stat"].get("strikeOuts", 0), BB=x["stat"].get("baseOnBalls", 0), H=x["stat"].get("hits", 0), ER=x["stat"].get("earnedRuns", 0),
                 gs=int(x["stat"].get("gamesStarted") == 1), pts=round(pts_of(x["stat"]), 1)) for x in sp]
    recent = allg[-10:]
    summary = rolling(allg, asof); summary["starts"] = n
    summary["pts_start"] = round(sum(p for p in pts) / n, 1) if n else None
    return dict(mlbam_id=int(pid), n_starts=n, g_total=len(sp), n_relief=len(sp) - n,
                start_dates=[str(x.get("date"))[:10] for x in starts], recent=recent, summary=summary,
                pts_start=(sum(pts) / n if n else np.nan), ip_start=(sum(_ip(s["stat"].get("inningsPitched", 0)) for s in starts) / n if n else np.nan),
                last5_pts=(sum(pts[-5:]) / len(pts[-5:]) if n else np.nan), last3_pts=(sum(pts[-3:]) / len(pts[-3:]) if n else np.nan),
                k_start=(sum(s["stat"].get("strikeOuts", 0) for s in starts) / n if n else np.nan),
                ip_start_3=(np.mean([_ip(s["stat"].get("inningsPitched", 0)) for s in starts[-3:]]) if n else np.nan),
                last_start=last_start, last_app=last_app, relief_after_start=relief_after, starts_30=starts_30,
                days_since_start=((date.fromisoformat(asof) - date.fromisoformat(last_start)).days if (asof and last_start) else np.nan))


def starter_logs(ids, season=2026, weights=None, asof: str | None = None, workers: int = 10) -> pd.DataFrame:
    """Starts-only line per pitcher from MLB game logs: points per start, IP per start, recent form, and the role signals
    (relief appearances since his last start, days since his last start, starts in the last 30 days)."""
    W = dict(IP=1, K=1, ER=-1, QS=4, CG=10, SHO=8, BS=-3); W.update(weights or {})
    ids = [int(i) for i in ids]
    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(lambda p: _log_row(p, season, W, asof), ids))
    return pd.DataFrame(rows)


WHIFF = ("swinging_strike", "swinging_strike_blocked", "foul_tip", "missed_bunt")
SWING = WHIFF + ("foul", "foul_bunt", "hit_into_play", "hit_into_play_score", "hit_into_play_no_out")
FASTBALL = ("FF", "SI", "FC", "FA")


COLS_P = ["game_date", "pitcher", "pitch_type", "release_speed", "description"]


def _pitches(ids, gt: str, lt: str) -> pd.DataFrame:
    q = "".join(f"&pitchers_lookup%5B%5D={int(i)}" for i in ids)
    u = ("https://baseballsavant.mlb.com/statcast_search/csv?all=true&hfGT=R%7C&hfSea=2026%7C&player_type=pitcher"
         f"{q}&game_date_gt={gt}&game_date_lt={lt}&type=details")
    for attempt in range(3):
        try:
            r = _S.get(u, timeout=240)
            if r.ok and len(r.content) > 200:
                return pd.read_csv(io.StringIO(r.content.decode()), low_memory=False, usecols=lambda c: c in COLS_P)
        except Exception:
            pass
        time.sleep(4 + 4 * attempt)
    return pd.DataFrame(columns=COLS_P)


PITCH_CACHE = DATA / "statcast" / "pitches"
SEASON_OPEN = "2026-03-18"


def pitch_cache(ids, end: str, group: int = 80) -> pd.DataFrame:
    """Every pitch these starters have thrown this season, cached one week at a time under data/statcast/pitches.

    A finished week is never re-downloaded; only the week in progress is, and only the pitchers a cached week has never
    been asked about are fetched for it. So the first build pays for the season (about twelve minutes) and every build
    after it pays for one partial week (about thirty seconds), which is what the daily job actually costs."""
    PITCH_CACHE.mkdir(parents=True, exist_ok=True)
    ids = sorted({int(i) for i in ids}); end_d = date.fromisoformat(end)
    weeks = []; d0 = date.fromisoformat(SEASON_OPEN)
    while d0 <= end_d:
        d1 = min(d0 + timedelta(days=6), end_d)
        weeks.append((d0.isoformat(), d1.isoformat())); d0 = d1 + timedelta(days=1)
    def grab(who, gt, lt):
        parts = [c for i in range(0, len(who), group) for c in [_pitches(who[i:i + group], gt, lt)] if len(c)]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLS_P)
    frames = []; full = 0; topup = 0
    for gt, lt in weeks:
        f = PITCH_CACHE / f"{gt}.parquet"; fi = PITCH_CACHE / f"{gt}.ids"
        done = lt < end                       # a week that has finished never changes again
        if f.exists() and done:
            d = pd.read_parquet(f)
            asked = set(json.loads(fi.read_text())) if fi.exists() else set(d["pitcher"].astype(int))
            missing = sorted(set(ids) - asked)
            missing = [i for i in missing if i not in set(d["pitcher"].astype(int))]
            if missing:                        # a starter the cache has never been asked about: fetch only him
                add = grab(missing, gt, lt)
                if len(add): d = pd.concat([d, add], ignore_index=True)
                d.to_parquet(f, index=False); fi.write_text(json.dumps(sorted(asked | set(ids)))); topup += 1
            frames.append(d); continue
        d = grab(ids, gt, lt)
        d.to_parquet(f, index=False); fi.write_text(json.dumps(ids)); full += 1
        frames.append(d)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(out):
        out["game_date"] = pd.to_datetime(out["game_date"])
        out = out[out["pitcher"].isin(ids)]   # each week holds each pitcher exactly once, so no dedupe (two identical-looking pitches in one game are two pitches)
    print(f"  statcast cache: {len(weeks)} weeks ({full} downloaded, {topup} topped up, {len(weeks)-full-topup} reused), {len(out):,} pitches")
    return out


def recent_stuff(ids, end: str, start_dates: dict | None = None, days: int = 30) -> pd.DataFrame:
    """Velocity, CSW% and whiff rate over the last 30 days against the same pitcher's season, **counting only his starts**
    when we know which games those were. Relief outings are a different animal (one inning, max effort), so mixing them
    into a starter's baseline hides exactly the change this is meant to catch.

    Also flags a pitch he is throwing now and was not throwing earlier in the year."""
    d = pitch_cache(ids, end)
    if not len(d): return pd.DataFrame(columns=["mlbam_id", "velo_30", "csw_30", "whiff_30", "pitches_30"])
    if start_dates:
        keys = {(int(p), pd.Timestamp(x)) for p, ds in start_dates.items() for x in ds}
        st = pd.Series(list(zip(d["pitcher"].astype(int), d["game_date"])), index=d.index).isin(keys)
        if st.sum() > 0.3 * len(d): d = d[st]     # only trust the filter if it keeps most of the data
    cut = pd.Timestamp(date.fromisoformat(end) - timedelta(days=days))
    recent = stuff_features(d[d["game_date"] >= cut]).rename(columns={"pitcher": "mlbam_id"})
    season = stuff_features(d).rename(columns={"pitcher": "mlbam_id"}).rename(columns={"velo_30": "velo_st", "csw_30": "csw_st", "whiff_30": "whiff_st", "pitches_30": "pitches_st"})
    out = season.merge(recent, on="mlbam_id", how="left")
    out["dvelo"] = (out["velo_30"] - out["velo_st"]).clip(-4, 4)
    out["dcsw"] = (out["csw_30"] - out["csw_st"]).clip(-12, 12)
    out["dwhiff"] = (out["whiff_30"] - out["whiff_st"]).clip(-15, 15)
    out = out.merge(new_pitches(d, cut), on="mlbam_id", how="left")
    return out


def new_pitches(d: pd.DataFrame, cut) -> pd.DataFrame:
    """Pitch types making up 4%+ of his last 30 days that were under 1% before: a genuinely new offering."""
    rows = []
    for pid, g in d.groupby("pitcher"):
        new_, old = g[g["game_date"] >= cut], g[g["game_date"] < cut]
        if len(new_) < 100 or len(old) < 200: rows.append(dict(mlbam_id=int(pid), new_pitch=None)); continue
        un, uo = new_["pitch_type"].value_counts(normalize=True), old["pitch_type"].value_counts(normalize=True)
        got = [f"{t} {un[t]*100:.0f}%" for t in un.index if un[t] >= 0.04 and uo.get(t, 0) < 0.01]
        rows.append(dict(mlbam_id=int(pid), new_pitch=", ".join(got) or None))
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["mlbam_id", "new_pitch"])


def stuff_features(d: pd.DataFrame) -> pd.DataFrame:
    """Per-pitcher velocity / CSW / whiff from a frame of pitches (used live and in the backtest)."""
    d = d.copy(); d["is_fb"] = d["pitch_type"].isin(FASTBALL)
    fb = d[d["is_fb"]]
    # his own most-thrown fastball, so a pitcher who mixes four-seam and sinker is measured on one of them
    top = fb.groupby(["pitcher", "pitch_type"]).size().reset_index(name="n").sort_values("n", ascending=False).drop_duplicates("pitcher")
    velo = fb.merge(top[["pitcher", "pitch_type"]], on=["pitcher", "pitch_type"]).groupby("pitcher")["release_speed"].mean()
    g = d.groupby("pitcher")
    out = pd.DataFrame({
        "velo_30": velo,
        "csw_30": 100 * g["description"].apply(lambda s: s.isin(WHIFF + ("called_strike",)).mean()),
        "whiff_30": 100 * g["description"].apply(lambda s: s.isin(WHIFF).sum() / max(1, s.isin(SWING).sum())),
        "pitches_30": g.size(),
    }).reset_index()
    return out


HIT_W = dict(x1b=2, x2b=3, x3b=4, hr=5, r=1, rbi=1, bb=1, hbp=1, sb=3)


def rolling(games: list[dict], asof: str | None) -> dict:
    """League points over the season and over the last 7 / 14 / 30 days, with the games that made them."""
    out = dict(pts=round(sum(g["pts"] for g in games), 1), g=len(games))
    if not asof: return out
    a = date.fromisoformat(asof)
    for d in (7, 14, 30):
        cut = (a - timedelta(days=d)).isoformat()
        w = [g for g in games if g["date"] >= cut]
        out[f"pts{d}"] = round(sum(g["pts"] for g in w), 1); out[f"g{d}"] = len(w)
    return out


def _hit_log(pid: int, season: int, n: int, asof: str | None) -> dict:
    j = _get(f"{API}/people/{int(pid)}/stats", stats="gameLog", group="hitting", season=season)
    sp = sorted((j.get("stats") or [{}])[0].get("splits", []), key=lambda x: str(x.get("date") or ""))
    out = []
    for x in sp:
        st = x["stat"]; h = st.get("hits", 0); d = st.get("doubles", 0); t = st.get("triples", 0); hr = st.get("homeRuns", 0)
        pts = (HIT_W["x1b"] * (h - d - t - hr) + HIT_W["x2b"] * d + HIT_W["x3b"] * t + HIT_W["hr"] * hr + HIT_W["r"] * st.get("runs", 0)
               + HIT_W["rbi"] * st.get("rbi", 0) + HIT_W["bb"] * st.get("baseOnBalls", 0) + HIT_W["hbp"] * st.get("hitByPitch", 0)
               + HIT_W["sb"] * st.get("stolenBases", 0))
        out.append(dict(date=str(x.get("date"))[:10], opp=((x.get("opponent") or {}).get("abbreviation") or (x.get("opponent") or {}).get("name")),
                        home=x.get("isHome"), PA=st.get("plateAppearances", 0), AB=st.get("atBats", 0), H=h, HR=hr, R=st.get("runs", 0),
                        RBI=st.get("rbi", 0), BB=st.get("baseOnBalls", 0), K=st.get("strikeOuts", 0), SB=st.get("stolenBases", 0), pts=round(pts, 1)))
    return dict(mlbam_id=int(pid), recent=out[-n:], summary=rolling(out, asof))


def hitter_logs(ids, season=2026, n=10, asof: str | None = None, workers: int = 12) -> dict:
    """Last `n` games per hitter plus his league-points totals — what the page leads with when you open a player."""
    ids = [int(i) for i in ids]
    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(lambda p: _hit_log(p, season, n, asof), ids))
    return {str(r["mlbam_id"]): dict(r=r["recent"], s=r["summary"]) for r in rows if r["recent"]}


def bullpen_moved(logs: pd.DataFrame, asof: str | None = None, gap_days: int = 12) -> set:
    """Starters who are not in a rotation any more: their most recent outing was in relief, or they have not started in
    almost two turns. A listed probable always overrides this — it only stops the model from inventing a start."""
    if not len(logs): return set()
    d = logs.copy()
    moved = d["relief_after_start"].fillna(0) > 0
    stale = d["days_since_start"].fillna(0) >= gap_days
    return set(d.loc[moved | stale, "mlbam_id"].astype(int))


# ---------------------------------------------------------------- scoring
def hitter_matchups(games: pd.DataFrame, hitters: pd.DataFrame, hsplits: pd.DataFrame, pitchers: pd.DataFrame, psplits: pd.DataFrame,
                    pf: pd.DataFrame, wx: pd.DataFrame, ven: pd.DataFrame, weights: dict | None = None) -> pd.DataFrame:
    W = dict(sp_power=0.6, park_power=1.0, temp_per_deg=0.003, wind_per_mph=0.004, home=1.02, form_power=0.5, platoon_power=1.0)
    if weights: W.update(weights)
    h = hitters.merge(hsplits, on="mlbam_id", how="left")
    # baseline rate: half actual pts/PA, half expected (xLP) pts/PA, shrunk to the league mean by PA.
    #
    # Both the prior and the shrinkage were wrong until 2026-09-14, and in opposite directions.
    #   prior: hardcoded 0.90 against a real league rate of 0.963 this season — every hitter 7% low.
    #   k: 250, when regressing rest-of-season rate on prior rate over 44,107 hitter-games implies a k near 950,
    #      and remarkably flat across sample sizes: 20-80 PA slope 0.057 (k 972), 80-150 slope 0.130 (k 844),
    #      150-300 slope 0.178 (k 934), 300-600 slope 0.237 (k 1112).
    # The practical effect is that a thin hot sample carried far too much: hitters under 150 PA sitting on 1.29 points
    # per PA went on to hit 0.969, barely above league average, while the model had them at 0.98 and climbing.
    base_rate = 0.5 * h["pts_pa"].fillna(h["xLP_pa"]) + 0.5 * h["xLP_pa"].fillna(h["pts_pa"])
    lg_rate = float(np.nansum(base_rate * h["PA"]) / np.nansum(np.where(base_rate.notna(), h["PA"], np.nan))) if h["PA"].notna().any() else 0.963
    if not np.isfinite(lg_rate) or not (0.7 < lg_rate < 1.3): lg_rate = 0.963
    h["base_rate"] = shrink(base_rate, h["PA"], lg_rate, 900)
    h["pa_g"] = shrink(h["pa_30"] / h["g_30"].replace(0, np.nan), h["g_30"], (h["PA"] / h["G"].replace(0, np.nan)).fillna(3.8), 10)
    h["form"] = (shrink(h["woba_30"], h["pa_30"], h["woba_season"].fillna(LG_WOBA), 100) / h["woba_season"].fillna(LG_WOBA).clip(0.2, 0.5)).clip(0.8, 1.2)
    # platoon: hitter's wOBA vs L / vs R relative to his overall, shrunk
    for c in ("vl", "vr"):
        h[f"plat_{c}"] = (shrink(h[f"woba_{c}"], h[f"pa_{c}"], h["woba_season"].fillna(LG_WOBA), 200) / h["woba_season"].fillna(LG_WOBA).clip(0.2, 0.5)).clip(0.85, 1.15)
    p = pitchers.merge(psplits, on="mlbam_id", how="left")
    p["sp_xwoba"] = shrink(p["xwoba"], p["BF"], LG_XWOBA_P, 300)
    for c in ("vl", "vr"):  # SP wOBA allowed vs L/R relative to his overall (regressed)
        p[f"sp_plat_{c}"] = (shrink(p[f"woba_{c}"], p[f"pa_{c}"], p["woba_season"].fillna(LG_WOBA), 250) / p["woba_season"].fillna(LG_WOBA).clip(0.2, 0.5)).clip(0.85, 1.15)
    pidx = p.set_index("mlbam_id")
    pfa = pf.set_index(["venue_id", "bat_side"])
    wxi = wx.set_index("gamePk") if len(wx) else pd.DataFrame()
    veni = ven.set_index("venue_id")
    rows = []
    gteam = games.copy()
    for _, g in gteam.iterrows():
        opp_sp = g["opp_sp_id"]; sp = pidx.loc[opp_sp] if pd.notna(opp_sp) and opp_sp in pidx.index else None
        sp_throws = (sp["throws"] if sp is not None else None) or "R"
        v = veni.loc[g["venue_id"]] if g["venue_id"] in veni.index else None
        w = wxi.loc[g["gamePk"]] if len(wxi) and g["gamePk"] in wxi.index else None
        roof_closed = v is not None and str(v.get("roof", "")).lower() in ("dome", "retractable", "fixed")
        tm = 1.0; wind_c = 0.0
        if w is not None and not roof_closed:
            tm = 1 + W["temp_per_deg"] * (float(w["temp_f"]) - 70); tm = float(np.clip(tm, 0.94, 1.06))
            wind_c = wind_component(w["wind_dir"], v["azimuth"] if v is not None else np.nan) * float(w["wind_mph"])
            tm *= float(np.clip(1 + W["wind_per_mph"] * wind_c, 0.92, 1.08))
        cand = h[h["team_id"] == g["team_id"]]
        for _, r in cand.iterrows():
            side = "L" if r.get("bats") == "L" else ("R" if r.get("bats") == "R" else ("L" if sp_throws == "R" else "R"))
            vs = "vl" if sp_throws == "L" else "vr"
            spf = 1.0
            if sp is not None:
                spf = (float(sp["sp_xwoba"]) / LG_XWOBA_P) * float(sp[f"sp_plat_{'vl' if side=='L' else 'vr'}"])
            spf = float(np.clip(spf, 0.75, 1.3)) ** W["sp_power"]
            pfk = (g["venue_id"], side); park = (pfa.loc[pfk]["pf_runs"] / 100) if pfk in pfa.index else 1.0
            park = float(park) ** W["park_power"]
            plat = float(r[f"plat_{vs}"]) ** W["platoon_power"] if pd.notna(r.get(f"plat_{vs}")) else 1.0
            form = float(r["form"]) ** W["form_power"] if pd.notna(r.get("form")) else 1.0
            homef = W["home"] if g["home"] else 1.0
            mult = spf * park * tm * plat * form * homef
            exp = float(r["base_rate"]) * float(r["pa_g"]) * mult
            rows.append(dict(mlbam_id=r["mlbam_id"], name=r["name"], team=g["team"], bats=r.get("bats"), elig=r.get("elig"), owner=r.get("owner"),
                             date=g["date"], gamePk=g["gamePk"], home=g["home"], opp=g["opp"], venue=g["venue"],
                             opp_sp=g["opp_sp_name"], opp_sp_throws=sp_throws, opp_sp_source=g["opp_sp_source"],
                             sp_xwoba=(float(sp["sp_xwoba"]) if sp is not None else np.nan), sp_k=(float(sp["k_percent"]) if sp is not None and pd.notna(sp.get("k_percent")) else np.nan),
                             sp_pitching_plus=(float(sp["pitching_plus"]) if sp is not None and pd.notna(sp.get("pitching_plus")) else np.nan),
                             park_runs=(int(pfa.loc[pfk]["pf_runs"]) if pfk in pfa.index else np.nan), park_hr=(int(pfa.loc[pfk]["pf_hr"]) if pfk in pfa.index else np.nan),
                             temp_f=(float(w["temp_f"]) if w is not None else np.nan), wind_mph=(float(w["wind_mph"]) if w is not None else np.nan), wind_out=round(wind_c, 1),
                             precip_prob=(float(w["precip_prob"]) if w is not None else np.nan), roof=(v["roof"] if v is not None else None), local_start=(w["local_start"] if w is not None else None),
                             base_rate=round(float(r["base_rate"]), 3), pa_g=round(float(r["pa_g"]), 2), f_sp=round(spf, 3), f_park=round(park, 3), f_wx=round(tm, 3), f_platoon=round(plat, 3), f_form=round(form, 3), f_home=homef,
                             mult=round(mult, 3), exp_pts=round(exp, 2), woba_30=r.get("woba_30"), pa_30=r.get("pa_30")))
    return pd.DataFrame(rows)


# Points in a neutral matchup for one start, fitted on 3,212 real 2026 starts (predicting each start from what was known
# before it, five-fold by pitcher so a pitcher never trains and tests together, non-negative weights).
#   r 0.367, MAE 4.48 — against r 0.315, MAE 4.67 for the old "season rate blended with the model, shrunk toward 7.0".
# What it says:
#  - a starter's own fantasy scoring rate adds nothing once you know his stuff and how deep he goes;
#  - recent RESULTS are worse than the season line (last three starts alone: r 0.241 vs 0.288 for season to date);
#  - recent STUFF does carry, but only measured over his STARTS: velocity is worth 0.84 points a start per mph that way
#    against 0.15 when relief outings are mixed in, because a reliever's max-effort inning is not his starting velocity;
#  - starts-only SKILL levels (CSW, K-BB, velo over his starts) predict worse than season Pitching+ across every
#    appearance (r 0.303 vs 0.364), so the level term stays season-long and only the trend is filtered to starts.
#
# Which of the metrics analysts actually name predict THIS league's points, one at a time against the next start:
#   Pitching+ .332 | xwOBA against .327 | Stuff+ .320 | strikeouts per start .291 | his own points per start .294
#   K-BB% .260 | K% .247 | IP per start .229 | SwStr% .227 | CSW% .212 | whiff% .205 | ERA .197 | BB% .105
# Two league-specific readings of that list: strikeouts per START beat K% (.291 vs .247) because this league pays a
# point per strikeout rather than rewarding a rate, and BB% barely registers because there is no walk penalty at all —
# walks only cost through baserunners and shortened outings. K-BB% is the analysts' simple workhorse and it does beat
# K% here, but a blend with contact quality beats both: adding xwOBA to the ridge is what earned its place, while
# strikeouts per start, quality-start rate and K-BB% all took a zero weight once Pitching+ and innings were in.
BASE = dict(b0=0.27, pplus=0.1096, ip_mix=1.2026, dcsw=0.0390, dvelo=0.8078, xwoba=-26.698)
LG_XWOBA_AGAINST = 0.312
SKILL_RATE = (0.1936, -9.980)   # Pitching+ -> points per start, the anchor a thin sample is shrunk toward
LG_IP_GS = 5.2


def _base_rate(p: pd.DataFrame) -> pd.DataFrame:
    """Expected points in a neutral matchup, per start."""
    this_rate = p["pts_start"].fillna(p["pts_gs"]); n = p["n_starts"].fillna(p["GS"]).fillna(0)
    ip_season = p["ip_start"].fillna(p["IP"] / p["GS"].replace(0, np.nan))
    ip_mix = (0.5 * ip_season.fillna(LG_IP_GS) + 0.5 * p.get("ip_start_3", ip_season).fillna(ip_season).fillna(LG_IP_GS))
    # thin samples lean on the league's length rather than on three starts of noise
    p["ip_mix"] = (ip_mix * n + LG_IP_GS * 3) / (n + 3)
    pplus = p["pitching_plus"]
    for c in ("dvelo", "dcsw", "dwhiff"):
        if c not in p.columns: p[c] = np.nan
    xw = p["xwoba"].fillna(LG_XWOBA_AGAINST) if "xwoba" in p.columns else pd.Series(LG_XWOBA_AGAINST, index=p.index)
    base = (BASE["b0"] + BASE["pplus"] * pplus + BASE["ip_mix"] * p["ip_mix"] + BASE["dcsw"] * p["dcsw"].fillna(0)
            + BASE["dvelo"] * p["dvelo"].fillna(0) + BASE["xwoba"] * xw)
    # no Savant line (a rookie under the 50-IP cutoff): his own rate, shrunk toward whatever skills we do have
    anchor = np.where(pplus.notna(), SKILL_RATE[0] * pplus + SKILL_RATE[1], 7.0)
    shrunk = (this_rate.fillna(7.0) * n + anchor * 8) / (n + 8)
    p["base_gs"] = np.where(pplus.notna(), base, 0.935 * shrunk + 0.558)
    return p


def lineup_strength(hitters: pd.DataFrame, hsplits: pd.DataFrame, min_pa_split: int = 30) -> dict:
    """Each club's wOBA against a hand, weighted by the plate appearances its hitters are ACTUALLY taking now.

    A club's season split mixes in plate appearances from players who have since been traded, hurt or sent down. This
    re-weights the same season splits by who is playing, which separates "the personnel changed" from "they have been
    hot", the distinction that made strikeout-rate recency work and wOBA recency fail.

    DISPLAY ONLY. Measured on 3,072 starts it does not predict better than the club's season split (r .3783 vs .3782,
    paired MAE +0.002 with a CI straddling zero), there is no gain in the tail where the shift is largest, and none in
    September when rosters churn. So it is shown, not applied — a big gap is worth a human knowing about, and Bort can
    act on it with the lineup controls. Do not wire this into exp_pts without a fresh test that actually passes."""
    h = hitters.merge(hsplits, on="mlbam_id", how="left")
    if "pa_30" not in h.columns or "team_id" not in h.columns: return {}
    out = {}
    for tid, g in h.groupby("team_id"):
        for hand in ("vl", "vr"):
            w, pa = g.get(f"woba_{hand}"), g.get(f"pa_{hand}")
            if w is None or pa is None: continue
            use = g[w.notna() & (pa.fillna(0) >= min_pa_split) & (g["pa_30"].fillna(0) > 0)]
            if not len(use): continue
            wt = use["pa_30"].astype(float)
            out[(int(tid), hand)] = float((use[f"woba_{hand}"].astype(float) * wt).sum() / wt.sum())
    return out


def pitcher_starts(games: pd.DataFrame, pitchers: pd.DataFrame, psplits: pd.DataFrame, tsplits: pd.DataFrame, pf: pd.DataFrame, wx: pd.DataFrame, ven: pd.DataFrame,
                   weights: dict | None = None, tform: pd.DataFrame | None = None, lstr: dict | None = None) -> pd.DataFrame:
    # matchup strengths checked against 3,072 real 2026 starts (below): the opponent's bat plays bigger than 2.5 implied
    # and the strikeout bonus slightly smaller. Home stays at 1.03 because the park factor already carries most of it.
    # k_move_w: how much of a club's recent strikeout-rate movement to carry. Half, from team_form's note above -
    # 0.5 was the best of 0.25/0.5/0.75/1.0 both in sample (r .3939 vs .3921) and out of it, and full weight overshoots.
    W = dict(opp_per_woba=3.5, park_power=0.5, wx_power=0.5, home=1.03, k_weight=0.85, k_move_w=0.5)
    if weights: W.update(weights)
    tfi = tform.set_index("team_id") if tform is not None and len(tform) else None
    p = pitchers.merge(psplits, on="mlbam_id", how="left")
    p = _base_rate(p)
    this_n = p["n_starts"].fillna(p["GS"]).fillna(0)
    p["k_per_gs"] = shrink(p["k_start"].fillna(p["K"] / p["GS"].replace(0, np.nan)), this_n, 5.0, 10)
    ip_gs = p["ip_start"].fillna(p["IP"] / p["GS"].replace(0, np.nan))
    p["role"] = np.where(this_n < 3, "spot/relief", np.where(ip_gs < 3.8, "opener/bulk", "starter"))
    pidx = p.set_index("mlbam_id"); ts = tsplits.set_index(["team_id", "vs"]); pfa = pf.set_index(["venue_id", "bat_side"]); veni = ven.set_index("venue_id")
    wxi = wx.set_index("gamePk") if len(wx) else pd.DataFrame()
    rows = []
    for _, g in games.dropna(subset=["sp_id"]).iterrows():
        if g["sp_id"] not in pidx.index:
            continue
        sp = pidx.loc[g["sp_id"]]; hand = "vl" if (sp.get("throws") == "L") else "vr"
        key_t = (g["opp_id"], hand); tw = float(ts.loc[key_t]["t_woba"]) if key_t in ts.index else LG_WOBA; tk = float(ts.loc[key_t]["t_k"]) if key_t in ts.index else LG_K
        # the club's bat stays on the season split (recency there is pure noise); its strikeout rate gets a half-weight
        # nudge for how the last 30 days compare with the rest of its season
        # who is actually taking the opponent's plate appearances, against the club's season split. Shown, never applied.
        lw = (lstr or {}).get((int(g["opp_id"]), hand))
        lshift = round(lw - tw, 4) if lw is not None else np.nan
        kmv = 0.0
        if tfi is not None and g["opp_id"] in tfi.index:
            m = tfi.loc[g["opp_id"]]["k_move"]
            if pd.notna(m): kmv = float(m); tk = tk * (1 + W["k_move_w"] * kmv)
        oppf = float(np.clip(1 - W["opp_per_woba"] * (tw - LG_WOBA), 0.7, 1.3))
        park = pfa.loc[(g["venue_id"], "All")]["pf_runs"] / 100 if (g["venue_id"], "All") in pfa.index else 1.0
        parkf = float((1 / park) ** W["park_power"])
        v = veni.loc[g["venue_id"]] if g["venue_id"] in veni.index else None; w = wxi.loc[g["gamePk"]] if len(wxi) and g["gamePk"] in wxi.index else None
        roof_closed = v is not None and str(v.get("roof", "")).lower() in ("dome", "retractable", "fixed")
        tm = 1.0; wind_c = 0.0
        if w is not None and not roof_closed:
            tm = float(np.clip(1 + 0.003 * (float(w["temp_f"]) - 70), 0.94, 1.06)); wind_c = wind_component(w["wind_dir"], v["azimuth"] if v is not None else np.nan) * float(w["wind_mph"]); tm *= float(np.clip(1 + 0.004 * wind_c, 0.92, 1.08))
        wxf = float((1 / tm) ** W["wx_power"]); homef = W["home"] if g["home"] else 1.0
        kbonus = W["k_weight"] * float(sp["k_per_gs"]) * (tk / LG_K - 1)
        exp = float(sp["base_gs"]) * oppf * parkf * wxf * homef + kbonus
        rows.append(dict(mlbam_id=g["sp_id"], name=g["sp_name"], team=g["team"], throws=sp.get("throws"), owner=sp.get("owner"), date=g["date"], gamePk=g["gamePk"], home=g["home"], opp=g["opp"], venue=g["venue"],
                         sp_source=g["sp_source"], opp_woba_vs_hand=round(tw, 3), opp_k_vs_hand=round(tk, 3), opp_k_move=round(kmv, 3),
                         opp_lineup_woba=(round(lw, 3) if lw is not None else np.nan), opp_lineup_shift=lshift, park_runs=(int(park * 100)), temp_f=(float(w["temp_f"]) if w is not None else np.nan),
                         wind_out=round(wind_c, 1), precip_prob=(float(w["precip_prob"]) if w is not None else np.nan), local_start=(w["local_start"] if w is not None else None),
                         base_gs=round(float(sp["base_gs"]), 2),
                         # STARTS ONLY. pts_gs is season points over games started, which for a swingman divides his
                         # relief points by his start count: Ian Seymour showed 17.0 a start on 14 starts and 44
                         # appearances when his actual starts-only rate was 10.7, and the model (which already used
                         # pts_start) looked wrong next to a number that was itself wrong.
                         pts_gs_2026=(round(float(sp["pts_start"]), 2) if pd.notna(sp.get("pts_start"))
                                      else (round(float(sp["pts_gs"]), 2) if pd.notna(sp.get("pts_gs")) else np.nan)),
                         pts_all_gs=(round(float(sp["pts_gs"]), 2) if pd.notna(sp.get("pts_gs")) else np.nan),
                         n_relief=(int(sp["n_relief"]) if pd.notna(sp.get("n_relief")) else 0),
                         GS=int(sp["GS"]) if pd.notna(sp.get("GS")) else 0,
                         ip_mix=(round(float(sp["ip_mix"]), 2) if pd.notna(sp.get("ip_mix")) else np.nan), velo_30=(round(float(sp["velo_30"]), 1) if pd.notna(sp.get("velo_30")) else np.nan),
                         dvelo=(round(float(sp["dvelo"]), 1) if pd.notna(sp.get("dvelo")) else np.nan), whiff_30=(round(float(sp["whiff_30"]), 1) if pd.notna(sp.get("whiff_30")) else np.nan),
                         dwhiff=(round(float(sp["dwhiff"]), 1) if pd.notna(sp.get("dwhiff")) else np.nan), csw_30=(round(float(sp["csw_30"]), 1) if pd.notna(sp.get("csw_30")) else np.nan),
                         last3_pts=(round(float(sp["last3_pts"]), 1) if pd.notna(sp.get("last3_pts")) else np.nan), ip_start_3=(round(float(sp["ip_start_3"]), 1) if pd.notna(sp.get("ip_start_3")) else np.nan),
                         relief_after_start=int(sp.get("relief_after_start") or 0), days_since_start=(int(sp["days_since_start"]) if pd.notna(sp.get("days_since_start")) else np.nan),
                         dcsw=(round(float(sp["dcsw"]), 1) if pd.notna(sp.get("dcsw")) else np.nan), velo_st=(round(float(sp["velo_st"]), 1) if pd.notna(sp.get("velo_st")) else np.nan),
                         csw_st=(round(float(sp["csw_st"]), 1) if pd.notna(sp.get("csw_st")) else np.nan), whiff_st=(round(float(sp["whiff_st"]), 1) if pd.notna(sp.get("whiff_st")) else np.nan),
                         new_pitch=(sp.get("new_pitch") if isinstance(sp.get("new_pitch"), str) else None),
                         pitching_plus=sp.get("pitching_plus"), stuff_plus=sp.get("stuff_plus"), xera=sp.get("xera"), k_percent=sp.get("k_percent"),
                         xwoba_against=(round(float(sp["xwoba"]), 3) if pd.notna(sp.get("xwoba")) else np.nan),
                         k_start=(round(float(sp["k_start"]), 1) if pd.notna(sp.get("k_start")) else np.nan),
                         woba_30=sp.get("woba_30"), gs_30=sp.get("gs_30"), role=sp.get("role"), ip_gs=(round(float(sp["ip_start"]), 1) if pd.notna(sp.get("ip_start")) else np.nan), n_starts=(int(sp["n_starts"]) if pd.notna(sp.get("n_starts")) else 0), last5_pts=(round(float(sp["last5_pts"]), 1) if pd.notna(sp.get("last5_pts")) else np.nan), f_opp=round(oppf, 3), f_park=round(parkf, 3), f_wx=round(wxf, 3), f_home=homef, k_bonus=round(kbonus, 2), exp_pts=round(exp, 2)))
    out = pd.DataFrame(rows)
    if len(out):
        n = out.groupby("mlbam_id")["gamePk"].transform("count"); out["two_start"] = n >= 2
        out["week_pts"] = out.groupby("mlbam_id")["exp_pts"].transform("sum")
    return out

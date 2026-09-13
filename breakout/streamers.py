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
import json, math, time
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


def project_rotations(sch: pd.DataFrame, recent: pd.DataFrame, min_apps: int = 2, regular_ids=None, active_ids=None, exclude_ids=None) -> pd.DataFrame:
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
        if len(regs) < 3:
            regs = order[:5]
        rot = list(reversed(regs))  # oldest -> most recent
        last = rot[-1][0]
        idx = [r[0] for r in rot].index(last)
        for gi in grp.index:
            if pd.notna(sch.at[gi, "sp_id"]):
                # a listed probable resets the cycle position if he is in the rotation
                pid = sch.at[gi, "sp_id"]
                if pid in [r[0] for r in rot]:
                    idx = [r[0] for r in rot].index(pid)
                continue
            idx = (idx + 1) % len(rot)
            sch.at[gi, "sp_id"], sch.at[gi, "sp_name"], sch.at[gi, "sp_source"] = rot[idx][0], rot[idx][1], "projected"
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
    return pd.DataFrame(rows)


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


def player_splits(ids, group="hitting", start30=None, end30=None) -> pd.DataFrame:
    rows = []
    for pid in ids:
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
        rows.append(rec); time.sleep(0.08)
    return pd.DataFrame(rows)


def _ip(v):
    a, _, b = str(v).partition("."); return int(a or 0) + int(b or 0) / 3


def starter_logs(ids, season=2026, weights=None) -> pd.DataFrame:
    """Starts-only line per pitcher from MLB game logs: starts, points per start in league scoring, IP per start, last-5 form."""
    W = dict(IP=1, K=1, ER=-1, QS=4, CG=10, SHO=8, BS=-3); W.update(weights or {})
    rows = []
    for pid in ids:
        j = _get(f"{API}/people/{int(pid)}/stats", stats="gameLog", group="pitching", season=season)
        sp = (j.get("stats") or [{}])[0].get("splits", [])
        starts = [s for s in sp if s["stat"].get("gamesStarted") == 1]
        pts = []
        for s in starts:
            st = s["stat"]; ip = _ip(st.get("inningsPitched", 0)); er = st.get("earnedRuns", 0)
            pts.append(W["IP"] * ip + W["K"] * st.get("strikeOuts", 0) + W["ER"] * er + W["QS"] * (1 if ip >= 6 and er <= 3 else 0) + W["CG"] * st.get("completeGames", 0) + W["SHO"] * st.get("shutouts", 0) + W["BS"] * st.get("blownSaves", 0))
        n = len(starts)
        rows.append(dict(mlbam_id=int(pid), n_starts=n, g_total=len(sp), pts_start=(sum(pts) / n if n else np.nan), ip_start=(sum(_ip(s["stat"].get("inningsPitched", 0)) for s in starts) / n if n else np.nan),
                         last5_pts=(sum(pts[-5:]) / len(pts[-5:]) if n else np.nan), k_start=(sum(s["stat"].get("strikeOuts", 0) for s in starts) / n if n else np.nan)))
        time.sleep(0.06)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- scoring
def hitter_matchups(games: pd.DataFrame, hitters: pd.DataFrame, hsplits: pd.DataFrame, pitchers: pd.DataFrame, psplits: pd.DataFrame,
                    pf: pd.DataFrame, wx: pd.DataFrame, ven: pd.DataFrame, weights: dict | None = None) -> pd.DataFrame:
    W = dict(sp_power=0.6, park_power=1.0, temp_per_deg=0.003, wind_per_mph=0.004, home=1.02, form_power=0.5, platoon_power=1.0)
    if weights: W.update(weights)
    h = hitters.merge(hsplits, on="mlbam_id", how="left")
    # baseline rate: half actual pts/PA, half expected (xLP) pts/PA, shrunk to league mean by PA
    base_rate = 0.5 * h["pts_pa"].fillna(h["xLP_pa"]) + 0.5 * h["xLP_pa"].fillna(h["pts_pa"])
    h["base_rate"] = shrink(base_rate, h["PA"], 0.90, 250)
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


def pitcher_starts(games: pd.DataFrame, pitchers: pd.DataFrame, psplits: pd.DataFrame, tsplits: pd.DataFrame, pf: pd.DataFrame, wx: pd.DataFrame, ven: pd.DataFrame,
                   weights: dict | None = None) -> pd.DataFrame:
    W = dict(opp_per_woba=2.5, park_power=0.5, wx_power=0.5, home=1.03, k_weight=1.0)
    if weights: W.update(weights)
    p = pitchers.merge(psplits, on="mlbam_id", how="left")
    # base rate: this season's pts/GS blended with the model's rate, shrunk toward a replacement-level 7.0 with the prior season's
    # starts counting as extra sample (a pitcher with 8 GS in 2026 and 25 last year is not a 8-GS unknown)
    # starts-only rates from game logs (relief innings are excluded, so swingmen are not inflated)
    this_rate = p["pts_start"].fillna(p["pts_gs"]); this_n = p["n_starts"].fillna(p["GS"]).fillna(0)
    prior_rate = p["pts_start_prev"].fillna(p["pts_gs_prev"]).fillna(7.0); prior_n = p["n_starts_prev"].fillna(p["GS_prev"]).fillna(0).clip(0, 20)
    model = p["proj_pts_gs_raw"]
    blend = np.where(model.notna(), 0.6 * this_rate.fillna(7.0) + 0.4 * model.fillna(7.0), this_rate.fillna(7.0))
    p["base_gs"] = (blend * this_n + prior_rate * prior_n + 7.0 * 10) / (this_n + prior_n + 10)
    # recent form nudges the base a little (last five starts vs season)
    form = (p["last5_pts"] - this_rate).fillna(0).clip(-3, 3)
    p["base_gs"] = p["base_gs"] + 0.25 * form
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
                         sp_source=g["sp_source"], opp_woba_vs_hand=round(tw, 3), opp_k_vs_hand=round(tk, 3), park_runs=(int(park * 100)), temp_f=(float(w["temp_f"]) if w is not None else np.nan),
                         wind_out=round(wind_c, 1), precip_prob=(float(w["precip_prob"]) if w is not None else np.nan), local_start=(w["local_start"] if w is not None else None),
                         base_gs=round(float(sp["base_gs"]), 2), pts_gs_2026=round(float(sp["pts_gs"]), 2) if pd.notna(sp.get("pts_gs")) else np.nan, GS=int(sp["GS"]) if pd.notna(sp.get("GS")) else 0,
                         pitching_plus=sp.get("pitching_plus"), stuff_plus=sp.get("stuff_plus"), xera=sp.get("xera"), k_percent=sp.get("k_percent"),
                         woba_30=sp.get("woba_30"), gs_30=sp.get("gs_30"), role=sp.get("role"), ip_gs=(round(float(sp["ip_start"]), 1) if pd.notna(sp.get("ip_start")) else np.nan), n_starts=(int(sp["n_starts"]) if pd.notna(sp.get("n_starts")) else 0), last5_pts=(round(float(sp["last5_pts"]), 1) if pd.notna(sp.get("last5_pts")) else np.nan), f_opp=round(oppf, 3), f_park=round(parkf, 3), f_wx=round(wxf, 3), f_home=homef, k_bonus=round(kbonus, 2), exp_pts=round(exp, 2)))
    out = pd.DataFrame(rows)
    if len(out):
        n = out.groupby("mlbam_id")["gamePk"].transform("count"); out["two_start"] = n >= 2
        out["week_pts"] = out.groupby("mlbam_id")["exp_pts"].transform("sum")
    return out

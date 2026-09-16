"""Daily build of the Home Run Board.   python -m hrboard.build [--date YYYY-MM-DD]

Writes site/hr/data/hr.js (today's and tomorrow's slates, the model pieces the page needs to re-run the numbers when
lineups or weather change, market prices and the scorecard) and logs every pick to data/hrboard/picks/<date>.json so
it can be graded once the games are played.
"""
from __future__ import annotations
import argparse, json, math, time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

from breakout.config import DATA
from breakout.names import key as name_key
from breakout import streamers as ST
from . import features as F, dataset as D, game as G, model as M, context as C, odds as O, savant, scorecard as SC, books as BK

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site" / "hr"
PICKS = DATA / "hrboard" / "picks"
ET = ZoneInfo("America/New_York")
API = "https://statsapi.mlb.com/api/v1"
ORDER = list(M.ORDER)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def platt(p, ab):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return 1 / (1 + np.exp(-(ab[0] + ab[1] * np.log(p / (1 - p)))))


# ------------------------------------------------------------------ inputs
def load_history(today: str) -> pd.DataFrame:
    yday = (date.fromisoformat(today) - timedelta(days=1)).isoformat()
    parts = [savant.load(2024, refresh_open=False), savant.load(2025, refresh_open=False), savant.load(2026, end=yday)]
    d = F.prepare(pd.concat(parts, ignore_index=True))
    return d[d["game_date"] < pd.Timestamp(today)]


def fetch_games(day: str) -> list[dict]:
    j = C._get(f"{API}/schedule", sportId=1, date=day, gameType="R",
               hydrate="probablePitcher,venue,weather,lineups,team,linescore")
    out = []
    for dd in j.get("dates", []):
        for g in dd["games"]:
            if g.get("gameType") != "R": continue
            out.append(g)
    return out


def rosters(team_ids) -> pd.DataFrame:
    def one(tid):
        j = C._get(f"{API}/teams/{tid}/roster", rosterType="active", hydrate="person")
        return [dict(team_id=tid, pid=r["person"]["id"], name=r["person"]["fullName"], pos=r["position"]["abbreviation"],
                     ptype=r["position"]["type"], bats=(r["person"].get("batSide") or {}).get("code", "R"),
                     throws=(r["person"].get("pitchHand") or {}).get("code", "R")) for r in j.get("roster", [])]
    with ThreadPoolExecutor(8) as ex:
        rows = [x for part in ex.map(one, list(team_ids)) for x in part]
    return pd.DataFrame(rows)


def people(ids) -> pd.DataFrame:
    ids = [int(i) for i in ids if pd.notna(i)]
    rows = []
    for i in range(0, len(ids), 50):
        j = C._get(f"{API}/people", personIds=",".join(map(str, ids[i:i + 50])))
        rows += [dict(pid=p["id"], name=p["fullName"], bats=(p.get("batSide") or {}).get("code", "R"),
                      throws=(p.get("pitchHand") or {}).get("code", "R")) for p in j.get("people", [])]
    return pd.DataFrame(rows, columns=["pid", "name", "bats", "throws"])


def recent_lineups(d: pd.DataFrame) -> pd.DataFrame:
    """Every 2026 starting lineup, with the hand of the starter it faced."""
    x = d[d["season"] == d["season"].max()]
    st = G.starters(x)
    hands = x.groupby("opp_sp")["p_throws"].agg(lambda s: s.mode().iat[0]).rename("opp_hand")
    return st.merge(hands, left_on="opp_sp", right_index=True, how="left")


def projected(st: pd.DataFrame, team: str, hand: str, active: set) -> pd.DataFrame:
    """P(starts) and usual slot for each hitter, from the club's last 12 games and its last 20 against this hand."""
    t = st[st["bat_team"] == team]
    gl = t.drop_duplicates("game_pk").sort_values("game_date")
    last = set(gl["game_pk"].tail(12)); lh = set(gl.loc[gl["opp_hand"] == hand, "game_pk"].tail(20))
    a = t[t["game_pk"].isin(last)].groupby("batter").agg(n_all=("game_pk", "nunique"), slot_all=("slot", "mean"), last=("game_date", "max"))
    h = t[t["game_pk"].isin(lh)].groupby("batter").agg(n_hand=("game_pk", "nunique"), slot_hand=("slot", "mean"))
    p = a.join(h, how="outer").fillna({"n_all": 0, "n_hand": 0})
    s_all = p["n_all"] / max(len(last), 1)
    s_hand = p["n_hand"] / max(len(lh), 1)
    p["p_start"] = np.where(len(lh) >= 4, 0.5 * s_all + 0.5 * s_hand, s_all)
    # somebody who has not started in the last week is probably not in there now
    p.loc[p["last"] < gl["game_date"].tail(7).min(), "p_start"] *= 0.5
    p["slot_mean"] = p["slot_hand"].fillna(p["slot_all"])
    p = p[p.index.isin(active)].reset_index()
    p = p.sort_values(["p_start", "last"], ascending=False)
    top = p.head(9).sort_values("slot_mean")
    top["proj_slot"] = np.arange(1, len(top) + 1)
    p = p.merge(top[["batter", "proj_slot"]], on="batter", how="left")
    p["proj_slot"] = p["proj_slot"].fillna(p["slot_mean"].round().clip(1, 9)).fillna(9).astype(int)
    return p


def forecast(games: list[dict], ven: pd.DataFrame) -> pd.DataFrame:
    """Open-Meteo hourly forecast for every park in ONE request (it takes coordinate lists), averaged over the first
    three hours from first pitch."""
    import requests
    vi = ven.set_index("venue_id")
    g = [(x["gamePk"], x["venue"]["id"], pd.Timestamp(x["gameDate"])) for x in games if x["venue"]["id"] in vi.index]
    vids = sorted({v for _, v, _ in g})
    if not vids: return pd.DataFrame(columns=["gamePk", "temp_f", "wind_mph", "wind_dir", "precip_prob"])
    t0 = min(t for *_, t in g).date(); t1 = (max(t for *_, t in g) + pd.Timedelta(hours=4)).date()
    params = dict(latitude=",".join(str(vi.at[v, "lat"]) for v in vids), longitude=",".join(str(vi.at[v, "lon"]) for v in vids),
                  hourly="temperature_2m,wind_speed_10m,wind_direction_10m,precipitation_probability", temperature_unit="fahrenheit",
                  wind_speed_unit="mph", timezone="GMT", start_date=str(t0), end_date=str(t1))
    js = None
    for i in range(3):
        try:
            r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=25); r.raise_for_status(); js = r.json(); break
        except Exception:
            time.sleep(2 + 2 * i)
    rows = []
    if js is None:
        log("  weather: Open-Meteo unreachable, the page will fetch it live")
        return pd.DataFrame(columns=["gamePk", "temp_f", "wind_mph", "wind_dir", "precip_prob"])
    js = js if isinstance(js, list) else [js]
    byv = dict(zip(vids, js))
    for pk, v, t in g:
        h = byv[v].get("hourly") or {}
        times = pd.to_datetime(h.get("time", []), utc=True)
        if not len(times): continue
        i = int(np.searchsorted(times, t.floor("h")))
        sl = slice(i, min(i + 3, len(times)))
        def m(k):
            a = np.array(h.get(k, [])[sl], dtype=float); return float(np.nanmean(a)) if len(a) and not np.all(np.isnan(a)) else np.nan
        pp = np.array(h.get("precipitation_probability", [])[sl], dtype=float)
        rows.append(dict(gamePk=pk, temp_f=m("temperature_2m"), wind_mph=m("wind_speed_10m"),
                         wind_dir=float(h["wind_direction_10m"][i]) if i < len(times) and h["wind_direction_10m"][i] is not None else np.nan,
                         precip_prob=float(np.nanmax(pp)) if len(pp) and not np.all(np.isnan(pp)) else np.nan))
    return pd.DataFrame(rows)


def venue_weather(games: list[dict], ven: pd.DataFrame) -> pd.DataFrame:
    """First-pitch temperature and the wind blowing out, from Open-Meteo; MLB's own reading wins once it is posted."""
    if not games: return pd.DataFrame()
    wx = forecast(games, ven)
    out = []
    vi = ven.set_index("venue_id")
    for x in games:
        pk = x["gamePk"]; vid = x["venue"]["id"]
        v = vi.loc[vid] if vid in vi.index else None
        roof = str(v["roof"]).lower() if v is not None else ""
        w = wx[wx["gamePk"] == pk]
        temp = float(w["temp_f"].iat[0]) if len(w) and pd.notna(w["temp_f"].iat[0]) else np.nan
        mph = float(w["wind_mph"].iat[0]) if len(w) and pd.notna(w["wind_mph"].iat[0]) else np.nan
        comp = ST.wind_component(w["wind_dir"].iat[0], v["azimuth"] if v is not None else np.nan) if len(w) else 0.0
        pop = float(w["precip_prob"].iat[0]) if len(w) and pd.notna(w["precip_prob"].iat[0]) else np.nan
        src = "forecast"
        mw = C.parse_weather(x.get("weather"))
        if pd.notna(mw["temp_f"]):
            temp, src = mw["temp_f"], "MLB"
            if pd.notna(mw["wind_mph"]): mph, comp = mw["wind_mph"], (mw["wind_out"] / mw["wind_mph"] if mw["wind_mph"] else 0.0)
        if roof == "dome":
            indoor, roof_note = True, "dome"
        elif roof == "retractable":
            if mw["wx_cond"]:
                indoor = bool(mw["indoor"]); roof_note = "roof closed" if indoor else "roof open"
            else:
                indoor = bool((pop == pop and pop >= 35) or (temp == temp and (temp >= 88 or temp <= 55)))
                roof_note = "roof likely closed" if indoor else "roof likely open"
        else:
            indoor, roof_note = False, ""
        out.append(dict(game_pk=pk, venue_id=vid, temp_f=temp, wind_mph=mph, wind_out=(0.0 if indoor else (mph * comp if mph == mph else 0.0)),
                        wind_comp=comp, precip=pop, indoor=indoor, roof=roof_note, wx_src=src, wx_cond=mw["wx_cond"],
                        ))
    return pd.DataFrame(out)


# ------------------------------------------------------------------ one slate
def build_slate(day: str, d: pd.DataFrame, ctx: dict) -> dict | None:
    games = fetch_games(day)
    if not games: return None
    tt = time.time()
    mj, book, bpbook, bshare, work, stl = ctx["mj"], ctx["book"], ctx["bpbook"], ctx["bshare"], ctx["work"], ctx["starts"]
    T = mj["tables"]; q = np.array(T["q"]); curve = {int(k): v for k, v in T["curve"].items()}; resid = {int(k): v for k, v in T["resid"].items()}
    team_ids = sorted({x["teams"][s]["team"]["id"] for x in games for s in ("home", "away")})
    ros = rosters(team_ids)
    log(f"    rosters {time.time()-tt:.0f}s")
    wx = venue_weather(games, ctx["venues"])
    log(f"    weather {time.time()-tt:.0f}s")
    parks = ctx["parks"]
    season = int(day[:4]); gd = pd.Timestamp(day)
    sp_ids = [x["teams"][s].get("probablePitcher", {}).get("id") for x in games for s in ("home", "away")]
    sp_info = people([i for i in sp_ids if i])
    hitters, gmeta = [], []
    for x in games:
        pk = x["gamePk"]; status = x["status"]["detailedState"]; abstract = x["status"]["abstractGameState"]
        for side, other in (("home", "away"), ("away", "home")):
            t = x["teams"][side]; o = x["teams"][other]
            abbr = t["team"]["abbreviation"]; oabbr = o["team"]["abbreviation"]
            sp = o.get("probablePitcher") or {}
            spid = sp.get("id"); sp_throw = "R"
            if spid is not None and (sp_info["pid"] == spid).any():
                sp_throw = sp_info.loc[sp_info["pid"] == spid, "throws"].iat[0]
            lineup = [p["id"] for p in (x.get("lineups") or {}).get(f"{side}Players", [])]
            act = ros[(ros["team_id"] == t["team"]["id"]) & ((ros["ptype"] != "Pitcher") | (ros["pos"] == "TWP"))]
            proj = projected(stl, abbr, sp_throw, set(act["pid"]))
            pi = proj.set_index("batter")
            ids = list(dict.fromkeys(list(act["pid"]) + lineup))
            for pid in ids:
                r = act[act["pid"] == pid]
                name = r["name"].iat[0] if len(r) else None
                bats = r["bats"].iat[0] if len(r) else None
                in_l = pid in lineup
                if lineup:
                    p_start = 1.0 if in_l else 0.0; slot = lineup.index(pid) + 1 if in_l else int(pi["proj_slot"].get(pid, 9)) if pid in pi.index else 9
                else:
                    p_start = float(pi["p_start"].get(pid, 0.0)) if pid in pi.index else 0.0
                    slot = int(pi["proj_slot"].get(pid, 9)) if pid in pi.index else 9
                hitters.append(dict(pid=pid, name=name, bats=bats, team=abbr, opp=oabbr, game_pk=pk, is_home=int(side == "home"),
                                    sp=spid, sp_throws=sp_throw, lineup_posted=bool(lineup), in_lineup=in_l, slot=slot,
                                    p_start=p_start))
            gmeta.append(dict(game_pk=pk, side=side, team=abbr, team_id=t["team"]["id"], team_name=t["team"]["teamName"],
                              sp=spid, sp_name=sp.get("fullName"), sp_throws=sp_throw, lineup=lineup, status=status, abstract=abstract))
    H = pd.DataFrame(hitters)
    log(f"    lineups {time.time()-tt:.0f}s")
    miss = H[H["name"].isna()]["pid"].unique()
    if len(miss):
        pp = people(miss).set_index("pid")
        H.loc[H["name"].isna(), "bats"] = H.loc[H["name"].isna(), "pid"].map(pp["bats"])
        H.loc[H["name"].isna(), "name"] = H.loc[H["name"].isna(), "pid"].map(pp["name"])
    H["bats"] = H["bats"].fillna("R")
    H["stand_sp"] = G.stand_vs(H["bats"], H["sp_throws"])
    H["stand_bp"] = np.where(H["bats"] == "S", "L", H["bats"])
    H["season"] = season; H["game_date"] = gd
    # skills and design rows
    sch = wx.rename(columns={})[["game_pk", "venue_id", "temp_f", "wind_out", "indoor"]].assign(day_night=None)
    qs = pd.DataFrame(dict(batter=H["pid"], pitcher=H["sp"].astype("float").fillna(-1).astype(int), season=season, game_date=gd,
                           p_throws=H["sp_throws"], stand=H["stand_sp"], game_pk=H["game_pk"], is_home=H["is_home"]))
    s_sp = D.attach_context(D.skill_rows(book, qs), sch, parks)
    X_sp = D.design(s_sp)
    qb = pd.DataFrame(dict(batter=H["pid"], pitcher=H["opp"], season=season, game_date=gd, p_throws="R", stand=H["stand_bp"],
                           game_pk=H["game_pk"], is_home=H["is_home"]))
    b = book.batter_features(qb[["batter", "season", "game_date", "p_throws"]])
    p = bpbook.pitcher_features(qb[["pitcher", "season", "game_date", "stand"]])
    lgb = F.attach_league(qb, book.lga).add_prefix("lg_")
    s_bp = D.attach_context(pd.concat([qb, b, p, lgb], axis=1), sch, parks)
    X_bp = D.design(s_bp)
    shr = bshare[bshare["season"] == season].sort_values("game_date").groupby(["fld_team", "stand"])["same_share"].last()
    X_bp["platoon"] = [shr.get((t, s), 0.62 if s == "R" else 0.30) for t, s in zip(H["opp"], H["stand_bp"])]
    O_sp, O_bp = D.offsets(s_sp), D.offsets(s_bp)
    log(f"    skills {time.time()-tt:.0f}s")
    cols = mj["columns"]; W = np.array(mj["coef"]); bvec = np.array(mj["intercept"])
    ci = [mj["classes"].index(c) for c in ORDER]
    iw = [cols.index("temp"), cols.index("wind_out")]
    def z_no_wx(X, Oo):
        X0 = X[cols].to_numpy(dtype="float64").copy(); X0[:, iw] = 0.0
        return (X0 @ W.T + bvec + Oo)[:, ci]
    H_zsp, H_zbp = z_no_wx(X_sp, O_sp), z_no_wx(X_bp, O_bp)
    P_sp = M.predict(X_sp, mj, O_sp); P_bp = M.predict(X_bp, mj, O_bp)
    # starter workload
    bfx = {}
    for spid in H["sp"].dropna().unique():
        bfx[int(spid)] = bf_expect(work, int(spid), season)
    H["bf_exp"] = [bfx.get(int(s), G.LG_BF) if pd.notna(s) else G.LG_BF for s in H["sp"]]
    r = G.batter_game(P_sp, P_bp, H["slot"], H["is_home"], H["bf_exp"], q, curve)
    cal = mj["calib"]
    H["p_hr"] = platt(r["p_hr"], cal["hr"]) * H["p_start"]
    H["p_hit"] = platt(r["p_hit"], cal["hit"]) * H["p_start"]
    H["p_tb2"] = platt(r["p_tb2"], cal["tb2"]) * H["p_start"]
    H["p_hr_if_starts"] = platt(r["p_hr"], cal["hr"]); H["e_pa"] = r["e_pa"]
    H["pa_hr_sp"] = P_sp[:, ORDER.index("hr")]; H["pa_hr_bp"] = P_bp[:, ORDER.index("hr")]
    H["zsp"] = list(np.round(H_zsp, 4)); H["zbp"] = list(np.round(H_zbp, 4))
    # skill snapshot for the cards
    snap = s_sp[["b_hr", "b_brl", "b_pull_air", "b_hard", "b_bat_speed", "b_pa_eff", "bh_hr", "ps_hr", "p_brl", "p_gb", "p_k",
                 "pf_hr", "lg_hr", "lg_brl"]].reset_index(drop=True)
    H = pd.concat([H.reset_index(drop=True), snap], axis=1)
    # ---- pitchers
    SP = []
    for gm in gmeta:
        # the starter pitching TO this side's hitters is the other side's probable, stored on this side's row
        spid = gm["sp"]
        lh = H[(H["game_pk"] == gm["game_pk"]) & (H["team"] == gm["team"])]
        order = lineup_order(lh)
        if spid is None or len(order) < 9: continue
        idx = [H.index.get_loc(i) for i in order.index[:9]]
        res = G.pitcher_ks(P_sp[idx, ORDER.index("k")], bfx.get(int(spid), G.LG_BF), resid)
        ge = {n: float(platt(v, cal["k"])) for n, v in res["p_ge"].items()}
        SP.append(dict(pid=int(spid), name=gm["sp_name"], throws=gm["sp_throws"], game_pk=gm["game_pk"], vs=gm["team"],
                       e_k=round(res["e_k"], 2), bf_exp=round(bfx.get(int(spid), G.LG_BF), 3), p_ge={str(k): round(v, 4) for k, v in ge.items()},
                       lineup_ids=[int(H.at[i, "pid"]) for i in order.index[:9]]))
    SPd = pd.DataFrame(SP)
    # ---- team runs and win probability
    runs = mj["runs"]; G_out = []
    for x in games:
        pk = x["gamePk"]; mu = {}
        for side in ("home", "away"):
            ab = x["teams"][side]["team"]["abbreviation"]
            lh = H[(H["game_pk"] == pk) & (H["team"] == ab)]
            order = lineup_order(lh)
            if len(order) < 9: continue
            idx = [H.index.get_loc(i) for i in order.index[:9]]
            share = float(np.clip(H.loc[order.index[0], "bf_exp"] / 38.5, 0.2, 0.95))
            e_pa = r["e_pa"][idx]
            rates = (e_pa[:, None] * (share * P_sp[idx] + (1 - share) * P_bp[idx])).sum(0) / e_pa.sum()
            mu[side] = runs["a"] + runs["b"] * G.team_runs_rate(rates)
        if len(mu) < 2: continue
        wp = G.win_prob(mu["home"], mu["away"], runs["r"])
        G_out.append(dict(game_pk=pk, mu_home=mu["home"], mu_away=mu["away"], p_home=float(platt(wp["p_home"], cal["win"])),
                          total_pmf=wp["total_pmf"][:31].round(5).tolist()))
    Gd = pd.DataFrame(G_out)
    # ---- markets
    log(f"    model {time.time()-tt:.0f}s")
    mk = markets(day, games, H, SPd, ctx)
    mk = pregame_markets(day, games, H, SPd, mk)
    log(f"    markets {time.time()-tt:.0f}s")
    return assemble(day, games, gmeta, H, SPd, Gd, wx, mk, ctx)


def lineup_order(lh: pd.DataFrame) -> pd.DataFrame:
    """The nine the model uses for a club: the posted lineup, else the projected one."""
    if not len(lh): return lh
    if lh["lineup_posted"].any():
        return lh[lh["in_lineup"]].sort_values("slot", kind="mergesort")
    top = lh.assign(_neg=-lh["p_start"]).sort_values(["_neg", "slot", "pid"], kind="mergesort").head(9)
    return top.sort_values(["slot", "pid"], kind="mergesort").drop(columns="_neg")


def bf_expect(work: pd.DataFrame, pid: int, season: int) -> float:
    w = work[work["pitcher"] == pid]
    cur = w[w["season"] == season]; prev = w[w["season"] == season - 1]
    base = (cur["bf"].sum() + 0.5 * prev["bf"].sum() + 3 * G.LG_BF) / (len(cur) + 0.5 * len(prev) + 3)
    if len(cur):
        base = 0.6 * base + 0.4 * cur.sort_values("game_date")["bf"].tail(3).mean()
    return float(base)


# ------------------------------------------------------------------ markets
def markets(day, games, H, SPd, ctx) -> dict:
    out = dict(games={}, players={}, errors=[])
    gdf = pd.DataFrame([dict(game_pk=x["gamePk"], home_abbr=x["teams"]["home"]["team"]["abbreviation"],
                             away_abbr=x["teams"]["away"]["team"]["abbreviation"], time=x["gameDate"]) for x in games])
    try:
        eg = O.espn_games(day)
    except Exception as e:
        eg = pd.DataFrame(); out["errors"].append(f"espn games: {e}")
    if len(eg):
        eg = eg.merge(gdf, on=["home_abbr", "away_abbr"], how="inner")
        eg["dt"] = (pd.to_datetime(eg["date"], utc=True) - pd.to_datetime(eg["time"], utc=True)).abs()
        eg = eg.sort_values("dt").drop_duplicates("espn_id").drop_duplicates("game_pk")
        for r in eg.itertuples():
            out["games"].setdefault(int(r.game_pk), {})["dk"] = {k: (None if (isinstance(v, float) and math.isnan(v)) else v) for k, v in dict(
                espn_id=r.espn_id, home_ml=r.dk_home_ml, away_ml=r.dk_away_ml, home_p=r.dk_home_p, total=r.dk_total, over=r.dk_over,
                under=r.dk_under, home_rl=r.dk_home_rl, home_rl_odds=r.dk_home_rl_odds, away_rl_odds=r.dk_away_rl_odds,
                home_ml_open=r.dk_home_ml_open, links=r.dk_links, url=(r.dk_links or {}).get("event")).items()}
        try:
            props = O.espn_props(eg["espn_id"])
            ros = ctx["espn_rosters"]
            if len(props) and len(ros):
                props = props.merge(ros[["espn_athlete", "nkey", "team_abbr"]], on="espn_athlete", how="left")
                props = props.merge(eg[["espn_id", "game_pk"]], on="espn_id", how="left")
                idmap = player_keys(H, SPd)
                props["pid"] = [idmap.get((g, k)) for g, k in zip(props["game_pk"], props["nkey"])]
                for r in props.dropna(subset=["pid"]).itertuples():
                    kk = f"{r.market}{int(r.line)}"
                    if r.market != "k" and kk not in KEEP: continue
                    out["players"].setdefault(str(int(r.pid)), {}).setdefault("dk", {})[kk] = [r.dk_odds, _f(r.dk_p)]
                out["dk_props_espn_map"] = {str(int(p)): a for p, a in props.dropna(subset=["pid"])[["pid", "espn_athlete"]].drop_duplicates().itertuples(index=False)}
        except Exception as e:
            out["errors"].append(f"espn props: {e}")
    try:
        km = O.kalshi_markets(day)
        kg, kp = O.kalshi_split(km, gdf)
        for r in kg.itertuples():
            g = out["games"].setdefault(int(r.game_pk), {}).setdefault("ks", {})
            g.setdefault("urls", {})[r.series] = r.url
            if r.series == "game":
                g.setdefault("win", {})[r.team] = [_f(r.price), _f(r.bid), _f(r.ask), _f(r.volume), r.ticker]
            else:
                g.setdefault("total", {})[f"{r.line:g}"] = [_f(r.price), _f(r.bid), _f(r.ask), r.ticker]
        if len(kp):
            for r in kp.drop_duplicates(["game_pk", "series"]).itertuples():
                out["games"].setdefault(int(r.game_pk), {}).setdefault("ks", {}).setdefault("urls", {})[r.series] = r.url
            idmap = player_keys(H, SPd)
            kp["pid"] = [idmap.get((g, k)) for g, k in zip(kp["game_pk"], kp["nkey"])]
            for r in kp.dropna(subset=["pid", "line"]).itertuples():
                kk = f"{r.series}{int(r.line)}"
                if r.series != "k" and kk not in KEEP: continue
                out["players"].setdefault(str(int(r.pid)), {}).setdefault("ks", {})[kk] = [_f(r.price), _f(r.bid), _f(r.ask), r.ticker]
            out["kalshi_unmatched"] = int(kp["pid"].isna().sum())
        out["ks_fee"] = ctx.get("ks_fee") or {}
    except Exception as e:
        out["errors"].append(f"kalshi: {e}")
    try:
        names = {x["gamePk"]: (x["teams"]["away"]["team"]["name"], x["teams"]["home"]["team"]["name"]) for x in games}
        pm = O.polymarket_games(day, gdf)
        idmap = player_keys(H, SPd)
        for pk, v in pm.items():
            aw, hm = names.get(pk, ("", ""))
            gm = out["games"].setdefault(pk, {})
            gm["pm"] = dict(url=v["url"], slug=v["slug"], props_url=v.get("props_url"), props_slug=v.get("props_slug"),
                            ml=_pm_ml(v["ml"], aw, hm), totals={k: dict(over=_pm_px(t["over"]), under=_pm_px(t["under"]), slug=t["slug"])
                                                             for k, t in v["totals"].items()})
            for nm, mks in v["props"].items():
                pid = idmap.get((pk, name_key(nm)))
                if pid is None: continue
                for kk, (px, ask, bid, slug) in mks.items():
                    if not kk.startswith("k") and kk not in KEEP: continue
                    q = _pm_px([px, ask, bid])
                    if q is None: continue
                    out["players"].setdefault(str(pid), {}).setdefault("pm", {})[kk] = q + [slug]
    except Exception as e:
        out["errors"].append(f"polymarket: {e}")
    try:
        snap = BK.snapshot(day, gdf)
        for pk, books in (snap.get("games") or {}).items():
            out["games"].setdefault(int(pk), {})["books"] = dict(as_of=snap.get("as_of"), list=books)
    except Exception as e:
        out["errors"].append(f"books: {e}")
    out["as_of"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return out


def _pm_px(trio):
    """[market price, ask] from Polymarket's [price, ask, bid]. The ask is what one share costs right now; the market
    price is only kept when the book is two-sided and tight (bid and ask within 10 cents), otherwise it is noise."""
    px, ask, bid = (list(trio) + [None, None, None])[:3]
    if ask is None or ask >= 0.98 or ask <= 0: return None
    mkt = px if (bid is not None and bid > 0 and ask - bid <= 0.10) else None
    return [_f(mkt), _f(ask)]


def _pm_ml(ml, away_name, home_name):
    """Polymarket names the clubs; match each outcome to home or away by name rather than trusting the order."""
    if not ml or not ml.get("names"): return {}
    out = {"slug": ml.get("slug")}
    hn, an = home_name.lower(), away_name.lower()
    for nm, trio in zip(ml["names"], ml["prices"]):
        n = str(nm).lower()
        if n == hn: side = "home"
        elif n == an: side = "away"
        elif hn.split()[-1] != an.split()[-1]:          # "Sox" and "Sox" would be ambiguous
            side = "home" if n.split()[-1] == hn.split()[-1] else "away" if n.split()[-1] == an.split()[-1] else None
        else: side = None
        if side: out[side] = _pm_px(trio)
    return out


KEEP = {"hr1", "hr2", "hit1", "hit2", "tb2", "tb3"}
MK_CACHE = DATA / "hrboard" / "markets"


def pregame_markets(day, games, H, SPd, mk) -> dict:
    """Prices for a game are only comparable to a pre-game forecast if they were taken before first pitch. Keep the last
    pre-game snapshot of each game (data/hrboard/markets/<date>.json) and use it once the game is under way; a game
    first seen live gets no player or Kalshi prices at all (DraftKings' closing moneyline and total stay)."""
    MK_CACHE.mkdir(parents=True, exist_ok=True)
    for old in MK_CACHE.glob("*.json"):          # only today's and tomorrow's snapshots matter
        if old.stem < (date.fromisoformat(day) - timedelta(days=1)).isoformat(): old.unlink()
    f = MK_CACHE / f"{day}.json"
    cache = json.loads(f.read_text()) if f.exists() else {}
    who = {}
    for pid, pk in zip(H["pid"], H["game_pk"]): who.setdefault(int(pk), set()).add(str(int(pid)))
    if len(SPd):
        for pid, pk in zip(SPd["pid"], SPd["game_pk"]): who.setdefault(int(pk), set()).add(str(int(pid)))
    players = dict(mk["players"])
    for x in games:
        pk = x["gamePk"]; key = str(pk)
        started = x["status"]["abstractGameState"] != "Preview"
        gm = mk["games"].get(pk, {})
        if not started:
            cache[key] = dict(as_of=mk["as_of"], ks=gm.get("ks"), dk=gm.get("dk"), pm=gm.get("pm"), books=gm.get("books"),
                              players={p: players[p] for p in who.get(pk, ()) if p in players})
            continue
        old = cache.get(key)
        for p in who.get(pk, ()):
            players.pop(p, None)
        if old:
            for p, v in old["players"].items(): players[p] = v
            for k in ("ks", "pm", "books"):
                if old.get(k): gm[k] = old[k]
                else: gm.pop(k, None)
            if old.get("dk") and not gm.get("dk"): gm["dk"] = old["dk"]
            gm["pregame_as_of"] = old["as_of"]
        else:
            for k in ("ks", "pm"): gm.pop(k, None)
            gm["no_pregame"] = True
        mk["games"][pk] = gm
    mk["players"] = players
    f.write_text(json.dumps(cache, separators=(",", ":"), default=_json))
    return mk


def _f(x):
    try:
        x = float(x); return None if math.isnan(x) else round(x, 4)
    except (TypeError, ValueError):
        return None


def player_keys(H, SPd) -> dict:
    m = {(int(g), name_key(n)): int(p) for g, n, p in zip(H["game_pk"], H["name"], H["pid"]) if isinstance(n, str)}
    if len(SPd):
        for g, n, p in zip(SPd["game_pk"], SPd["name"], SPd["pid"]):
            if isinstance(n, str): m[(int(g), name_key(n))] = int(p)
    return m


# ------------------------------------------------------------------ payload
def _r(x, n=4):
    if x is None: return None
    try:
        if isinstance(x, (float, np.floating)) and math.isnan(x): return None
    except TypeError:
        pass
    return round(float(x), n)


def assemble(day, games, gmeta, H, SPd, Gd, wx, mk, ctx) -> dict:
    form = ctx["form"]
    wxi = wx.set_index("game_pk")
    gi = Gd.set_index("game_pk") if len(Gd) else pd.DataFrame()
    parks = ctx["parks_now"]
    G_list = []
    for x in games:
        pk = x["gamePk"]; w = wxi.loc[pk] if pk in wxi.index else None
        h, a = x["teams"]["home"], x["teams"]["away"]
        ls = x.get("linescore") or {}
        pf = parks[parks["venue_id"] == x["venue"]["id"]].set_index("stand")["pf_hr"].to_dict()
        G_list.append(dict(
            pk=pk, time=x["gameDate"], status=x["status"]["detailedState"], state=x["status"]["abstractGameState"],
            venue=x["venue"]["name"], venue_id=x["venue"]["id"], dh=x.get("doubleHeader"), game_no=x.get("gameNumber"),
            home=dict(abbr=h["team"]["abbreviation"], name=h["team"]["teamName"], full=h["team"]["name"], id=h["team"]["id"], score=h.get("score"),
                      rec=f'{(h.get("leagueRecord") or {}).get("wins", "")}-{(h.get("leagueRecord") or {}).get("losses", "")}',
                      sp=(h.get("probablePitcher") or {}).get("id"), sp_name=(h.get("probablePitcher") or {}).get("fullName")),
            away=dict(abbr=a["team"]["abbreviation"], name=a["team"]["teamName"], full=a["team"]["name"], id=a["team"]["id"], score=a.get("score"),
                      rec=f'{(a.get("leagueRecord") or {}).get("wins", "")}-{(a.get("leagueRecord") or {}).get("losses", "")}',
                      sp=(a.get("probablePitcher") or {}).get("id"), sp_name=(a.get("probablePitcher") or {}).get("fullName")),
            inning=ls.get("currentInningOrdinal"), half=ls.get("inningHalf"),
            wx=None if w is None else dict(temp=_r(w["temp_f"], 1), wind=_r(w["wind_mph"], 1), out=_r(w["wind_out"], 2), comp=_r(w["wind_comp"], 2),
                                           rain=_r(w["precip"], 0), indoor=bool(w["indoor"]), roof=w["roof"], src=w["wx_src"], cond=w["wx_cond"]),
            pf_hr=dict(L=_r(pf.get("L"), 2), R=_r(pf.get("R"), 2)),
            geo=_geo(ctx["venues"], x["venue"]["id"]),
            model=None if not len(gi) or pk not in gi.index else dict(mu_home=_r(gi.at[pk, "mu_home"], 2), mu_away=_r(gi.at[pk, "mu_away"], 2),
                                                                     p_home=_r(gi.at[pk, "p_home"], 4), total_pmf=gi.at[pk, "total_pmf"]),
            mk=mk["games"].get(pk, {})))
    fi = form.set_index("batter") if len(form) else pd.DataFrame()
    hitters = []
    for r in H.itertuples():
        if r.lineup_posted and not r.in_lineup:
            continue            # the lineup is out and he is not in it
        f = fi.loc[r.pid] if len(fi) and r.pid in fi.index else None
        hitters.append(dict(
            id=int(r.pid), n=r.name, b=r.bats, t=r.team, o=r.opp, pk=int(r.game_pk), h=int(r.is_home), sp=(None if pd.isna(r.sp) else int(r.sp)),
            spt=r.sp_throws, st=r.stand_sp, sl=int(r.slot), ps=_r(r.p_start, 3), il=bool(r.in_lineup), lp=bool(r.lineup_posted),
            bf=_r(r.bf_exp, 3), zsp=[_r(v, 4) for v in r.zsp], zbp=[_r(v, 4) for v in r.zbp],
            hr=_r(r.p_hr), hit=_r(r.p_hit), tb2=_r(r.p_tb2), epa=_r(r.e_pa, 2),
            sk=dict(hr=_r(r.b_hr * 600, 1), hrh=_r(r.bh_hr * 600, 1), brl=_r(r.b_brl * 100, 1), pull=_r(r.b_pull_air * 100, 1),
                    hard=_r(r.b_hard * 100, 1), bat=_r(r.b_bat_speed, 1), pa=_r(r.b_pa_eff, 0), p_hr=_r(r.ps_hr * 600, 1),
                    p_brl=_r(r.p_brl * 100, 1), p_gb=_r(r.p_gb * 100, 0), pf=_r(r.pf_hr, 2), lg_hr=_r(r.lg_hr * 600, 1), lg_brl=_r(r.lg_brl * 100, 1)),
            f=None if f is None else [int(f[k]) for k in FORM_KEYS],
            mk=mk["players"].get(str(int(r.pid)), {})))
    pitchers = []
    ps = ctx["pitcher_season"]
    for r in (SPd.itertuples() if len(SPd) else []):
        s = ps.get(int(r.pid), {})
        pitchers.append(dict(id=int(r.pid), n=r.name, th=r.throws, pk=int(r.game_pk), vs=r.vs, ek=r.e_k, bf=r.bf_exp, ge=r.p_ge,
                             lu=r.lineup_ids, s=s, mk=mk["players"].get(str(int(r.pid)), {})))
    return dict(date=day, games=G_list, hitters=hitters, pitchers=pitchers,
                markets=dict(as_of=mk["as_of"], errors=mk["errors"], kalshi_unmatched=mk.get("kalshi_unmatched"),
                             espn_map=mk.get("dk_props_espn_map", {}), ks_fee=mk.get("ks_fee", {})))


def _geo(ven, vid):
    v = ven[ven["venue_id"] == vid]
    if not len(v): return None
    v = v.iloc[0]
    return dict(lat=_r(v["lat"], 4), lon=_r(v["lon"], 4), az=_r(v["azimuth"], 1), roof=v["roof"], tz=v["tz"], city=v["city"])


def recent_form(d: pd.DataFrame, today: str) -> pd.DataFrame:
    """Last 7 / 14 days and season: HR, barrels, hard-hit balls. Shown on the page, not used in the number."""
    x = d[d["season"] == int(today[:4])]
    t = pd.Timestamp(today)
    out = x.groupby("batter").agg(s_pa=("pa", "sum"), s_hr=("hr", "sum"), s_brl=("brl", "sum"), s_bbe=("bbe", "sum"))
    for n in (7, 14):
        w = x[x["game_date"] >= t - pd.Timedelta(days=n)]
        a = w.groupby("batter").agg(**{f"pa{n}": ("pa", "sum"), f"hr{n}": ("hr", "sum"), f"brl{n}": ("brl", "sum"),
                                       f"bbe{n}": ("bbe", "sum"), f"hard{n}": ("hard", "sum")})
        out = out.join(a, how="left")
    out = out.fillna(0)
    out["hot"] = ((out["brl14"] >= 3) & (out["bbe14"] >= 12) & (out["brl14"] / out["bbe14"].clip(lower=1) >= 1.4 * (out["s_brl"] / out["s_bbe"].clip(lower=1)))
                  | (out["hr14"] >= 4)).astype(int)
    out["cold"] = ((out["pa14"] >= 30) & (out["brl14"] == 0) & (out["s_brl"] / out["s_bbe"].clip(lower=1) >= 0.08)).astype(int)
    for c in out.columns:
        if c not in ("hot", "cold"): out[c] = out[c].astype(int)
    return out.reset_index()


FORM_KEYS = ["s_pa", "s_hr", "s_brl", "s_bbe", "pa7", "hr7", "brl7", "bbe7", "pa14", "hr14", "brl14", "bbe14", "hard14", "hot", "cold"]


def pitcher_season(d: pd.DataFrame, season: int) -> dict:
    x = d[d["season"] == season]
    g = x.groupby("pitcher").agg(bf=("pa", "sum"), k=("k", "sum"), bb=("bb", "sum"), hr=("hr", "sum"), brl=("brl", "sum"),
                                 bbe=("bbe", "sum"), gb=("gb", "sum"))
    st = x[x["vs_sp"] == 1].groupby("pitcher")["game_pk"].nunique().rename("gs")
    g = g.join(st, how="left").fillna(0)
    return {int(i): dict(bf=int(r.bf), gs=int(r.gs), k=_r(r.k / max(r.bf, 1) * 100, 1), bb=_r(r.bb / max(r.bf, 1) * 100, 1),
                         hr9=_r(r.hr / max(r.bf, 1) * 38, 2), brl=_r(r.brl / max(r.bbe, 1) * 100, 1), gb=_r(r.gb / max(r.bbe, 1) * 100, 0))
            for i, r in g.iterrows()}


def model_block(mj: dict, report: dict) -> dict:
    cols = mj["columns"]; W = np.array(mj["coef"]); ci = [mj["classes"].index(c) for c in ORDER]
    return dict(order=ORDER, c_temp=W[ci, cols.index("temp")].round(5).tolist(), c_wind=W[ci, cols.index("wind_out")].round(5).tolist(),
                calib=mj["calib"], runs=mj["runs"], q=np.round(np.array(mj["tables"]["q"]), 4).tolist(),
                curve=mj["tables"]["curve"], resid=mj["tables"]["resid"], trained_on=mj["trained_on"],
                report=report)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--date"); ap.add_argument("--days", type=int, default=2)
    a = ap.parse_args(argv)
    t0 = time.time()
    today = a.date or datetime.now(ET).date().isoformat()
    log("slate", today)
    d = load_history(today)
    log(f"history {len(d):,} PA through {d['game_date'].max().date()}")
    mj = M.load()
    book = F.SkillBook(d, K_bat=mj["K_bat"], K_pit=mj["K_pit"])
    bp = G.bullpen_frame(d)
    ctx = dict(mj=mj, book=book, bpbook=F.SkillBook(bp, K_pit=dict(mj["K_pit"], hr=400, brl=300)), bshare=G.bullpen_same_share(bp),
               work=G.sp_workload(d), starts=recent_lineups(d), venues=C.venues(int(today[:4])),
               parks=D.park_table([int(today[:4])]), parks_now=D.park_table([int(today[:4])]),
               form=recent_form(d, today), pitcher_season=pitcher_season(d, int(today[:4])))
    try:
        ctx["ks_fee"] = O.kalshi_fees()
    except Exception as e:
        log("kalshi fees failed", e); ctx["ks_fee"] = {}
    try:
        ctx["espn_rosters"] = O.espn_rosters()
    except Exception as e:
        log("espn rosters failed", e); ctx["espn_rosters"] = pd.DataFrame()
    log(f"context ready ({time.time()-t0:.0f}s)")
    try:
        BK.maybe_fetch(today)
    except Exception as e:
        log("odds api step failed:", type(e).__name__)
    slates = []
    for k in range(a.days):
        day = (date.fromisoformat(today) + timedelta(days=k)).isoformat()
        s = build_slate(day, d, ctx)
        if s: slates.append(s); log(f"  {day}: {len(s['games'])} games, {len(s['hitters'])} hitters, {len(s['pitchers'])} starters")
    # log picks, grade the past
    PICKS.mkdir(parents=True, exist_ok=True)
    for s in slates:
        SC.log_picks(s, PICKS)
    card = SC.scorecard(d, PICKS, today)
    report = json.loads((DATA / "hrboard" / "backtest" / "report.json").read_text()) if (DATA / "hrboard" / "backtest" / "report.json").exists() else {}
    import os
    payload = dict(built=datetime.now(timezone.utc).isoformat(timespec="seconds"), today=today, slates=slates,
                   relay=(os.environ.get("KALSHI_RELAY") or "").strip().rstrip("/") or None,
                   books_status=BK.status(),
                   model=model_block(mj, SC.slim_report(report)), card=card, data_through=str(d["game_date"].max().date()),
                   form_keys=FORM_KEYS)
    SITE.joinpath("data").mkdir(parents=True, exist_ok=True)
    js = "window.HR=" + json.dumps(payload, separators=(",", ":"), default=_json) + ";\n"
    (SITE / "data" / "hr.js").write_text(js)
    try:
        from . import kalshi_live
        kalshi_live.write(payload)
    except Exception as e:
        log("kalshi.json skipped:", e)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
    idx = SITE / "index.html"
    if idx.exists():
        import re
        idx.write_text(re.sub(r'data/hr\.js\?v=[0-9]*', f"data/hr.js?v={stamp}", idx.read_text()))
    log(f"wrote {len(js)/1e3:.0f} kB in {time.time()-t0:.0f}s")


def _json(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return None if math.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (pd.Timestamp, datetime, date)): return o.isoformat()
    return str(o)


if __name__ == "__main__":
    main()

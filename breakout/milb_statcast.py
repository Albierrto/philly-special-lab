"""Minor-league Statcast (Triple-A parks, plus a handful of lower-level parks) per hitter, from Savant's
statcast-search-minors CSV export, aggregated into a TJStats-style batter line:

    PA, BIP, EV avg, EV90, max EV, hard-hit%, barrel%, sweet-spot%, GB%, xwOBA (est. wOBA on contact + BB/HBP/K),
    xwOBAcon, K%, BB%, whiff%, chase%, bat speed (competitive swings), swing length, fast-swing%
"""
from __future__ import annotations
import io, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import requests

from .config import DATA

_S = requests.Session(); _S.headers.update({"User-Agent": "Mozilla/5.0"})


def fetch_player(pid: int, year: int, refresh=False) -> pd.DataFrame:
    p = DATA / "milb_statcast" / f"{pid}_{year}.csv"
    if p.exists() and not refresh:
        return pd.read_csv(p, low_memory=False) if p.stat().st_size > 5 else pd.DataFrame()
    u = (f"https://baseballsavant.mlb.com/statcast-search-minors/csv?hfPT=&hfAB=&hfGT=R%7C&hfPR=&hfZ=&hfStadium=&hfBBL=&hfNewZones=&hfPull=&hfC=&hfSea={year}%7C&hfSit="
         f"&player_type=batter&hfOuts=&hfOpponent=&pitcher_throws=&batter_stands=&hfSA=&game_date_gt=&game_date_lt=&hfMo=&hfTeam=&home_road=&hfRO=&position=&hfInfield=&hfOutfield=&hfInn=&hfBBT=&hfFlag=&metric_1=&group_by=name&min_pitches=0&min_results=0&min_pas=0&sort_col=pitches&player_event_sort=api_p_release_speed&sort_order=desc"
         f"&batters_lookup%5B%5D={pid}&type=details&all=true&minors=true")
    r = _S.get(u, timeout=180)
    p.parent.mkdir(parents=True, exist_ok=True)
    txt = r.text.lstrip("﻿")
    df = pd.read_csv(io.StringIO(txt), low_memory=False) if len(txt) > 200 else pd.DataFrame()
    (df if len(df) else pd.DataFrame()).to_csv(p, index=False)
    return df


def summarize(df: pd.DataFrame) -> dict:
    if df is None or not len(df):
        return {}
    pa = df[df["events"].notna()]
    if len(pa) < 30:
        return {}
    bip = pa[pa["launch_speed"].notna() & pa["launch_angle"].notna() & ~pa["events"].isin(["strikeout", "walk", "hit_by_pitch", "strikeout_double_play", "intent_walk", "catcher_interf"])]
    ev = bip["launch_speed"]
    k = pa["events"].isin(["strikeout", "strikeout_double_play"]).sum(); bb = pa["events"].isin(["walk", "intent_walk"]).sum(); hbp = (pa["events"] == "hit_by_pitch").sum()
    xw_con = bip["estimated_woba_using_speedangle"].mean() if len(bip) else np.nan
    xwoba = ((bip["estimated_woba_using_speedangle"].sum() + 0.69 * bb + 0.72 * hbp) / len(pa)) if len(bip) else np.nan
    swings = df[df["description"].isin(["swinging_strike", "foul", "hit_into_play", "swinging_strike_blocked", "foul_tip", "foul_bunt", "missed_bunt"])]
    whiff = swings["description"].isin(["swinging_strike", "swinging_strike_blocked", "foul_tip", "missed_bunt"]).mean() if len(swings) else np.nan
    oz = df[df["zone"].notna()] if "zone" in df.columns else pd.DataFrame()
    chase = np.nan
    if len(oz):
        out_zone = oz[oz["zone"] >= 11]
        chase = out_zone["description"].isin(["swinging_strike", "foul", "hit_into_play", "swinging_strike_blocked", "foul_tip"]).mean() if len(out_zone) else np.nan
    bs = pd.to_numeric(swings.get("bat_speed"), errors="coerce").dropna() if "bat_speed" in swings.columns else pd.Series(dtype=float)
    comp = bs[bs >= bs.quantile(0.10)] if len(bs) >= 20 else bs
    sl = pd.to_numeric(swings.get("swing_length"), errors="coerce").dropna() if "swing_length" in swings.columns else pd.Series(dtype=float)
    return dict(
        sc_PA=int(len(pa)), sc_BIP=int(len(bip)), sc_ev_avg=round(ev.mean(), 1) if len(ev) else np.nan,
        sc_ev90=round(ev.quantile(0.9), 1) if len(ev) else np.nan, sc_ev_max=round(ev.max(), 1) if len(ev) else np.nan,
        sc_hard_hit=round(100 * (ev >= 95).mean(), 1) if len(ev) else np.nan,
        sc_barrel=round(100 * (bip["launch_speed_angle"] == 6).mean(), 1) if len(bip) else np.nan,
        sc_sweet_spot=round(100 * bip["launch_angle"].between(8, 32).mean(), 1) if len(bip) else np.nan,
        sc_gb=round(100 * (bip["launch_angle"] < 10).mean(), 1) if len(bip) else np.nan,
        sc_xwoba=round(xwoba, 3) if pd.notna(xwoba) else np.nan, sc_xwobacon=round(xw_con, 3) if pd.notna(xw_con) else np.nan,
        sc_k=round(100 * k / len(pa), 1), sc_bb=round(100 * bb / len(pa), 1),
        sc_whiff=round(100 * whiff, 1) if pd.notna(whiff) else np.nan, sc_chase=round(100 * chase, 1) if pd.notna(chase) else np.nan,
        sc_bat_speed=round(comp.mean(), 1) if len(comp) else np.nan, sc_fast_swing=round(100 * (comp >= 75).mean(), 1) if len(comp) else np.nan,
        sc_swing_length=round(sl.mean(), 2) if len(sl) else np.nan,
        sc_parks=", ".join(sorted(df["home_team"].dropna().unique())[:8]),
    )


def build(player_ids, years=(2025, 2026), sleep=0.4) -> pd.DataFrame:
    rows = []
    for pid in player_ids:
        for y in years:
            try:
                s = summarize(fetch_player(int(pid), y))
            except Exception as e:  # network hiccup: skip
                s = {}
            if s:
                s.update(mlbam_id=int(pid), season=y); rows.append(s)
            time.sleep(sleep)
    return pd.DataFrame(rows)

"""Known 2027 availability the models cannot see: season-ending surgery, a midseason return, a lost rotation spot.

data/overrides/availability_2027.csv, one row per player, every row sourced and dated. starts_2027 replaces a starter's
projected starts (0 = out for the season), pa_2027 a hitter's projected plate appearances. The model's own number is
kept as the 2028 baseline (proj_GS_28 / proj_PA_28), so a pitcher out for 2027 is not also written off for 2028.
Nothing here touches a player's rate. Only news that changes playing time belongs in this file, and a row should come
out when it stops being true (a player signed, a timeline changed).
"""
from __future__ import annotations
import pandas as pd
from .config import DATA

PATH = DATA / "overrides" / "availability_2027.csv"


def load() -> pd.DataFrame:
    if not PATH.exists():
        return pd.DataFrame(columns=["mlbam_id", "name", "kind", "starts_2027", "pa_2027", "note", "source", "as_of"])
    o = pd.read_csv(PATH)
    o["mlbam_id"] = pd.to_numeric(o["mlbam_id"], errors="coerce").astype("Int64")
    return o


def apply_pitchers(p: pd.DataFrame) -> pd.DataFrame:
    p = p.copy(); p["proj_GS_28"] = p["proj_GS"]; p["avail_note"] = None
    o = load(); o = o[(o["kind"] == "P") & o["starts_2027"].notna()]
    for r in o.itertuples():
        m = p["mlbam_id"] == int(r.mlbam_id)
        if not m.any(): continue
        p.loc[m, "proj_GS"] = float(r.starts_2027)
        p.loc[m, "proj_pts"] = (p.loc[m, "proj_pts_gs"] * float(r.starts_2027)).round(0)
        p.loc[m, "avail_note"] = f"{r.note} ({r.as_of})"
    if "proj_pts" in p.columns and "proj_rank" in p.columns:
        p["proj_rank"] = p["proj_pts"].rank(ascending=False).astype(int)
    return p


def apply_hitters(h: pd.DataFrame) -> pd.DataFrame:
    h = h.copy(); h["proj_PA_28"] = h["proj_PA"]; h["avail_note"] = None
    o = load(); o = o[(o["kind"] == "H") & o["pa_2027"].notna()]
    for r in o.itertuples():
        m = h["mlbam_id"] == int(r.mlbam_id)
        if not m.any(): continue
        h.loc[m, "proj_PA"] = float(r.pa_2027)
        h.loc[m, "proj_pts"] = (h.loc[m, "proj_rate"] * float(r.pa_2027)).round(0)
        h.loc[m, "avail_note"] = f"{r.note} ({r.as_of})"
    return h

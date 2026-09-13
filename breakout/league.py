"""Your Fantrax league: draft-pick value, your own picks, and keeper candidates.

Reads data/fantrax/draft_<year>.psv (overall|round|pick|team|player|pos|mlb|type) exported read-only from the
league's Draft Results page, and (optionally) data/fantrax/hitters_<year>.csv (the league Players table).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from .config import DATA
from .names import key


def load_drafts() -> pd.DataFrame:
    frames = []
    for p in sorted((DATA / "fantrax").glob("draft_*.psv")):
        d = pd.read_csv(p, sep="|")
        d["season"] = int(p.stem.split("_")[1])
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["nkey"] = df["player"].apply(key)
    df["is_pitcher"] = df["pos"].isin(["SP", "RP", "P", "SP,RP"])
    return df


def draft_value(scored: pd.DataFrame, my_team: str = "Basketball") -> pd.DataFrame:
    """Every drafted hitter: pick slot vs what he produced (league points, hitter rank, vs ADP curve)."""
    dr = load_drafts()
    if dr.empty:
        return dr
    dr = dr[~dr["is_pitcher"]].copy()
    cols = ["season", "nkey", "name", "PA", "pts", "pts_g", "final_hitter_rank", "adp", "adp_hitter_rank",
            "exp_pts", "pts_over_exp", "beat", "big_beat", "produced_like_rank"]
    s = scored[cols].drop_duplicates(["season", "nkey"])
    m = dr.merge(s, on=["season", "nkey"], how="left")
    m["mine"] = m["team"] == my_team
    # hitters drafted in this league, ranked by points within the draft season
    m["league_draft_hitter_rank"] = m.groupby("season")["overall"].rank(method="first").astype(int)
    m["league_finish_rank"] = m.groupby("season")["pts"].rank(ascending=False, method="min")
    m["slot_gain"] = m["league_draft_hitter_rank"] - m["league_finish_rank"]
    return m.sort_values(["season", "overall"])


def team_draft_report(dv: pd.DataFrame) -> pd.DataFrame:
    """Per fantasy team per season: how many hitter picks beat the ADP curve, total points over expectation."""
    g = dv.groupby(["season", "team"]).agg(hitters_drafted=("player", "size"), pts=("pts", "sum"),
                                           pts_over_exp=("pts_over_exp", "sum"), beats=("beat", "sum"),
                                           big_beats=("big_beat", "sum"))
    return g.reset_index().sort_values(["season", "pts_over_exp"], ascending=[True, False])


def keeper_candidates(cands: pd.DataFrame, my_team_abbrev: str = "BB", season: int = 2026) -> pd.DataFrame:
    """Hitters on your roster (from the league Players table) with next-year projection + breakout score."""
    p = DATA / "fantrax" / f"hitters_{season}.csv"
    if not p.exists():
        return pd.DataFrame()
    h = pd.read_csv(p)
    h["nkey"] = h["player"].apply(key)
    mine = h[h["owner"] == my_team_abbrev][["player", "pos", "mlb", "fpts", "fpg", "nkey"]]
    c = cands[["nkey", "age", "PA", "pts", "final_hitter_rank", "proj_pts", "proj_rank", "proj_pts_600pa",
               "proj_rank_600pa", "skills_score", "breakout_score", "tag"]]
    return mine.merge(c, on="nkey", how="left").sort_values("proj_pts", ascending=False)

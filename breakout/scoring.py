"""League points from counting stats. Default weights = Bort's Fantrax league (verified exact vs Fantrax)."""
from __future__ import annotations
import pandas as pd
from .config import LEAGUE


def hitting_points(df: pd.DataFrame, scoring: dict | None = None) -> pd.Series:
    w = scoring or LEAGUE["scoring_hitting"]
    pts = pd.Series(0.0, index=df.index)
    for stat, weight in w.items():
        if weight and stat in df.columns:
            pts = pts + weight * df[stat].fillna(0)
    return pts


def replacement_level(season_df: pd.DataFrame, teams: int, hitters_started: int, bench_hitters: int = 3) -> float:
    """Points of the last 'startable' hitter: teams*(starters + a few bench bats)."""
    n = teams * (hitters_started + bench_hitters)
    s = season_df["pts"].sort_values(ascending=False)
    return float(s.iloc[min(n, len(s) - 1)])

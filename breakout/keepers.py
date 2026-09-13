"""Keeper economics for a keeper league where kept players cost only the roster spot.

Keeper Surplus Value (KSV) = projected league points next year - the points you could expect from the best hitter
still on the board when the draft opens (the "draft-pool replacement": once every team keeps its hitters, the pool
starts around the (kept + 12)th best projected hitter, and by the time you pick again it's ~24 deeper).

    KSV        = proj_pts - proj_pts[ rank = n_kept + 12 ]            (worth keeping if > 0)
    KSV_2nd    = proj_pts - proj_pts[ rank = n_kept + 24 ]            (worth keeping over a 2nd-round pick)

Inferred keepers: Fantrax doesn't publish keeper lists, so a player is treated as kept in year Y if he was inside
the top-N preseason ADP hitters (N = hitters kept league-wide + a margin) and does NOT appear in that year's draft
results. Paste the real list into data/fantrax/keepers_<year>.csv (player,team) to override.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from .config import DATA
from .names import key
from .league import load_drafts


def inferred_keepers(scored: pd.DataFrame, year: int, hitters_kept_per_team=5, teams=12, adp_margin=15) -> pd.DataFrame:
    p = DATA / "fantrax" / f"keepers_{year}.csv"
    if p.exists():
        k = pd.read_csv(p); k["nkey"] = k["player"].apply(key); k["source"] = "league file"
        return k
    dr = load_drafts(); dr = dr[dr["season"] == year]
    s = scored[(scored["season"] == year)].dropna(subset=["adp_hitter_rank"]).sort_values("adp_hitter_rank")
    s = s[~s["nkey"].isin(set(dr["nkey"]))]
    n = hitters_kept_per_team * teams + adp_margin
    k = s[s["adp_hitter_rank"] <= n][["name", "nkey", "mlbam_id", "adp_hitter_rank", "pts", "final_hitter_rank"]].copy()
    k["source"] = "inferred (top ADP, not drafted)"
    return k.rename(columns={"name": "player"})


def keeper_values(proj: pd.DataFrame, hitters_kept_per_team=5, teams=12, owner_col="owner") -> pd.DataFrame:
    """proj must have proj_pts, proj_rank (next season). Adds draft-pool replacement and KSV."""
    d = proj.sort_values("proj_pts", ascending=False).reset_index(drop=True).copy()
    n_kept = hitters_kept_per_team * teams
    def at(rank):
        rank = min(max(rank, 1), len(d)); return float(d["proj_pts"].iloc[rank - 1])
    r1, r2 = at(n_kept + 12), at(n_kept + 24)
    d["pool_repl_1st"] = r1; d["pool_repl_2nd"] = r2
    d["KSV"] = (d["proj_pts"] - r1).round(0)
    d["KSV_2nd"] = (d["proj_pts"] - r2).round(0)
    d["keep_tier"] = np.select([d["KSV"] >= 120, d["KSV"] >= 40, d["KSV"] > 0], ["lock", "clear keep", "marginal"], default="let go")
    return d


def team_keeper_board(kv: pd.DataFrame, owners: pd.DataFrame, owner_col="owner", n=5) -> pd.DataFrame:
    """Each fantasy team's best n keeper candidates (by KSV) from its current roster."""
    m = kv.copy() if owner_col in kv.columns else kv.merge(owners[["nkey", owner_col]], on="nkey", how="inner")
    m = m[~m[owner_col].isin(["FA", "W (Sun)"]) & m[owner_col].notna()]
    m["team_rank"] = m.groupby(owner_col)["KSV"].rank(ascending=False, method="first")
    return m[m["team_rank"] <= n + 2].sort_values([owner_col, "team_rank"])

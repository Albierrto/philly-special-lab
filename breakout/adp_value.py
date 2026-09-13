"""How much did each hitter out-produce the price the market put on him (preseason ADP)?

For every season we fit an "expected points at this ADP slot" curve (rolling median of actual league points by
hitter ADP rank, forced monotone with isotonic regression), then measure every hitter against it:

    pts_over_exp   = league points - expected points at his ADP rank        (points)
    rank_gain      = ADP hitter rank - final hitter rank                     (ranks; + = beat)
    beat / big_beat flags using thresholds in config

Undrafted hitters (no ADP that year) are treated as drafted one slot after the last drafted hitter.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from . import config as C


def expected_curve(season_df: pd.DataFrame, value_col="pts", window=25) -> pd.Series:
    """Return a Series indexed by adp_hitter_rank (1..N) with expected value_col."""
    d = season_df.dropna(subset=["adp_hitter_rank"]).sort_values("adp_hitter_rank")
    x = d["adp_hitter_rank"].values.astype(float)
    y = d[value_col].fillna(0).values.astype(float)
    roll = pd.Series(y).rolling(window, center=True, min_periods=5).median().values
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(x, roll)
    ranks = np.arange(1, int(x.max()) + 2)
    return pd.Series(iso.predict(ranks.astype(float)), index=ranks)


def score_seasons(ps: pd.DataFrame) -> pd.DataFrame:
    out = []
    for season, d in ps.groupby("season"):
        d = d.copy()
        curve_pts = expected_curve(d, "pts")
        curve_ppg = expected_curve(d, "pts_g")
        n_drafted = int(d["adp_hitter_rank"].max())
        d["undrafted"] = d["adp_hitter_rank"].isna()
        d["adp_hitter_rank_f"] = d["adp_hitter_rank"].fillna(n_drafted + 1)
        d["exp_pts"] = d["adp_hitter_rank_f"].clip(upper=n_drafted + 1).map(curve_pts)
        d["exp_pts_g"] = d["adp_hitter_rank_f"].clip(upper=n_drafted + 1).map(curve_ppg)
        d["pts_over_exp"] = d["pts"] - d["exp_pts"]
        d["ppg_over_exp"] = d["pts_g"] - d["exp_pts_g"]
        d["rank_gain"] = d["adp_hitter_rank_f"] - d["final_hitter_rank"]
        d["beat"] = ((d["pts_over_exp"] >= C.ADP_BEAT_PTS) & (d["rank_gain"] >= C.ADP_BEAT_RANKS)
                     & (d["final_hitter_rank"] <= C.RELEVANT_RANK))
        d["big_beat"] = ((d["pts_over_exp"] >= C.BIG_BEAT_PTS) & (d["rank_gain"] >= C.BIG_BEAT_RANKS)
                         & (d["final_hitter_rank"] <= C.BIG_RELEVANT_RANK))
        # "produced like" = the ADP hitter rank whose expected points match what he actually scored
        inv = curve_pts[::-1]
        d["produced_like_rank"] = d["pts"].apply(lambda v: int(np.searchsorted(inv.values, v, side="left")) and int(inv.index[max(0, np.searchsorted(inv.values, v, side="left") - 1)]) or int(inv.index[0]))
        d["bust"] = (d["pts_over_exp"] <= -C.ADP_BEAT_PTS) & (~d["undrafted"])
        out.append(d)
    return pd.concat(out, ignore_index=True)


def multi_year(scored: pd.DataFrame, min_pa_undrafted=150, seasons=None) -> pd.DataFrame:
    """One row per player: how often (and how consecutively) he beat his draft price."""
    seasons = seasons or sorted(scored["season"].unique())
    s = scored[(~scored["undrafted"]) | (scored["PA"] >= min_pa_undrafted)].copy()
    rows = []
    for pid, g in s.groupby("mlbam_id"):
        g = g.sort_values("season")
        yrs = list(g["season"]); beats = list(g["beat"]); bigs = list(g["big_beat"])
        # current streak: consecutive beat seasons ending at the latest season the player has
        streak = 0
        for b in reversed(beats):
            if b: streak += 1
            else: break
        best = cur = 0
        for b in beats:
            cur = cur + 1 if b else 0; best = max(best, cur)
        detail = "; ".join(
            f"{int(r.season)}: ADP#{'UD' if r.undrafted else int(r.adp_hitter_rank)}->fin#{int(r.final_hitter_rank)} "
            f"({r.pts:+.0f}pts, {r.pts_over_exp:+.0f} vs exp){' *BEAT*' if r.beat else ''}"
            for r in g.itertuples())
        last = g.iloc[-1]
        rows.append(dict(
            mlbam_id=pid, name=last["name"], age_2026=last["age"] if last["season"] == max(seasons) else np.nan,
            elig=last["elig"], seasons_evaluated=len(g), seasons_list=",".join(map(str, yrs)),
            beats=int(sum(beats)), big_beats=int(sum(bigs)), busts=int(g["bust"].sum()),
            beat_rate=round(sum(beats) / len(g), 2), current_streak=streak, longest_streak=best,
            total_pts_over_exp=round(g["pts_over_exp"].sum(), 0), avg_pts_over_exp=round(g["pts_over_exp"].mean(), 0),
            last_season=int(last["season"]), last_pts=last["pts"], last_adp=last["adp"], last_final_rank=int(last["final_hitter_rank"]),
            detail=detail))
    df = pd.DataFrame(rows)
    return df.sort_values(["beats", "current_streak", "total_pts_over_exp"], ascending=False).reset_index(drop=True)

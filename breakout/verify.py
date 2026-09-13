"""Sanity checks:  python -m breakout.verify

1. Points formula vs Fantrax's own table (data/fantrax/hitters_<season>.csv) — should match except grand slams /
   cycles (not available from the public stat feeds; worth 5 / 10 pts and rare).
2. Backtest: build breakout candidates from season t data only, check how often they beat ADP in season t+1
   versus every non-star hitter (the base rate).
3. Leave-one-season-out cross-validation of the next-season points/PA model.
"""
from __future__ import annotations
import sys
import pandas as pd
from .config import DATA, CURRENT_SEASON
from .names import key
from .adp_value import score_seasons
from .breakout import breakout_candidates, cross_validate


def check_points(ps: pd.DataFrame, season=CURRENT_SEASON) -> None:
    p = DATA / "fantrax" / f"hitters_{season}.csv"
    if not p.exists():
        print("no Fantrax hitters table to check against"); return
    ft = pd.read_csv(p); ft["nkey"] = ft["player"].apply(key)
    ft["mlb_fix"] = ft["mlb"].replace({"CHW": "CWS", "WAS": "WSH"})
    m = ps[ps["season"] == season].merge(ft[["nkey", "fpts", "mlb_fix"]], on="nkey")
    # duplicates (two Max Muncys): keep the row whose team matches
    m = m[(~m["nkey"].duplicated(keep=False)) | (m["team_abbr"] == m["mlb_fix"])]
    diff = (m["pts"] - m["fpts"])
    exact = (diff == 0).mean()
    print(f"points check vs Fantrax {season}: {len(m)} hitters matched, {exact:.1%} exact, "
          f"mean diff {diff.mean():+.1f}, max |diff| {diff.abs().max():.0f} (grand slams/cycles not in public feeds)")


def backtest(scored: pd.DataFrame, k=40) -> pd.DataFrame:
    rows = []
    for y in sorted(scored["season"].unique())[1:-1]:
        hist = scored[scored["season"] <= y]
        c = breakout_candidates(hist, y)
        nxt = scored[scored["season"] == y + 1][["mlbam_id", "pts_over_exp", "beat", "big_beat"]]
        v = c.merge(nxt, on="mlbam_id", how="left", suffixes=("", "_next"))
        v["beat_next"] = v["beat_next"].fillna(False); v["big_beat_next"] = v["big_beat_next"].fillna(False)
        v["pts_over_exp_next"] = v["pts_over_exp_next"].fillna(-150)
        pool = v[(v["market_rank"] > 40) & (v["PA"] >= 200)]
        top = pool.sort_values("breakout_score", ascending=False).head(k)
        rows.append(dict(built_from=y, predicts=y + 1, pool=len(pool),
                         base_beat_rate=round(pool["beat_next"].mean(), 2), top40_beat_rate=round(top["beat_next"].mean(), 2),
                         base_big_beat=round(pool["big_beat_next"].mean(), 2), top40_big_beat=round(top["big_beat_next"].mean(), 2),
                         base_avg_pts_over_exp=round(pool["pts_over_exp_next"].mean(), 0),
                         top40_avg_pts_over_exp=round(top["pts_over_exp_next"].mean(), 0)))
    return pd.DataFrame(rows)


def main():
    ps = pd.read_parquet(DATA / "player_seasons.parquet")
    check_points(ps)
    sc = score_seasons(ps)
    print("\nbacktest of breakout candidates (top-40 by breakout_score, non-stars, 200+ PA):")
    print(backtest(sc).to_string(index=False))
    print("\nnext-season model, leave-one-season-out:")
    print(cross_validate(sc).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())

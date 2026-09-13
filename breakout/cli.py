"""Command line:  python -m breakout run [--season 2026] [--league data/fantrax/league_config.json] [--refresh]"""
from __future__ import annotations
import argparse, sys, warnings
import pandas as pd

from . import config as C
from .build import player_seasons
from .adp_value import score_seasons, multi_year
from .breakout import breakout_candidates, retro_breakouts, cross_validate
from .league import draft_value, team_draft_report, keeper_candidates
from .report import write_all


def main(argv=None):
    ap = argparse.ArgumentParser(prog="breakout")
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("run", help="build everything and write output/")
    r.add_argument("--season", type=int, default=C.CURRENT_SEASON, help="latest completed/current season")
    r.add_argument("--league", default=str(C.DATA / "fantrax" / "league_config.json"))
    r.add_argument("--my-team", default="Basketball")
    r.add_argument("--my-abbrev", default="BB")
    r.add_argument("--refresh", action="store_true", help="re-download sources instead of using the cache")
    a = ap.parse_args(argv)
    if a.cmd != "run":
        ap.print_help(); return 1
    warnings.filterwarnings("ignore")
    league = C.load_league(a.league)
    seasons = [s for s in C.SEASONS if s <= a.season]
    print(f"[1/6] building player-seasons {seasons[0]}-{seasons[-1]} for {league['name']} ...")
    if a.refresh:
        from . import sources
        for y in seasons:
            sources.savant_custom(y, refresh=True); sources.mlb_season("hitting", y, refresh=True)
            sources.mlb_season("fielding", y, refresh=True); sources.fantasypros_adp(y, refresh=True)
    ps = player_seasons(seasons, league["scoring_hitting"])
    ps.to_parquet(C.DATA / "player_seasons.parquet")
    print(f"[2/6] scoring vs ADP curves ... ({len(ps)} hitter-seasons)")
    sc = score_seasons(ps)
    multi = multi_year(sc)
    print("[3/6] cross-validating next-season model ...")
    cv = cross_validate(sc)
    print(cv.to_string(index=False))
    print(f"[4/6] breakout candidates for {a.season + 1} ...")
    cands = breakout_candidates(sc, a.season)
    retro = retro_breakouts(sc, a.season)
    print("[5/6] league draft value ...")
    dv = draft_value(sc, a.my_team)
    teams = team_draft_report(dv) if len(dv) else pd.DataFrame()
    keepers = keeper_candidates(cands, a.my_abbrev, a.season)
    print("[6/6] writing output/ ...")
    write_all(sc, multi, cands, retro, dv, teams, cv, keepers, a.season)
    print("done ->", C.OUT / "REPORT.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())

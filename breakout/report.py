"""Write the CSV outputs and a markdown summary."""
from __future__ import annotations
import pandas as pd
from .config import OUT


def _fmt(df: pd.DataFrame, cols: list[str], n: int) -> str:
    d = df[cols].head(n).copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].round(2 if d[c].abs().max() < 20 else 0)
    return d.to_markdown(index=False)


def write_all(scored, multi, cands, retro, dv, teams, cv, keepers, season: int) -> None:
    OUT.mkdir(exist_ok=True)
    scored.to_csv(OUT / "player_seasons_scored.csv", index=False)
    multi.to_csv(OUT / "adp_beaters_multi_year.csv", index=False)
    cands.to_csv(OUT / f"breakout_candidates_{season + 1}.csv", index=False)
    retro.to_csv(OUT / f"breakouts_{season}_retrospective.csv", index=False)
    if len(dv):
        dv.to_csv(OUT / "league_draft_value.csv", index=False)
        teams.to_csv(OUT / "league_team_draft_report.csv", index=False)
    cv.to_csv(OUT / "model_cross_validation.csv", index=False)
    if len(keepers):
        keepers.to_csv(OUT / "my_roster_keeper_view.csv", index=False)

    md = [f"# Breakout & ADP-value report ({season} season data)\n"]
    md.append("## Multi-year ADP beaters (active in %d)\n" % season)
    act = multi[multi["last_season"] == season]
    md.append(_fmt(act, ["name", "age_2026", "elig", "seasons_evaluated", "beats", "big_beats", "busts",
                         "current_streak", "avg_pts_over_exp", "last_pts", "last_adp", "last_final_rank"], 40))
    md.append(f"\n## {season} breakouts (career-best by 100+, top-75 finish, beat ADP curve)\n")
    md.append(_fmt(retro, ["name", "age", "elig", "PA", "pts", "prev_best_pts", "jump", "adp_hitter_rank",
                           "final_hitter_rank", "pts_over_exp"], 40))
    md.append(f"\n## {season + 1} breakout candidates — full-time bats\n")
    ft = cands[(cands["pool"] == "full-time") & (cands["market_rank"] > 40)]
    md.append(_fmt(ft, ["name", "age", "elig", "PA", "pts", "final_hitter_rank", "adp_hitter_rank", "proj_pts",
                        "proj_rank", "skills_score", "breakout_score", "tag"], 40))
    md.append(f"\n## {season + 1} breakout candidates — playing-time dependent\n")
    pt = cands[(cands["pool"] != "full-time") & (cands["market_rank"] > 40)]
    md.append(_fmt(pt, ["name", "age", "elig", "PA", "pts", "final_hitter_rank", "proj_pts_600pa",
                        "proj_rank_600pa", "skills_score", "breakout_score", "tag"], 40))
    md.append("\n## Model cross-validation (next-season points/PA)\n")
    md.append(cv.to_markdown(index=False))
    if len(keepers):
        md.append("\n## Your roster: keeper view\n")
        md.append(_fmt(keepers, ["player", "pos", "age", "PA", "pts", "final_hitter_rank", "proj_pts", "proj_rank",
                                 "breakout_score", "tag"], 30))
    (OUT / "REPORT.md").write_text("\n".join(md))

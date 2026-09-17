"""v2 pipeline:  python -m breakout.pipeline2  [--season 2026]

Builds the full analytical table (every hitter-season with league points, xLP, LPAR, league fit, roto z, IL history,
career context, prospect status), the 2027 Breakout Index + projections, keeper economics and the real-draft
analysis for the league, then writes output/v2/*.csv and the explorer dataset (output/v2/explorer_data.json).
"""
from __future__ import annotations
import argparse, json, sys, warnings
import numpy as np
import pandas as pd

from . import config as C
from .adp_value import score_seasons, expected_curve
from .breakout import cross_validate
from .breakout2 import prepare, cross_validate_bi, breakout_index, project_v2
from .formulas import PRESETS, aging_curve, expected_points, lpar, league_fit, roto_values
from .injuries import injury_features
from .keepers import inferred_keepers, keeper_values, team_keeper_board
from .league import load_drafts
from .names import key
from .prospects import prospect_features


def league_pick_curve(dv: pd.DataFrame) -> pd.Series:
    """Expected league points by in-league hitter pick number, pooled over the league's drafts."""
    d = dv.dropna(subset=["pts"]).copy()
    d["hp"] = d.groupby("season")["overall"].rank(method="first")
    d = d.sort_values("hp")
    roll = d.groupby(d["hp"].round())["pts"].median().rolling(9, center=True, min_periods=3).median()
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(roll.index.values, roll.values)
    ranks = np.arange(1, int(d["hp"].max()) + 1)
    return pd.Series(iso.predict(ranks), index=ranks)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--season", type=int, default=C.CURRENT_SEASON)
    ap.add_argument("--league", default=str(C.DATA / "fantrax" / "league_config.json"))
    ap.add_argument("--my-abbrev", default="BB"); ap.add_argument("--my-team", default="Basketball")
    a = ap.parse_args(argv); warnings.filterwarnings("ignore")
    league = C.load_league(a.league); scoring = league["scoring_hitting"]; teams = league["teams"]
    out = C.OUT / "v2"; out.mkdir(parents=True, exist_ok=True)
    seasons = [s for s in C.SEASONS if s <= a.season]

    print("[1/8] base table + ADP curves")
    ps = pd.read_parquet(C.DATA / "player_seasons.parquet")
    sc = score_seasons(ps)
    print("[2/8] injuries, prospects")
    inj = injury_features(seasons)
    pf = prospect_features(ps, a.season)
    print("[3/8] formulas: xLP, LPAR, league fit, roto, aging")
    xp = expected_points(sc, scoring)
    lp = lpar(xp, league["positions"], teams)
    lf = league_fit(lp, scoring)
    curve = aging_curve(lf)
    d = prepare(lf, pf, inj)
    d.to_parquet(C.DATA / "prepared.parquet")
    print("[4/8] models: projection CV, Breakout Index CV")
    cv_proj = cross_validate(d); cv_bi = cross_validate_bi(d)
    print(cv_bi.to_string(index=False))
    bi = breakout_index(d, a.season)
    pj = project_v2(d, a.season)
    cur = bi.merge(pj[["mlbam_id", "proj_rate_raw", "age_step", "proj_rate", "proj_PA", "durability", "proj_pts",
                       "proj_pts_600", "proj_rank", "proj_rank_600"]], on="mlbam_id", how="left")
    print("[5/8] league: drafts, keepers, KSV")
    dr = load_drafts()
    owners = pd.read_csv(C.DATA / "fantrax" / f"hitters_{a.season}.csv")[["player", "owner"]]
    owners["nkey"] = owners["player"].apply(key)
    cur = cur.merge(owners[["nkey", "owner"]], on="nkey", how="left")
    cur["owner"] = cur["owner"].fillna("FA")
    kv = keeper_values(cur.dropna(subset=["proj_pts"]), league["keepers"] if league["keepers"] <= 5 else 5, teams)
    board = team_keeper_board(kv, owners, n=5)
    keepers = {y: inferred_keepers(sc, y, 5, teams) for y in sorted(dr["season"].unique())} if len(dr) else {}
    # real drafts with keepers removed = the pool; value vs in-league pick curve
    from .league import draft_value
    dv = draft_value(sc, a.my_team)
    pcurve = league_pick_curve(dv)
    dv["hitter_pick"] = dv.groupby("season")["overall"].rank(method="first")
    dv["exp_pts_slot"] = dv["hitter_pick"].round().map(pcurve)
    dv["pts_over_slot"] = dv["pts"] - dv["exp_pts_slot"]
    # 2027 draft board: everyone not likely kept (each team keeps its top-5 KSV hitters), ranked by projection
    likely_kept = set(board[board["team_rank"] <= 5]["nkey"])
    kv["likely_kept"] = kv["nkey"].isin(likely_kept)
    pool = kv[~kv["likely_kept"]].sort_values("proj_pts", ascending=False)
    pool["pool_rank"] = np.arange(1, len(pool) + 1)
    print("[6/8] prospects for 2027")
    pros = pf[(pf["prospect_status"].isin(["top-100 prospect", "rookie"])) | (pf["pipeline_rank"].notna())].copy()
    pros = pros.merge(cur[["mlbam_id", "BI", "proj_pts_600", "proj_rank_600", "owner"]], on="mlbam_id", how="left")
    print("[7/8] write csv")
    d.to_csv(out / "hitter_seasons_full.csv", index=False)
    cur.to_csv(out / f"projections_{a.season + 1}.csv", index=False)
    kv.to_csv(out / "keeper_values.csv", index=False); board.to_csv(out / "team_keeper_board.csv", index=False)
    pool.to_csv(out / f"draft_board_{a.season + 1}_pool.csv", index=False)
    dv.to_csv(out / "league_drafts_valued.csv", index=False)
    pros.sort_values("PAS", ascending=False).to_csv(out / "prospects.csv", index=False)
    curve.to_csv(out / "aging_curve.csv", index=False)
    cv_proj.to_csv(out / "cv_projection.csv", index=False); cv_bi.to_csv(out / "cv_breakout_index.csv", index=False)
    for y, k in keepers.items():
        k.to_csv(out / f"keepers_inferred_{y}.csv", index=False)
    print("[8/8] explorer dataset")
    write_explorer(d, cur, kv, pool, dv, pros, curve, cv_proj, cv_bi, keepers, pcurve, league, a, out)
    print("done ->", out)
    return 0


EXPLORER_COLS = ["mlbam_id", "name", "season", "age", "elig", "elig_next", "team_abbr", "bats", "G", "PA", "first_game", "avail_games", "il_games_in_window", "pa_pace_162", "g_share", "days_late", "AB", "H", "1B", "2B", "3B", "HR", "R", "RBI",
                 "BB", "HBP", "SB", "CS", "K", "E", "CSA", "AOF", "pts", "pts_pa", "pts_g", "final_hitter_rank", "adp", "adp_hitter_rank",
                 "adp_nfbc", "adp_fantrax", "exp_pts", "pts_over_exp", "rank_gain", "beat", "big_beat", "bust", "produced_like_rank",
                 "xLP", "xLP_pa", "luck_pts", "x1B", "x2B", "x3B", "xHR", "xR", "xRBI", "LPAR", "repl_pts", "pct_league", "pct_std_points",
                 "pct_roto", "fit_vs_points", "fit_vs_roto", "roto_z", "xba", "batting_avg", "xslg", "slg_percent", "xwoba", "woba",
                 "xobp", "on_base_percent", "xiso", "isolated_power", "babip", "k_percent", "bb_percent", "barrel_batted_rate",
                 "hard_hit_percent", "exit_velocity_avg", "avg_best_speed", "launch_angle_avg", "sweet_spot_percent", "avg_swing_speed",
                 "fast_swing_rate", "ideal_angle_rate", "squared_up_swing", "blasts_swing", "attack_angle", "oz_swing_percent",
                 "whiff_percent", "iz_contact_percent", "z_swing_percent", "swing_take_run_value", "pull_percent", "flyballs_percent",
                 "groundballs_percent", "linedrives_percent", "sprint_speed", "il_days", "il_stints", "il_days_3yr", "il_60", "il_reasons",
                 "career_best_pa", "prior_best_pa", "jump_vs_prior", "prior_seasons_200", "prior_top90", "bo_status", "gap_to_best", "seasons_200", "career_PA", "prospect_status", "pipeline_rank", "PAS"]


def write_explorer(d, cur, kv, pool, dv, pros, curve, cv_proj, cv_bi, keepers, pcurve, league, a, out):
    def recs(df, cols, columnar=False):
        cols = [c for c in cols if c in df.columns]
        x = df[cols].copy()
        for c in x.columns:
            if x[c].dtype.kind == "f":
                mx = x[c].abs().max()
                x[c] = x[c].round(3 if mx < 5 else (1 if mx < 2000 else 0))
            if x[c].dtype == bool:
                x[c] = x[c].astype(int)
            if x[c].dtype == object:
                x[c] = x[c].astype(str).str.slice(0, 80).replace({"nan": None, "None": None})
        if columnar:
            return {"cols": list(x.columns), "rows": json.loads(x.to_json(orient="values"))}
        return json.loads(x.to_json(orient="records"))
    seasons = recs(d[d["PA"] >= 50], EXPLORER_COLS, columnar=True)
    proj_cols = ["mlbam_id", "BI", "proj_rate_raw", "age_step", "proj_rate", "proj_PA", "durability", "proj_pts", "proj_pts_600",
                 "proj_rank", "proj_rank_600", "owner", "KSV", "KSV_2nd", "keep_tier", "likely_kept"]
    kvj = kv[[c for c in proj_cols if c in kv.columns]]
    projections = recs(kvj, proj_cols, columnar=True)
    payload = dict(
        meta=dict(season=a.season, league=league["name"], teams=league["teams"], scoring=league["scoring_hitting"],
                  positions=league["positions"], presets=PRESETS, generated=pd.Timestamp.today().strftime("%Y-%m-%d"),
                  my_abbrev=a.my_abbrev, n_rows=len(seasons)),
        seasons=seasons, projections=projections,
        pool=recs(pool, ["mlbam_id", "name", "pool_rank", "proj_pts", "proj_rank", "owner", "KSV"]),
        drafts=recs(dv, ["season", "overall", "round", "pick", "team", "player", "pos", "mlb", "PA", "pts", "final_hitter_rank",
                         "adp_hitter_rank", "exp_pts", "pts_over_exp", "hitter_pick", "exp_pts_slot", "pts_over_slot", "beat", "mine"]),
        pick_curve={int(k): round(float(v), 1) for k, v in pcurve.items()},
        prospects=recs(pros.sort_values("PAS", ascending=False), ["mlbam_id", "name", "age", "pipeline_rank", "prospect_pos", "prospect_org",
                       "milb_level", "milb_PA", "milb_AVG", "milb_OBP", "milb_SLG", "milb_OPS", "milb_ISO", "milb_K", "milb_BB", "milb_HR",
                       "milb_SB", "PA", "career_PA", "pedigree", "level_age_score", "milb_bat_score", "mlb_sample", "PAS", "prospect_status",
                       "BI", "proj_pts_600", "proj_rank_600", "owner"]),
        aging=recs(curve, ["age", "rel_to_27", "yoy_delta", "n_pairs"]),
        cv_proj=recs(cv_proj, list(cv_proj.columns)), cv_bi=recs(cv_bi, list(cv_bi.columns)),
        keepers={str(y): recs(k, ["player", "adp_hitter_rank", "pts", "final_hitter_rank", "source"]) for y, k in keepers.items()},
    )
    (out / "explorer_data.json").write_text(json.dumps(payload, separators=(",", ":")))
    print("explorer_data.json", round((out / "explorer_data.json").stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    sys.exit(main())

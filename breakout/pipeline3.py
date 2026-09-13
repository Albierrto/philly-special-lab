"""v3 pipeline = v2 hitters + starting pitchers + prospect cards + real keepers.   python -m breakout.pipeline3"""
from __future__ import annotations
import argparse, json, sys, warnings
import numpy as np
import pandas as pd

from . import config as C
from . import pipeline2 as P2
from .adp_value import score_seasons
from .breakout import cross_validate
from .breakout2 import prepare, cross_validate_bi, breakout_index, project_v2
from .formulas import PRESETS, aging_curve, expected_points, lpar, league_fit
from .injuries import injury_features
from .keepers import inferred_keepers, keeper_values, team_keeper_board
from .league import load_drafts, draft_value
from .names import key
from .prospects import prospect_features, milb_table
from . import pitchers as PT


def recs(df, cols, columnar=False):
    cols = [c for c in cols if c in df.columns]
    x = df[cols].copy()
    for c in x.columns:
        if x[c].dtype.kind == "f":
            mx = x[c].abs().max(); x[c] = x[c].round(3 if mx < 5 else (1 if mx < 2000 else 0))
        if x[c].dtype == bool:
            x[c] = x[c].astype(int)
        if x[c].dtype == object:
            x[c] = x[c].astype(str).str.slice(0, 120).replace({"nan": None, "None": None})
    if columnar:
        return {"cols": list(x.columns), "rows": json.loads(x.to_json(orient="values"))}
    return json.loads(x.to_json(orient="records"))


PIT_COLS = ["mlbam_id", "name", "season", "age", "throws", "team", "G", "GS", "IP", "K", "BB", "ER", "H", "HR", "QS", "CG", "SHO", "BS", "W", "L", "SV", "HLD", "ERA", "WHIP",
            "pts", "pts_gs", "pts_ip", "sp_rank", "k_percent", "bb_percent", "k_minus_bb", "xera", "xwoba", "woba", "xba", "barrel_batted_rate", "hard_hit_percent",
            "exit_velocity_avg", "whiff_percent", "z_swing_miss_percent", "oz_swing_percent", "in_zone_percent", "edge_percent", "meatball_percent",
            "f_strike_percent", "groundballs_percent", "flyballs_percent", "fb_velo", "ff_avg_speed", "movement_uniqueness", "n_pitches_10",
            "arsenal_whiff", "arsenal_putaway", "arsenal_rv100", "best_pitch", "best_pitch_whiff", "n_plus_pitches", "arsenal_desc",
            "stuff_plus", "location_plus", "pitching_plus", "il_days", "il_stints", "il_days_3yr", "il_reasons", "career_best_gs", "gap_to_best",
            "seasons_10gs", "career_GS", "pitches_per_ip", "yoy_stuff_change", "yoy_velo_change"]


def attach_owner(df: pd.DataFrame, owners: pd.DataFrame) -> pd.DataFrame:
    """Owner by normalized name; names that appear twice in the Fantrax table (two Max Muncys) are matched on the MLB club as well."""
    dup = set(owners.loc[owners["nkey"].duplicated(keep=False), "nkey"])
    uniq = owners[~owners["nkey"].isin(dup)].drop_duplicates("nkey")
    df = df.merge(uniq[["nkey", "owner"]], on="nkey", how="left")
    if dup and "team_abbr" in df.columns:
        d2 = owners[owners["nkey"].isin(dup)].rename(columns={"mlb": "team_abbr", "owner": "owner_dup"})
        df = df.merge(d2[["nkey", "team_abbr", "owner_dup"]], on=["nkey", "team_abbr"], how="left")
        df["owner"] = df["owner"].fillna(df["owner_dup"]); df = df.drop(columns=["owner_dup"])
    df["owner"] = df["owner"].fillna("FA")
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--season", type=int, default=C.CURRENT_SEASON)
    ap.add_argument("--league", default=str(C.DATA / "fantrax" / "league_config.json"))
    ap.add_argument("--my-abbrev", default="BB"); ap.add_argument("--my-team", default="Basketball")
    a = ap.parse_args(argv); warnings.filterwarnings("ignore")
    league = C.load_league(a.league); scoring = league["scoring_hitting"]; teams = league["teams"]
    cfg = json.loads(open(a.league).read()); hk = cfg.get("keeper", {}).get("hitters", 5); pk = cfg.get("keeper", {}).get("pitchers", 2)
    out = C.OUT / "v3"; out.mkdir(parents=True, exist_ok=True)
    seasons = [s for s in C.SEASONS if s <= a.season]

    # ---------------------------------------------------------------- hitters (v2)
    print("[1/9] hitters")
    ps = pd.read_parquet(C.DATA / "player_seasons.parquet"); sc = score_seasons(ps)
    inj = injury_features(seasons); pf = prospect_features(ps, a.season)
    d = prepare(league_fit(lpar(expected_points(sc, scoring), league["positions"], teams), scoring), pf, inj)
    curve = aging_curve(d); cv_proj = cross_validate(d); cv_bi = cross_validate_bi(d)
    cur = breakout_index(d, a.season).merge(project_v2(d, a.season)[["mlbam_id", "proj_rate_raw", "age_step", "proj_rate", "proj_PA", "durability",
                                                                    "proj_pts", "proj_pts_600", "proj_rank", "proj_rank_600"]], on="mlbam_id", how="left")
    owners = pd.read_csv(C.DATA / "fantrax" / f"hitters_{a.season}.csv")[["player", "owner", "mlb"]]; owners["nkey"] = owners["player"].apply(key)
    cur = attach_owner(cur, owners)
    # real keepers file: slot column tells who is a pitcher keeper (Ohtani) -> not a hitter-slot candidate for that team
    kfile = C.DATA / "fantrax" / f"keepers_{a.season}.csv"
    real_k = pd.read_csv(kfile) if kfile.exists() else pd.DataFrame(columns=["player", "team", "slot"])
    real_k["nkey"] = real_k["player"].apply(key)
    pitcher_keeper_keys = set(real_k[real_k["slot"] == "pitcher"]["nkey"])
    kv = keeper_values(cur.dropna(subset=["proj_pts"]), hk, teams)
    kv["kept_2026_as"] = kv["nkey"].map(dict(zip(real_k["nkey"], real_k["slot"]))).fillna("")
    kv_h = kv[~kv["nkey"].isin(pitcher_keeper_keys)]
    board = team_keeper_board(kv_h, owners, n=hk)
    likely = set(board[board["team_rank"] <= hk]["nkey"]) | pitcher_keeper_keys; kv["likely_kept"] = kv["nkey"].isin(likely)
    pool = kv[~kv["likely_kept"]].sort_values("proj_pts", ascending=False); pool["pool_rank"] = np.arange(1, len(pool) + 1)
    dr = load_drafts(); keepers = {y: inferred_keepers(sc, y, hk, teams) for y in sorted(dr["season"].unique())} if len(dr) else {}
    dv = draft_value(sc, a.my_team); pcurve = P2.league_pick_curve(dv)
    dv["hitter_pick"] = dv.groupby("season")["overall"].rank(method="first"); dv["exp_pts_slot"] = dv["hitter_pick"].round().map(pcurve); dv["pts_over_slot"] = dv["pts"] - dv["exp_pts_slot"]

    # ---------------------------------------------------------------- pitchers
    print("[2/9] pitchers")
    pp = PT.pitcher_seasons(seasons)
    pp = pp.merge(inj[["mlbam_id", "season", "il_days", "il_stints", "il_days_3yr", "il_60", "il_reasons"]], on=["mlbam_id", "season"], how="left")
    for c in ("il_days", "il_stints", "il_days_3yr", "il_60"): pp[c] = pp[c].fillna(0)
    pp = PT.career_context(PT.fit_pitching_plus(PT.proxies(pp)))
    coef = pp.attrs.get("pitching_plus_coef", {})
    cv_pit = PT.cross_validate(pp)
    pbi, cv_pbi = PT.breakout_index(pp, a.season); pproj, page = PT.project(pp, a.season)
    pcur = pbi.merge(pproj[["mlbam_id", "proj_pts_gs_raw", "age_step", "proj_pts_gs", "proj_GS", "durability", "proj_pts", "proj_rank", "proj_rank_gs"]], on="mlbam_id")
    powners = pd.read_csv(C.DATA / "fantrax" / f"pitchers_{a.season}.csv")[["player", "owner"]]; powners["nkey"] = powners["player"].apply(key)
    pcur = pcur.merge(powners[["nkey", "owner"]].drop_duplicates("nkey"), on="nkey", how="left"); pcur["owner"] = pcur["owner"].fillna("FA")
    # two-way: add hitting projection to a pitcher's total for keeper purposes (match on MLBAM id, e.g. Ohtani)
    hit_proj = cur.groupby("mlbam_id")["proj_pts"].max()
    pcur["proj_pts_hitting"] = pcur["mlbam_id"].map(hit_proj).fillna(0)
    pcur["proj_pts_total"] = pcur["proj_pts"] + pcur["proj_pts_hitting"]
    psorted = pcur.sort_values("proj_pts_total", ascending=False).reset_index(drop=True)
    nk = pk * teams
    r1 = float(psorted["proj_pts_total"].iloc[min(nk + teams - 1, len(psorted) - 1)]); r2 = float(psorted["proj_pts_total"].iloc[min(nk + 2 * teams - 1, len(psorted) - 1)])
    pcur["KSV"] = (pcur["proj_pts_total"] - r1).round(0); pcur["KSV_2nd"] = (pcur["proj_pts_total"] - r2).round(0)
    pcur["keep_tier"] = np.select([pcur["KSV"] >= 80, pcur["KSV"] >= 25, pcur["KSV"] > 0], ["lock", "clear keep", "marginal"], default="let go")
    pcur["kept_2026_as"] = pcur["nkey"].map(dict(zip(real_k["nkey"], real_k["slot"]))).fillna("")
    pb = pcur[~pcur["owner"].isin(["FA", "W (Sun)", "W (Mon)"])].copy(); pb["team_rank"] = pb.groupby("owner")["KSV"].rank(ascending=False, method="first")
    plikely = set(pb[pb["team_rank"] <= pk]["nkey"]); pcur["likely_kept"] = pcur["nkey"].isin(plikely)
    ppool = pcur[~pcur["likely_kept"]].sort_values("proj_pts", ascending=False); ppool["pool_rank"] = np.arange(1, len(ppool) + 1)

    # ---------------------------------------------------------------- prospect cards
    print("[3/9] prospect cards")
    pros = pf[(pf["prospect_status"].isin(["top-100 prospect", "rookie"])) | (pf["pipeline_rank"].notna())].copy()
    pros = pros.merge(cur[["mlbam_id", "BI", "proj_pts_600", "proj_rank_600", "owner"]], on="mlbam_id", how="left")
    own_map = dict(zip(owners["nkey"], owners["owner"]))  # Fantrax table covers minors-eligible players too
    pros["owner"] = pros["owner"].fillna(pros["name"].apply(lambda n: own_map.get(key(n)))).fillna("FA")
    ids = set(pros["mlbam_id"].dropna().astype(int))
    lines = milb_table((2024, 2025, 2026)); lines = lines[lines["mlbam_id"].isin(ids)].copy()
    # within-level percentiles for each stat line (against every MiLB hitter with 100+ PA at that level-season)
    allm = milb_table((2024, 2025, 2026)); allm = allm[allm["milb_PA"] >= 100]
    for col, better_low in [("milb_OPS", False), ("milb_ISO", False), ("milb_OBP", False), ("milb_K", True), ("milb_BB", False), ("milb_SB600", False)]:
        pct = allm.groupby(["season", "level"])[col].rank(pct=True)
        allm[col + "_pct"] = (100 * (1 - pct if better_low else pct)).round(0)
    lines = lines.merge(allm[["mlbam_id", "season", "level"] + [c for c in allm.columns if c.endswith("_pct")]], on=["mlbam_id", "season", "level"], how="left")
    scp = C.DATA / "milb_statcast_summary.parquet"
    sc_sum = pd.read_parquet(scp) if scp.exists() else pd.DataFrame(columns=["mlbam_id", "season"])
    if len(sc_sum):
        for col in [c for c in sc_sum.columns if c.startswith("sc_") and sc_sum[c].dtype.kind == "f"]:
            low = col in ("sc_k", "sc_whiff", "sc_chase", "sc_gb")
            pct = sc_sum.groupby("season")[col].rank(pct=True); sc_sum[col + "_pct"] = (100 * (1 - pct if low else pct)).round(0)

    # ---------------------------------------------------------------- write
    print("[4/9] csv")
    cur.to_csv(out / f"hitter_projections_{a.season + 1}.csv", index=False); kv.to_csv(out / "hitter_keeper_values.csv", index=False)
    board.to_csv(out / "hitter_keeper_board.csv", index=False); pool.to_csv(out / f"hitter_draft_pool_{a.season + 1}.csv", index=False)
    pp.to_csv(out / "pitcher_seasons_full.csv", index=False); pcur.to_csv(out / f"pitcher_projections_{a.season + 1}.csv", index=False)
    pb.sort_values(["owner", "team_rank"]).to_csv(out / "pitcher_keeper_board.csv", index=False); ppool.to_csv(out / f"pitcher_draft_pool_{a.season + 1}.csv", index=False)
    cv_pit.to_csv(out / "cv_pitcher_projection.csv", index=False); cv_pbi.to_csv(out / "cv_pitcher_breakout_index.csv", index=False)
    pros.sort_values("PAS", ascending=False).to_csv(out / "prospects.csv", index=False); lines.to_csv(out / "prospect_milb_lines.csv", index=False)
    if len(sc_sum): sc_sum.to_csv(out / "prospect_milb_statcast.csv", index=False)
    print("[5/9] explorer json")
    payload = dict(
        meta=dict(season=a.season, league=league["name"], teams=teams, scoring=scoring, pitching_scoring=PT.PSCORE, positions=league["positions"],
                  presets=PRESETS, generated=pd.Timestamp.today().strftime("%Y-%m-%d"), my_abbrev=a.my_abbrev, my_team=a.my_team,
                  hitter_keepers=hk, pitcher_keepers=pk, real_keepers=recs(real_k, ["player", "team", "slot"]), pitching_plus_coef={k: float(v) for k, v in coef.items()}),
        seasons=recs(d[d["PA"] >= 50], P2.EXPLORER_COLS, columnar=True),
        projections=recs(kv, ["mlbam_id", "BI", "proj_rate_raw", "age_step", "proj_rate", "proj_PA", "durability", "proj_pts", "proj_pts_600", "proj_rank",
                              "proj_rank_600", "owner", "KSV", "KSV_2nd", "keep_tier", "likely_kept", "kept_2026_as"], columnar=True),
        pool=recs(pool, ["mlbam_id", "name", "pool_rank", "proj_pts", "proj_rank", "owner", "KSV"]),
        drafts=recs(dv, ["season", "overall", "round", "pick", "team", "player", "pos", "mlb", "PA", "pts", "final_hitter_rank", "adp_hitter_rank", "exp_pts",
                         "pts_over_exp", "hitter_pick", "exp_pts_slot", "pts_over_slot", "beat", "mine"]),
        pick_curve={int(k): round(float(v), 1) for k, v in pcurve.items()},
        prospects=recs(pros.sort_values("PAS", ascending=False), ["mlbam_id", "name", "age", "pipeline_rank", "prospect_pos", "prospect_org", "milb_level", "milb_PA",
                       "milb_AVG", "milb_OBP", "milb_SLG", "milb_OPS", "milb_ISO", "milb_K", "milb_BB", "milb_HR", "milb_SB", "PA", "career_PA", "pedigree",
                       "level_age_score", "milb_bat_score", "mlb_sample", "PAS", "prospect_status", "BI", "proj_pts_600", "proj_rank_600", "owner", "rookie_eligible"]),
        prospect_lines=recs(lines.sort_values(["mlbam_id", "season"]), ["mlbam_id", "season", "level", "age", "milb_PA", "milb_AVG", "milb_OBP", "milb_SLG", "milb_OPS",
                            "milb_ISO", "milb_K", "milb_BB", "milb_HR", "milb_SB", "milb_OPS_pct", "milb_ISO_pct", "milb_OBP_pct", "milb_K_pct", "milb_BB_pct", "milb_SB600_pct"]),
        prospect_statcast=recs(sc_sum, list(sc_sum.columns)) if len(sc_sum) else [],
        pitchers=recs(pp[(pp["is_sp"]) & (pp["GS"] >= 3)], PIT_COLS, columnar=True),
        pitcher_proj=recs(pcur, ["mlbam_id", "BI", "proj_pts_gs_raw", "age_step", "proj_pts_gs", "proj_GS", "durability", "proj_pts", "proj_rank", "proj_rank_gs",
                                 "owner", "proj_pts_hitting", "proj_pts_total", "KSV", "KSV_2nd", "keep_tier", "likely_kept", "kept_2026_as"], columnar=True),
        pitcher_pool=recs(ppool, ["mlbam_id", "name", "pool_rank", "proj_pts", "proj_rank", "owner", "KSV"]),
        aging=recs(curve, ["age", "rel_to_27", "yoy_delta", "n_pairs"]), pitcher_aging=recs(page, ["age", "yoy_delta"]),
        cv_proj=recs(cv_proj, list(cv_proj.columns)), cv_bi=recs(cv_bi, list(cv_bi.columns)),
        cv_pit=recs(cv_pit, list(cv_pit.columns)), cv_pbi=recs(cv_pbi, list(cv_pbi.columns)),
        keepers={str(y): recs(k, ["player", "adp_hitter_rank", "pts", "final_hitter_rank", "source"]) for y, k in keepers.items()},
    )
    (out / "explorer_data.json").write_text(json.dumps(payload, separators=(",", ":")))
    print("explorer_data.json", round((out / "explorer_data.json").stat().st_size / 1e6, 2), "MB")
    print("done ->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

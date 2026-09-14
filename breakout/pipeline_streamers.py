"""Build the week's streamer tables: python -m breakout.pipeline_streamers --start 2026-09-14 --end 2026-09-20"""
from __future__ import annotations
import argparse, json, sys, warnings
from datetime import date, timedelta
import numpy as np
import pandas as pd

from . import config as C
from . import streamers as ST
from .names import key
from . import ownership as OW


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--start"); ap.add_argument("--end"); ap.add_argument("--asof", default=date.today().isoformat())
    ap.add_argument("--min-pa", type=int, default=60); ap.add_argument("--my-abbrev", default="BB")
    a = ap.parse_args(argv); warnings.filterwarnings("ignore")
    asof = date.fromisoformat(a.asof)
    start = a.start or ((asof + timedelta(days=1)) if asof.weekday() == 6 else (asof - timedelta(days=asof.weekday()))).isoformat()  # this week's Monday; on a Sunday, next week
    end = a.end or (date.fromisoformat(start) + timedelta(days=6)).isoformat()
    out = C.OUT / "streamers"; out.mkdir(parents=True, exist_ok=True)
    print(f"window {start} .. {end} (as of {asof})")

    print("[1/7] schedule, venues, rotations")
    tm = ST.teams(); sch = ST.schedule(start, end); ven = ST.venues()
    rec = ST.recent_starters((asof).isoformat(), days=16)
    sp_table = pd.read_csv(C.OUT / "v3" / "pitcher_seasons_full.csv"); s26 = sp_table[sp_table["season"] == 2026]
    regular = set(s26[(s26["GS"] >= 5) & (s26["IP"] / s26["GS"] >= 4.0)]["mlbam_id"])
    openers = set(s26[(s26["GS"] >= 3) & (s26["IP"] / s26["GS"] < 3.8)]["mlbam_id"])
    active, status = ST.active_rosters(tm["team_id"].tolist())
    sch = ST.project_rotations(sch, rec, regular_ids=regular, active_ids=active, exclude_ids=openers)
    # opponent's starter on each row
    opp = sch[["gamePk", "team_id", "sp_id", "sp_name", "sp_source"]].rename(columns={"team_id": "opp_id", "sp_id": "opp_sp_id", "sp_name": "opp_sp_name", "sp_source": "opp_sp_source"})
    sch = sch.merge(opp, on=["gamePk", "opp_id"], how="left")
    print(f"  {sch.gamePk.nunique()} games; probables listed {int((sch.sp_source=='listed').sum())}, projected {int((sch.sp_source=='projected').sum())}, unknown {int(sch.sp_id.isna().sum())}")

    print("[2/7] park factors + weather")
    pf = ST.park_factors(2026); wx = ST.weather(sch, ven)
    print(f"  park factors {pf.venue_id.nunique()} venues; weather for {len(wx)} games")

    print("[3/7] hitters")
    hs = pd.read_csv(C.OUT / "v3" / "hitter_projections_2027.csv")
    hs = hs[hs["PA"] >= a.min_pa][["mlbam_id", "name", "bats", "team", "PA", "G", "pts_pa", "xLP_pa", "elig", "owner", "sprint_speed", "xwoba", "k_percent"]].copy()
    # players below the projection cutoff (PA<100) still matter as streamers: pull from the full seasons file
    full = pd.read_csv(C.OUT / "v2" / "hitter_seasons_full.csv"); full = full[(full["season"] == 2026) & (full["PA"] >= a.min_pa)]
    extra = full[~full["mlbam_id"].isin(hs["mlbam_id"])][["mlbam_id", "name", "bats", "team", "team_abbr", "PA", "G", "pts_pa", "xLP_pa", "elig", "sprint_speed", "xwoba", "k_percent"]].copy()
    extra = extra.drop(columns=["team_abbr"]); hs = pd.concat([hs, extra], ignore_index=True)
    # ownership always comes from this morning's Fantrax sync, never from the (older) projection table
    ow = OW.owners(); hs = OW.stamp(hs, "name", ow=ow)
    # anyone on a Fantrax roster belongs in the tables whatever his playing time: a September call-up a manager just picked
    # up (Leo Bernal, 49 PA) has to show in his own lineup even though he is far below the streamer cutoff
    rostered = {k: o for k, o in zip(ow.rows["player"].apply(key), ow.rows["owner"])}
    fullall = pd.read_csv(C.OUT / "v2" / "hitter_seasons_full.csv"); fullall = fullall[fullall["season"] == 2026]
    thin = fullall[~fullall["mlbam_id"].isin(hs["mlbam_id"]) & fullall["name"].apply(lambda n: key(n) in rostered)].copy()
    if len(thin):
        thin = thin[["mlbam_id", "name", "bats", "team", "PA", "G", "pts_pa", "xLP_pa", "elig", "sprint_speed", "xwoba", "k_percent"]]
        thin["owner"] = thin["name"].apply(lambda n: rostered.get(key(n), "FA")); hs = pd.concat([hs, thin], ignore_index=True)
        print(f"  + {len(thin)} rostered hitters below the {a.min_pa}-PA cutoff: {', '.join(thin['name'].head(8))}")
    hs = hs.merge(tm[["team_id", "name"]].rename(columns={"name": "team"}), on="team", how="left")
    # traded players carry team = "multi": resolve the current club from the API
    multi = hs[hs["team_id"].isna()]["mlbam_id"].tolist()
    if multi:
        cur = ST.current_teams(multi); hs = hs.merge(cur, on="mlbam_id", how="left", suffixes=("", "_cur"))
        hs["team_id"] = hs["team_id"].fillna(hs["team_id_cur"]); hs["team"] = np.where(hs["team_cur"].notna(), hs["team_cur"], hs["team"]); hs = hs.drop(columns=["team_id_cur", "team_cur"])
    hs = hs[hs["team_id"].notna()].copy(); hs["team_id"] = hs["team_id"].astype(int)
    d30 = (asof - timedelta(days=30)).isoformat()
    hsp = ST.player_splits(hs["mlbam_id"].tolist(), "hitting", d30, asof.isoformat())
    print(f"  {len(hs)} hitters, splits for {len(hsp)}")

    print("[4/7] pitchers")
    ppall = pd.read_csv(C.OUT / "v3" / "pitcher_seasons_full.csv"); pp = ppall[ppall["season"] == 2026].copy()
    prev = ppall[ppall["season"] == 2025][["mlbam_id", "pts_gs", "GS"]].rename(columns={"pts_gs": "pts_gs_prev", "GS": "GS_prev"})
    pp = pp.merge(prev, on="mlbam_id", how="left")
    pj = pd.read_csv(C.OUT / "v3" / "pitcher_projections_2027.csv")[["mlbam_id", "proj_pts_gs_raw"]]
    pp = pp.merge(pj, on="mlbam_id", how="left")
    sp_ids = set(sch["sp_id"].dropna().astype(int))
    # starters this week that are not in the SP table (spot starters, openers): minimal rows
    missing = sp_ids - set(pp["mlbam_id"])
    if missing:
        add = sch[sch["sp_id"].isin(missing)].drop_duplicates("sp_id")[["sp_id", "sp_name"]].rename(columns={"sp_id": "mlbam_id", "sp_name": "name"})
        pp = pd.concat([pp, add], ignore_index=True)
    pp = OW.stamp(pp, "name", ow=ow)   # Fantrax sync wins here too
    print(f"  owners as of the Fantrax sync {ow.synced} ({ow.n} rostered)")
    for c in ("pts_gs", "GS", "K", "BF", "IP", "xwoba", "k_percent", "pitching_plus", "stuff_plus", "xera", "proj_pts_gs_raw", "pts_gs_prev", "GS_prev"):
        if c not in pp.columns: pp[c] = np.nan
    psp = ST.player_splits(sorted(sp_ids), "pitching", d30, asof.isoformat())
    logs = ST.starter_logs(sorted(sp_ids), 2026); pp = pp.merge(logs, on="mlbam_id", how="left")
    logs_prev = ST.starter_logs(sorted(sp_ids), 2025).rename(columns={"n_starts": "n_starts_prev", "pts_start": "pts_start_prev"})[["mlbam_id", "n_starts_prev", "pts_start_prev"]]
    pp = pp.merge(logs_prev, on="mlbam_id", how="left")
    tsp = ST.team_splits(tm["team_id"].tolist())
    print(f"  {len(sp_ids)} starters this week ({len(missing)} not in the SP table), team splits {len(tsp)}")

    # second pass on rotations: game logs know who is really an opener (IP per start < 3.8) or a swingman (< 3 starts)
    bad = set(logs[(logs["n_starts"] < 3) | (logs["ip_start"] < 3.8)]["mlbam_id"])
    if bad:
        sch.loc[(sch["sp_source"] == "projected") & (sch["sp_id"].isin(bad)), ["sp_id", "sp_name", "sp_source"]] = [np.nan, None, None]
        sch = ST.project_rotations(sch, rec, regular_ids=regular, active_ids=active, exclude_ids=openers | bad)
        opp = sch[["gamePk", "team_id", "sp_id", "sp_name", "sp_source"]].rename(columns={"team_id": "opp_id", "sp_id": "opp_sp_id", "sp_name": "opp_sp_name", "sp_source": "opp_sp_source"})
        sch = sch.drop(columns=["opp_sp_id", "opp_sp_name", "opp_sp_source"]).merge(opp, on=["gamePk", "opp_id"], how="left")
        new_ids = set(sch["sp_id"].dropna().astype(int)) - set(pp["mlbam_id"])
        print(f"  rotation second pass removed {len(bad)} openers/swingmen; {len(new_ids)} new starters not in table")
    print("[5/7] matchups")
    hd = ST.hitter_matchups(sch, hs, hsp, pp, psp, pf, wx, ven)
    ps = ST.pitcher_starts(sch, pp, psp, tsp, pf, wx, ven)
    # reference lists (PitcherList 9/8 tiers, CBS Week 26 lists) by normalized name
    ref = C.DATA / "reference"
    pl = pd.read_csv(ref / "pitcherlist_tiers_2026-09-08.csv"); pl["nkey"] = pl["pitcher"].apply(key)
    cbs = pd.read_csv(ref / "cbs_week26_2026-09-14.csv"); cbs["nkey"] = cbs["player"].apply(key)
    ps["nkey"] = ps["name"].apply(key); ps = ps.merge(pl[["nkey", "tier", "note"]].rename(columns={"tier": "pl_tier", "note": "pl_note"}).drop_duplicates("nkey"), on="nkey", how="left")
    cbs_p = cbs[cbs["list"] != "sleeper_hitter"].groupby("nkey").apply(lambda g: "; ".join(f"{'sleeper #' if r['list']=='sleeper_pitcher' else '2-start: '}{r['tier_or_rank']}" for _, r in g.iterrows())).rename("cbs")
    ps = ps.merge(cbs_p, on="nkey", how="left").drop(columns=["nkey"])
    hd["nkey"] = hd["name"].apply(key); cbs_h = cbs[cbs["list"] == "sleeper_hitter"].set_index("nkey")["tier_or_rank"].astype(str).radd("sleeper #").rename("cbs")
    hd = hd.merge(cbs_h, on="nkey", how="left").drop(columns=["nkey"])
    hd["active"] = hd["mlbam_id"].isin(active); hd["status"] = hd["mlbam_id"].map(status).fillna("not on 40-man")
    ps["active"] = ps["mlbam_id"].isin(active); ps["status"] = ps["mlbam_id"].map(status).fillna("not on 40-man")
    print(f"  inactive hitters with games: {hd[~hd.active].mlbam_id.nunique()}; inactive projected starters: {ps[~ps.active].mlbam_id.nunique()}")
    # weekly hitter aggregate
    hw = hd[hd["active"]].groupby(["mlbam_id", "name", "team", "bats", "elig", "owner"], dropna=False).agg(games=("gamePk", "nunique"), week_pts=("exp_pts", "sum"), avg_mult=("mult", "mean"),
                                                                                             base_rate=("base_rate", "first"), pa_g=("pa_g", "first"), woba_30=("woba_30", "first"), pa_30=("pa_30", "first")).reset_index()
    hw["week_pts"] = hw["week_pts"].round(1); hw["avg_mult"] = hw["avg_mult"].round(3)
    hw = hw.sort_values("week_pts", ascending=False); hw["rank"] = np.arange(1, len(hw) + 1)

    print("[6/7] write")
    sch.to_csv(out / "schedule_week.csv", index=False); wx.to_csv(out / "weather_week.csv", index=False); pf.to_csv(out / "park_factors_2026_3yr.csv", index=False)
    hd.to_csv(out / "hitter_days.csv", index=False); hw.to_csv(out / "hitter_week.csv", index=False); ps.to_csv(out / "pitcher_starts.csv", index=False)
    two = ps[ps["two_start"]].drop_duplicates("mlbam_id").sort_values("week_pts", ascending=False)
    two.to_csv(out / "two_start_pitchers.csv", index=False)
    payload = dict(meta=dict(asof=asof.isoformat(), start=start, end=end, generated=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), lg_woba=ST.LG_WOBA, lg_k=ST.LG_K,
                             listed=int((sch.drop_duplicates("gamePk").sp_source == "listed").sum()), games=int(sch.gamePk.nunique())),
                   games=json.loads(sch.drop_duplicates(["gamePk", "team_id"]).merge(wx, on="gamePk", how="left").to_json(orient="records")),
                   hitter_days=json.loads(hd.to_json(orient="records")), hitter_week=json.loads(hw.to_json(orient="records")),
                   pitcher_starts=json.loads(ps.to_json(orient="records")), park_factors=json.loads(pf.to_json(orient="records")),
                   venues=json.loads(ven[ven["venue_id"].isin(sch["venue_id"].unique())].to_json(orient="records")))
    (out / "streamers.json").write_text(json.dumps(payload, separators=(",", ":")))
    print("[7/7] top lines")
    print("Top FA hitters for the week:"); print(hw[hw.owner.isin(["FA", "W (Sun)"])].head(15)[["rank", "name", "team", "bats", "elig", "games", "week_pts", "avg_mult", "woba_30"]].to_string(index=False))
    print("Two-start pitchers (all):"); print(two.head(25)[["name", "team", "owner", "week_pts", "base_gs", "pitching_plus", "sp_source"]].to_string(index=False))
    print("Best single FA starts:"); print(ps[ps.owner.isin(["FA", "W (Sun)"])].sort_values("exp_pts", ascending=False).head(15)[["date", "name", "team", "opp", "home", "sp_source", "opp_woba_vs_hand", "park_runs", "temp_f", "wind_out", "exp_pts"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

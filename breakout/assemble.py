"""Assemble the explorer page: template + explorer_data.json + streamers.json -> philly_special_hitter_lab.html"""
from __future__ import annotations
import json, sys
import pandas as pd
from . import config as C
from . import picks

HD_COLS = ["mlbam_id", "name", "team", "bats", "elig", "owner", "date", "gamePk", "home", "opp", "opp_sp", "opp_sp_throws", "opp_sp_source", "sp_xwoba", "sp_k", "sp_pitching_plus",
           "park_runs", "park_hr", "temp_f", "wind_mph", "wind_out", "precip_prob", "roof", "local_start", "base_rate", "pa_g", "f_sp", "f_park", "f_wx", "f_platoon", "f_form", "mult", "exp_pts", "woba_30", "pa_30", "active", "status", "cbs"]
PS_COLS = ["mlbam_id", "name", "team", "throws", "owner", "date", "gamePk", "home", "opp", "venue", "sp_source", "opp_woba_vs_hand", "opp_k_vs_hand", "park_runs", "temp_f", "wind_out", "precip_prob", "local_start",
           "base_gs", "pts_gs_2026", "GS", "pitching_plus", "stuff_plus", "xera", "k_percent", "woba_30", "gs_30", "f_opp", "f_park", "f_wx", "k_bonus", "exp_pts", "two_start", "week_pts", "active", "status", "role", "ip_gs", "n_starts", "last5_pts", "pl_tier", "pl_note", "cbs"]
G_COLS = ["gamePk", "date", "team", "home", "opp", "venue", "sp_name", "sp_source", "opp_sp_name", "opp_sp_source", "local_start", "temp_f", "wind_mph", "wind_dir", "precip_prob", "dayNight"]


def columnar(rows, cols):
    df = pd.DataFrame(rows); cols = [c for c in cols if c in df.columns]; x = df[cols].copy()
    for c in x.columns:
        if x[c].dtype.kind == "f": x[c] = x[c].round(3)
        if x[c].dtype == bool: x[c] = x[c].astype(int)
    return {"cols": cols, "rows": json.loads(x.to_json(orient="values"))}


def main(argv=None):
    tpl = (C.OUT / "v4" / "explorer_template.html") if (C.OUT / "v4" / "explorer_template.html").exists() else (C.OUT / "v3" / "explorer_template.html"); data = picks.augment(json.loads((C.OUT / "v3" / "explorer_data.json").read_text()))
    sp = C.OUT / "streamers" / "streamers.json"
    if sp.exists():
        s = json.loads(sp.read_text())
        data["streamers"] = dict(meta=s["meta"], hitter_days=columnar(s["hitter_days"], HD_COLS), pitcher_starts=columnar(s["pitcher_starts"], PS_COLS), games=columnar(s["games"], G_COLS),
                                 park_factors=s["park_factors"], venues=s.get("venues", []))
    cfg = json.loads((C.DATA / "fantrax" / "league_config.json").read_text()); data["meta"]["team_names"] = cfg.get("fantasy_team_abbrevs", {})
    tpl_txt = tpl.read_text(encoding="utf-8")
    # Bort's edition (defaults to his team) and a league edition (defaults to league view, a different title)
    data["meta"]["default_team"] = data["meta"].get("my_abbrev", "BB")
    html = tpl_txt.replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    for p in (C.OUT / "v3" / "philly_special_hitter_lab.html", C.OUT / "v2" / "philly_special_hitter_lab.html"):
        p.write_text(html, encoding="utf-8")
    data["meta"]["default_team"] = ""
    league = tpl_txt.replace("<title>Philly Special Hitter Lab</title>", "<title>Philly Special Lab</title>").replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    (C.OUT / "league").mkdir(exist_ok=True); (C.OUT / "league" / "philly_special_lab.html").write_text(league, encoding="utf-8")
    print("assembled", round(len(html) / 1e6, 2), "MB", "streamers" in data)
    return 0


if __name__ == "__main__":
    sys.exit(main())

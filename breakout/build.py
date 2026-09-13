"""Assemble one row per hitter-season: identity, counting stats, league points, Statcast skills, preseason ADP."""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import sources
from .config import SEASONS
from .names import key
from .scoring import hitting_points

OF_CODES = {"7", "8", "9", "LF", "CF", "RF", "OF"}
POS_MAP = {"2": "C", "3": "1B", "4": "2B", "5": "3B", "6": "SS", "7": "OF", "8": "OF", "9": "OF", "10": "DH", "1": "P"}

SAVANT_KEEP = [
    "player_age", "pa", "k_percent", "bb_percent", "batting_avg", "slg_percent", "on_base_percent", "isolated_power",
    "babip", "xba", "xslg", "woba", "xwoba", "xobp", "xiso", "wobacon", "xwobacon", "xbadiff", "xslgdiff", "wobadiff",
    "exit_velocity_avg", "launch_angle_avg", "sweet_spot_percent", "barrel_batted_rate", "hard_hit_percent",
    "avg_best_speed", "avg_hyper_speed", "z_swing_percent", "oz_swing_percent", "oz_contact_percent", "iz_contact_percent",
    "whiff_percent", "swing_percent", "pull_percent", "groundballs_percent", "flyballs_percent", "linedrives_percent",
    "popups_percent", "avg_swing_length", "avg_swing_speed", "blasts_swing", "squared_up_swing", "fast_swing_rate",
    "attack_angle", "ideal_angle_rate", "swing_take_run_value", "sprint_speed", "hp_to_1b", "b_hit_fly",
    "b_hit_ground", "b_hit_line_drive", "batted_ball", "meatball_swing_percent", "f_strike_percent",
]


def _mlb_hitting(year: int) -> pd.DataFrame:
    rows = []
    for s in sources.mlb_season("hitting", year):
        p, st = s["player"], s["stat"]
        pp = p.get("primaryPosition", {})
        rows.append(dict(
            mlbam_id=p["id"], name=p["fullName"], season=year, birth_date=p.get("birthDate"),
            primary_pos=POS_MAP.get(pp.get("code"), pp.get("abbreviation")), bats=(p.get("batSide") or {}).get("code"),
            team=(s.get("team") or {}).get("name", "multi") if s.get("numTeams", 1) == 1 else "multi",
            G=st.get("gamesPlayed", 0), PA=st.get("plateAppearances", 0), AB=st.get("atBats", 0), H=st.get("hits", 0),
            **{"2B": st.get("doubles", 0), "3B": st.get("triples", 0)}, HR=st.get("homeRuns", 0), R=st.get("runs", 0),
            RBI=st.get("rbi", 0), BB=st.get("baseOnBalls", 0), HBP=st.get("hitByPitch", 0), SB=st.get("stolenBases", 0),
            CS=st.get("caughtStealing", 0), K=st.get("strikeOuts", 0), SF=st.get("sacFlies", 0),
        ))
    df = pd.DataFrame(rows)
    df["1B"] = df["H"] - df["2B"] - df["3B"] - df["HR"]
    return df


def _mlb_fielding(year: int) -> pd.DataFrame:
    """Per player: total errors, catcher caught-stealing (CSA), OF assists (AOF), games by position."""
    agg: dict[int, dict] = {}
    for s in sources.mlb_season("fielding", year):
        pid = s["player"]["id"]; st = s["stat"]
        pos = s.get("position", {}).get("abbreviation") or s.get("position", {}).get("code")
        a = agg.setdefault(pid, dict(E=0, CSA=0, AOF=0, g_C=0, g_1B=0, g_2B=0, g_3B=0, g_SS=0, g_OF=0, g_DH=0))
        a["E"] += st.get("errors", 0) or 0
        g = st.get("games", 0) or 0
        if pos == "C":
            a["CSA"] += st.get("caughtStealing", 0) or 0; a["g_C"] += g
        elif pos in OF_CODES:
            a["AOF"] += st.get("assists", 0) or 0; a["g_OF"] += g
        elif pos in ("1B", "2B", "3B", "SS", "DH"):
            a[f"g_{pos}"] += g
    df = pd.DataFrame.from_dict(agg, orient="index").reset_index().rename(columns={"index": "mlbam_id"})
    return df


def _savant(year: int) -> pd.DataFrame:
    d = sources.savant_custom(year)
    df = pd.DataFrame(d)
    keep = [c for c in SAVANT_KEEP if c in df.columns]
    df = df[["player_id"] + keep].rename(columns={"player_id": "mlbam_id", "pa": "pa_savant"})
    for c in df.columns:
        if c != "mlbam_id":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["mlbam_id"] = df["mlbam_id"].astype(int)
    return df


HITTER_POS = {"C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "OF", "DH"}


def _adp(year: int, hitter_keys: set | None = None) -> pd.DataFrame:
    """FantasyPros archive rows -> hitters only, ranked among hitters. Retired/FA players have no pos on the
    page, so a row also counts as a hitter if its name matches a hitter who played that season."""
    fp = sources.fantasypros_adp(year).copy()
    fp = fp[~fp["player"].str.contains(r"\(Batter\)|\(Pitcher\)", regex=True)]
    for c in ("AVG", "NFBC", "FT", "CBS", "Yahoo", "RTS", "ESPN"):
        if c in fp.columns:
            fp[c] = pd.to_numeric(fp[c], errors="coerce")
    fp["nkey"] = fp["player"].apply(key)
    pos_hit = fp["pos"].fillna("").apply(lambda p: any(x.strip() in HITTER_POS for x in p.split(",")))
    pos_pit = fp["pos"].fillna("").apply(lambda p: p != "" and all(x.strip() in ("SP", "RP", "P") for x in p.split(",")))
    in_mlb = fp["nkey"].isin(hitter_keys or set())
    fp["is_hitter"] = (pos_hit | in_mlb) & ~pos_pit
    fp = fp[fp["is_hitter"]].copy()
    fp["adp"] = fp["AVG"]
    fp = fp.sort_values("adp").reset_index(drop=True)
    fp["adp_hitter_rank"] = np.arange(1, len(fp) + 1)
    out = fp[["nkey", "player", "team", "pos", "adp", "adp_hitter_rank"] + [c for c in ("NFBC", "FT") if c in fp.columns]]
    out = out.rename(columns={"player": "adp_name", "team": "adp_team", "pos": "adp_pos", "NFBC": "adp_nfbc", "FT": "adp_fantrax"})
    return out.drop_duplicates("nkey", keep="first")


TEAM_ABBR = {"Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC", "Chicago White Sox": "CWS", "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL", "Detroit Tigers": "DET", "Houston Astros": "HOU", "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD", "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN", "New York Mets": "NYM", "New York Yankees": "NYY", "Athletics": "ATH",
    "Oakland Athletics": "ATH", "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT", "San Diego Padres": "SD",
    "San Francisco Giants": "SF", "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL", "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR", "Washington Nationals": "WSH", "Cleveland Indians": "CLE"}
ADP_TEAM_FIX = {"OAK": "ATH", "WAS": "WSH", "CHW": "CWS"}


def _merge_adp(df: pd.DataFrame, adp: pd.DataFrame) -> pd.DataFrame:
    """Join on normalised name; when two MLB hitters share a name in a season (two Max Muncys), use team."""
    df = df.copy()
    df["team_abbr"] = df["team"].map(TEAM_ABBR).fillna(df["team"])
    dup = df["nkey"].duplicated(keep=False)
    a = adp.copy(); a["adp_team"] = a["adp_team"].replace(ADP_TEAM_FIX)
    m1 = df[~dup].merge(a, on="nkey", how="left")
    m2 = df[dup].merge(a, left_on=["nkey", "team_abbr"], right_on=["nkey", "adp_team"], how="left")
    return pd.concat([m1, m2], ignore_index=True)


def player_seasons(seasons=SEASONS, scoring: dict | None = None) -> pd.DataFrame:
    frames = []
    for y in seasons:
        h = _mlb_hitting(y)
        f = _mlb_fielding(y)
        df = h.merge(f, on="mlbam_id", how="left")
        for c in ("E", "CSA", "AOF", "g_C", "g_1B", "g_2B", "g_3B", "g_SS", "g_OF", "g_DH"):
            df[c] = df[c].fillna(0).astype(int)
        df = df.merge(_savant(y), on="mlbam_id", how="left")
        df["nkey"] = df["name"].apply(key)
        hitters = set(df.loc[(df["primary_pos"] != "P") & (df["PA"] > 0), "nkey"])
        adp = _adp(y, hitters)
        df = _merge_adp(df, adp)
        df["pts"] = hitting_points(df, scoring)
        df["age"] = ((pd.Timestamp(f"{y}-06-30") - pd.to_datetime(df["birth_date"])).dt.days / 365.25).round(1)
        frames.append(df)
    allp = pd.concat(frames, ignore_index=True)
    allp = allp[(allp["primary_pos"] != "P") | (allp["PA"] >= 100)]   # keep two-way Ohtani, drop pitchers batting
    allp["pts_g"] = allp["pts"] / allp["G"].replace(0, np.nan)
    allp["pts_pa"] = allp["pts"] / allp["PA"].replace(0, np.nan)
    allp["final_hitter_rank"] = allp.groupby("season")["pts"].rank(ascending=False, method="min").astype(int)
    allp["elig"] = allp.apply(_elig, axis=1)
    return allp


def _elig(r, thresh=20) -> str:
    e = [p for p in ("C", "1B", "2B", "3B", "SS", "OF") if r.get(f"g_{p}", 0) >= thresh]
    if not e:
        e = [r["primary_pos"]] if r["primary_pos"] in ("C", "1B", "2B", "3B", "SS", "OF") else ["UT"]
    return "/".join(e)

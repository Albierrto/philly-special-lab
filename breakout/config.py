"""League configuration. Edit LEAGUE (or pass --league path/to/league.json) to re-tune the whole pipeline."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "output"

# Default = Bort's Fantrax league "Philly Special" (12-team H2H points, keeper, SP-only pitching).
# Hitting points were verified exactly against Fantrax's own 2026 table (500/500 hitters matched).
LEAGUE = {
    "name": "Philly Special (Fantrax)",
    "teams": 12,
    "format": "h2h_points",
    "hitters_started": 10,          # C,1B,2B,3B,SS,OF,OF,OF,UT,UT
    "bench": 7,
    "keepers": 7,
    "scoring_hitting": {
        "1B": 2, "2B": 3, "3B": 4, "HR": 5, "R": 1, "RBI": 1, "BB": 1, "HBP": 1, "SB": 3,
        "CS": 0, "K": 0, "E": -1, "CSA": 2, "AOF": 2, "GS": 5, "CYC": 10,
    },
    "positions": {"C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "OF": 3, "UT": 2},
    "pos_elig_games_prev": 20,
    "pos_elig_games_curr": 10,
}

# Seasons available from every source used here.
SEASONS = [2021, 2022, 2023, 2024, 2025, 2026]
CURRENT_SEASON = 2026

# Thresholds used by the analysis (tweak freely)
MIN_PA_SKILLS = 200          # minimum PA for a season to count in skill z-scores
ADP_BEAT_PTS = 75            # points above ADP-expected to count as "beat ADP"
ADP_BEAT_RANKS = 25          # hitter-rank improvement to count as "beat ADP"
BIG_BEAT_PTS = 150
BIG_BEAT_RANKS = 60
RELEVANT_RANK = 150          # a "beat" must also end the year inside the top-150 hitters (startable in a 12-teamer)
BIG_RELEVANT_RANK = 75       # a "big beat" must end inside the top-75 (top-6 rounds worth of hitters)


def load_league(path: str | None) -> dict:
    """Merge a JSON file (e.g. data/fantrax/league_config.json) over the defaults."""
    cfg = json.loads(json.dumps(LEAGUE))
    if path:
        j = json.loads(Path(path).read_text())
        if "scoring" in j and "hitting" in j["scoring"]:
            sc = {k: v for k, v in j["scoring"]["hitting"].items() if k != "note"}
            cfg["scoring_hitting"].update(sc)
        if "num_teams" in j:
            cfg["teams"] = j["num_teams"]
        if "roster" in j and "active" in j["roster"]:
            act = j["roster"]["active"]
            cfg["positions"] = {k: v for k, v in act.items() if k != "SP" and k != "RP" and k != "P"}
            cfg["hitters_started"] = sum(cfg["positions"].values())
        if "league_name" in j:
            cfg["name"] = j["league_name"]
    return cfg

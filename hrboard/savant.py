"""Plate-appearance-level Statcast (one row per finished PA), cached a week at a time under data/hrboard/pa/.

A finished week never changes, so it is downloaded once and committed; only the week in progress is fetched again.
One Savant query per week is ~8,000 rows and takes a few seconds.
"""
from __future__ import annotations
import io, time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
import pandas as pd
import requests

from breakout.config import DATA

CACHE = DATA / "hrboard" / "pa"
_S = requests.Session(); _S.headers.update({"User-Agent": "Mozilla/5.0"})

EVENTS = ["single", "double", "triple", "home_run", "field_out", "strikeout", "strikeout_double_play", "walk", "intent_walk",
          "hit_by_pitch", "force_out", "grounded_into_double_play", "double_play", "triple_play", "fielders_choice",
          "fielders_choice_out", "field_error", "sac_fly", "sac_bunt", "sac_fly_double_play", "sac_bunt_double_play", "catcher_interf"]
HFAB = "%7C".join(e.replace("_", "%5C.%5C.") for e in EVENTS) + "%7C"

COLS = ["game_date", "game_pk", "batter", "pitcher", "events", "stand", "p_throws", "home_team", "away_team", "inning",
        "inning_topbot", "at_bat_number", "bb_type", "launch_speed", "launch_angle", "launch_speed_angle", "hc_x", "hc_y",
        "estimated_woba_using_speedangle", "woba_value", "woba_denom", "bat_score", "post_bat_score", "n_thruorder_pitcher",
        "bat_speed", "age_bat", "player_name"]

# first and last dates Savant has regular-season games for (a little wide on purpose; empty weeks are skipped)
SEASONS = {2022: ("2022-04-07", "2022-10-05"), 2023: ("2023-03-30", "2023-10-01"), 2024: ("2024-03-20", "2024-09-30"),
           2025: ("2025-03-18", "2025-09-28"), 2026: ("2026-03-25", "2026-09-27")}


def _query(season: int, gt: str, lt: str) -> pd.DataFrame:
    u = (f"https://baseballsavant.mlb.com/statcast_search/csv?all=true&hfAB={HFAB}&hfGT=R%7C&hfSea={season}%7C"
         f"&player_type=batter&game_date_gt={gt}&game_date_lt={lt}&type=details")
    for attempt in range(4):
        try:
            r = _S.get(u, timeout=240)
            if r.ok:
                txt = r.content.decode("utf-8-sig")
                if len(txt) < 50:
                    return pd.DataFrame(columns=COLS)
                d = pd.read_csv(io.StringIO(txt), low_memory=False, usecols=lambda c: c in COLS)
                if len(d) >= 24990:
                    raise RuntimeError(f"Savant row cap hit for {gt}..{lt}")
                return d
        except RuntimeError:
            raise
        except Exception:
            pass
        time.sleep(3 + 5 * attempt)
    raise RuntimeError(f"Savant query failed for {gt}..{lt}")


def weeks(season: int, end: str | None = None):
    """Whole calendar weeks from the season's first date, through the week that contains `end`. Weeks are never cut
    short at `end` (a cut-short week saved as 'finished' would miss its last days forever); load() trims by date."""
    d0, d1 = (date.fromisoformat(x) for x in SEASONS[season])
    stop = min(d1, date.fromisoformat(end)) if end else d1
    out = []
    while d0 <= stop:
        e = min(d0 + timedelta(days=6), d1); out.append((d0.isoformat(), e.isoformat())); d0 = e + timedelta(days=1)
    return out


def _finished(lt: str, today: str) -> bool:
    return lt < (date.fromisoformat(today) - timedelta(days=1)).isoformat()


def _path(season, gt, lt, today):
    """A finished week is committed; the week in progress is re-fetched every build and kept out of git (.open)."""
    return CACHE / (f"{season}_{gt}.parquet" if _finished(lt, today) else f"{season}_{gt}.open.parquet")


def load(season: int, end: str | None = None, workers: int = 3, refresh_open: bool = True) -> pd.DataFrame:
    """Every finished PA of `season` through `end` (inclusive). Weeks that are complete are read from the cache."""
    CACHE.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    todo = []
    for gt, lt in weeks(season, end):
        f = _path(season, gt, lt, today)
        if f.exists() and (_finished(lt, today) or not refresh_open):
            continue
        todo.append((gt, lt, f))
    def grab(t):
        gt, lt, f = t
        d = _query(season, gt, lt)
        d = d.reindex(columns=COLS)
        for c in ("launch_speed", "launch_angle", "hc_x", "hc_y", "estimated_woba_using_speedangle", "woba_value", "bat_speed", "age_bat"):
            d[c] = pd.to_numeric(d[c], errors="coerce").astype("float32")
        for c in ("launch_speed_angle", "woba_denom", "bat_score", "post_bat_score", "n_thruorder_pitcher", "inning", "at_bat_number"):
            d[c] = pd.to_numeric(d[c], errors="coerce").astype("float32")
        d.to_parquet(f, index=False, compression="zstd")
        if not f.name.endswith(".open.parquet"):
            f.with_name(f.name.replace(".parquet", ".open.parquet")).unlink(missing_ok=True)
        return len(d)
    if todo:
        with ThreadPoolExecutor(workers) as ex:
            n = list(ex.map(grab, todo))
        print(f"  savant {season}: fetched {len(todo)} weeks, {sum(n):,} PA")
    frames = [pd.read_parquet(f) for gt, lt in weeks(season, end) for f in [_path(season, gt, lt, today)] if f.exists()]
    frames = [x for x in frames if len(x)]
    d = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLS)
    d["game_date"] = pd.to_datetime(d["game_date"])
    if end: d = d[d["game_date"] <= pd.Timestamp(end)]
    return d.drop_duplicates(["game_pk", "at_bat_number"]).reset_index(drop=True)


if __name__ == "__main__":
    import sys
    for s in (int(x) for x in sys.argv[1:] or [2026]):
        x = load(s); print(s, len(x), x["game_date"].min(), x["game_date"].max())

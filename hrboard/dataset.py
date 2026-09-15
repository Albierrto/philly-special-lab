"""Model matrix for plate appearances: skills (as-of), park, weather, platoon, home. Shared by backtest and the live build."""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import features as F
from . import context as C

PARK_COLS = ["pf_hr", "pf_1b", "pf_2b", "pf_3b", "pf_so", "pf_bb"]


def load_pa(seasons) -> pd.DataFrame:
    from . import savant
    parts = [savant.load(s, refresh_open=False) for s in seasons]
    return F.prepare(pd.concat(parts, ignore_index=True))


def game_context(seasons) -> pd.DataFrame:
    """game_pk -> venue, weather, park factors (rolling 3 years ending the season BEFORE, so nothing leaks)."""
    sch = pd.concat([C.season_schedule(s) for s in seasons], ignore_index=True)
    sch["season"] = pd.to_datetime(sch["date"]).dt.year
    return sch


def park_table(seasons, lag: int = 1) -> pd.DataFrame:
    rows = []
    for s in seasons:
        p = C.park_factors(s - lag).copy()
        cur = C.park_factors(s)
        # a venue with no history the year before (a club's temporary home) falls back to its own current-year factor
        miss = cur[~cur.set_index(["venue_id", "stand"]).index.isin(p.set_index(["venue_id", "stand"]).index)]
        p = pd.concat([p, miss], ignore_index=True)
        p["season"] = s
        rows.append(p)
    return pd.concat(rows, ignore_index=True)


def attach_context(rows: pd.DataFrame, sch: pd.DataFrame, parks: pd.DataFrame) -> pd.DataFrame:
    """rows need game_pk, season, stand. Adds venue_id, weather and park factor columns."""
    ctx = sch[["game_pk", "venue_id", "temp_f", "wind_out", "indoor", "day_night"]]
    out = rows.merge(ctx, on="game_pk", how="left")
    out = out.merge(parks[["venue_id", "stand", "season"] + PARK_COLS], on=["venue_id", "stand", "season"], how="left")
    for c in PARK_COLS:
        out[c] = out[c].fillna(1.0)
    return out


def offsets(df: pd.DataFrame, order=("out", "k", "bb", "s1", "xb", "hr")) -> np.ndarray:
    """log of the league rate of each outcome that morning: the model's baseline, so a change in the league's home run
    rate from one season to the next moves every forecast without a refit."""
    L = np.c_[[df[f"lg_{c}"].to_numpy() for c in F.RATE_STATS]].T
    out = 1 - L.sum(axis=1)
    full = dict(zip(F.RATE_STATS, L.T)); full["out"] = out
    return np.log(np.c_[[full[c] for c in order]].T)


def design(df: pd.DataFrame, lg=None) -> pd.DataFrame:
    """The regression inputs. Rate features are log-odds relative to the league, so a league-average batter against a
    league-average pitcher in a neutral park is all zeros. That is the generalized log5 (odds-ratio) form."""
    L = df[[f"lg_{c}" for c in F.LG_COLS]].rename(columns=lambda c: c[3:])
    X = pd.DataFrame(index=df.index)
    for c in F.RATE_STATS:
        lgc = F.lg_logit(L[c].to_numpy())
        X[f"b_{c}"] = F.lg_logit(df[f"bh_{c}"].to_numpy()) - lgc          # batter vs this hand (shrunk to his own line)
        X[f"p_{c}"] = F.lg_logit(df[f"ps_{c}"].to_numpy()) - lgc          # pitcher vs this side
    X["b_brl"] = F.lg_logit(df["b_brl"].to_numpy()) - F.lg_logit(L["brl"].to_numpy())
    X["b_pull_air"] = F.lg_logit(df["b_pull_air"].to_numpy()) - F.lg_logit(L["pull_air"].to_numpy())
    X["b_hard"] = F.lg_logit(df["b_hard"].to_numpy()) - F.lg_logit(L["hard"].to_numpy())
    X["b_bat_speed"] = (df["b_bat_speed"].to_numpy() - L["bat_speed"].to_numpy()) / 3.0
    X["p_brl"] = F.lg_logit(df["p_brl"].to_numpy()) - F.lg_logit(L["brl"].to_numpy())
    X["p_gb"] = F.lg_logit(df["p_gb"].to_numpy()) - F.lg_logit(L["gb"].to_numpy())
    X["platoon"] = (df["stand"].to_numpy() == df["p_throws"].to_numpy()).astype(float)   # same hand = batter disadvantage
    X["home"] = df["is_home"].astype(float).to_numpy()
    for c in PARK_COLS:
        X[f"log_{c}"] = np.log(df[c].astype(float).to_numpy())
    indoor = df["indoor"].fillna(False).astype(bool).to_numpy()
    temp = df["temp_f"].astype(float).to_numpy()
    X["temp"] = np.where(indoor | np.isnan(temp), 0.0, (np.nan_to_num(temp, nan=72.0) - 72.0) / 10.0)
    X["wind_out"] = np.where(indoor, 0.0, np.nan_to_num(df["wind_out"].astype(float).to_numpy(), nan=0.0)) / 10.0
    X["thin_b"] = 1.0 / np.sqrt(1.0 + df["b_pa_eff"].to_numpy() / 100.0)       # how thin the batter's record is
    return X.astype("float32")


def skill_rows(book: F.SkillBook, q: pd.DataFrame) -> pd.DataFrame:
    """q: batter, pitcher, season, game_date, stand, p_throws -> all skill columns."""
    b = book.batter_features(q[["batter", "season", "game_date", "p_throws"]])
    p = book.pitcher_features(q[["pitcher", "season", "game_date", "stand"]])
    lg = F.attach_league(q, book.lga).add_prefix("lg_")
    return pd.concat([q, b, p, lg], axis=1)

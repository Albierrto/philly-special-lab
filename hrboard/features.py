"""Leak-free skill estimates for every batter and pitcher, as of the morning of any date.

Everything is a count. For a batter on a given day:
    this season before today  +  0.8 x last season  +  0.5 x the season before   (Marcel-style weights)
and the rate is that blend shrunk toward the league with K phantom plate appearances of league-average results.
The K for each stat was fitted on 2024-25 plate appearances (see backtest.py); the fitted values live in model.json.

Platoon: a batter's line against this pitcher's hand is shrunk toward HIS OWN overall line (not the league's), with a
much larger K, because most of a split is noise. Same for pitchers against a batter side.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

CLASSES = ["out", "k", "bb", "s1", "xb", "hr"]          # plate-appearance outcomes the model predicts
RATE_STATS = ["k", "bb", "s1", "xb", "hr"]
B_EXTRA = ["brl", "pull_air", "hard"]                    # per PA
P_EXTRA = ["brl", "gb"]                                  # brl per PA; gb per ball in play (bbe)
W_PREV = (0.8, 0.5)

# phantom PA used to shrink each rate toward league average (defaults; backtest.py fits and overwrites these)
K_BAT = dict(k=60, bb=120, s1=500, xb=500, hr=200, brl=120, pull_air=150, hard=100, bat_speed=40)
K_PIT = dict(k=90, bb=200, s1=800, xb=900, hr=900, brl=500, gb=150)
K_SPLIT_BAT = 600
K_SPLIT_PIT = 600

OUT_EVENTS = {"field_out", "force_out", "grounded_into_double_play", "double_play", "triple_play", "fielders_choice",
              "fielders_choice_out", "field_error", "sac_fly", "sac_bunt", "sac_fly_double_play", "sac_bunt_double_play"}


def prepare(pa: pd.DataFrame) -> pd.DataFrame:
    """Per-PA outcome class and batted-ball flags."""
    d = pa[pa["events"].notna() & (pa["events"] != "catcher_interf") & (pa["events"] != "truncated_pa")].copy()
    ev = d["events"]
    d["cls"] = np.select([ev.isin(["strikeout", "strikeout_double_play"]), ev.isin(["walk", "intent_walk", "hit_by_pitch"]),
                          ev.eq("single"), ev.isin(["double", "triple"]), ev.eq("home_run")],
                         ["k", "bb", "s1", "xb", "hr"], "out")
    d["season"] = d["game_date"].dt.year.astype(int)
    d["pa"] = 1
    for c in RATE_STATS:
        d[c] = (d["cls"] == c).astype("int8")
    d["bbe"] = d["bb_type"].notna().astype("int8")
    d["brl"] = (d["launch_speed_angle"] == 6).astype("int8")
    d["hard"] = ((d["launch_speed"] >= 95) & d["bbe"].astype(bool)).astype("int8")
    d["gb"] = (d["bb_type"] == "ground_ball").astype("int8")
    # spray angle: negative = left field. Pulled = left field for a righty, right field for a lefty.
    ang = np.degrees(np.arctan2(d["hc_x"] - 125.42, 198.27 - d["hc_y"]))
    pulled = np.where(d["stand"] == "R", ang < -15, ang > 15)
    d["pull_air"] = (pulled & d["bb_type"].isin(["fly_ball", "line_drive"])).astype("int8")
    d["bs_n"] = d["bat_speed"].notna().astype("int8")
    d["bs_sum"] = d["bat_speed"].fillna(0).astype("float32")
    d["top"] = d["inning_topbot"].eq("Top")
    d["bat_team"] = np.where(d["top"], d["away_team"], d["home_team"])
    d["fld_team"] = np.where(d["top"], d["home_team"], d["away_team"])
    d["is_home"] = (~d["top"]).astype("int8")
    # starting pitcher = first pitcher the batting team saw that game
    first = d.sort_values("at_bat_number").groupby(["game_pk", "bat_team"])["pitcher"].first().rename("opp_sp")
    d = d.merge(first, left_on=["game_pk", "bat_team"], right_index=True, how="left")
    d["vs_sp"] = (d["pitcher"] == d["opp_sp"]).astype("int8")
    return d


B_COUNTS = ["pa"] + RATE_STATS + ["bbe"] + B_EXTRA + ["bs_n", "bs_sum"]
P_COUNTS = ["pa"] + RATE_STATS + ["bbe"] + P_EXTRA


def _asof(d: pd.DataFrame, who: str, cols: list[str], split: str | None = None) -> pd.DataFrame:
    """This-season counts THROUGH each date, one row per (who, [split], season, game_date) that appears in d.
    Attach with a strictly-before as-of join so a game never sees its own plate appearances."""
    keys = [who] + ([split] if split else []) + ["season"]
    day = d.groupby(keys + ["game_date"], observed=True)[cols].sum().reset_index().sort_values(keys + ["game_date"])
    day[cols] = day.groupby(keys, observed=True)[cols].cumsum().astype("float32")
    return day


def _season_totals(d: pd.DataFrame, who: str, cols: list[str], split: str | None = None) -> pd.DataFrame:
    keys = [who] + ([split] if split else []) + ["season"]
    return d.groupby(keys, observed=True)[cols].sum().reset_index()


def _blend(cur: pd.DataFrame, tot: pd.DataFrame, who: str, cols: list[str], split: str | None = None) -> pd.DataFrame:
    """cur (as-of this season) + W_PREV-weighted prior season totals."""
    keys = [who] + ([split] if split else [])
    out = cur.copy()
    for lag, w in zip((1, 2), W_PREV):
        t = tot.copy(); t["season"] = t["season"] + lag
        m = out[keys + ["season"]].merge(t, on=keys + ["season"], how="left")[cols].fillna(0).to_numpy(dtype="float32")
        out[cols] = out[cols].to_numpy(dtype="float32") + w * m
    return out


LG_COLS = RATE_STATS + ["brl", "pull_air", "hard", "gb", "bat_speed"]
ALL_COUNTS = sorted(set(B_COUNTS + P_COUNTS))


def _lg_from_counts(g: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=g.index)
    for c in RATE_STATS + ["brl", "pull_air", "hard"]:
        out[c] = g[c] / g["pa"]
    out["gb"] = g["gb"] / g["bbe"]
    out["bat_speed"] = (g["bs_sum"] / g["bs_n"].replace(0, np.nan))
    return out


def league_rates(d: pd.DataFrame) -> pd.DataFrame:
    """League rate for each stat by season (per PA; gb per bbe; bat speed mean)."""
    out = _lg_from_counts(d.groupby("season")[ALL_COUNTS].sum())
    out["bat_speed"] = out["bat_speed"].fillna(out["bat_speed"].mean())
    return out


LG_PRIOR_PA = 15000   # early in a season the league line leans on last season's; by mid-April it is this season's


def league_asof(d: pd.DataFrame) -> pd.DataFrame:
    """League rates through each date of each season (inclusive), blended with the prior season's full line.
    Join strictly-before, like the player tables, so a day's forecast never uses that day's results."""
    full = league_rates(d)
    day = d.groupby(["season", "game_date"])[ALL_COUNTS].sum().sort_index()
    cum = day.groupby(level="season").cumsum()
    rows = []
    for (season, gd), r in cum.iterrows():
        prior = full.loc[season - 1] if (season - 1) in full.index else full.loc[season]
        g = r.copy()
        for c in RATE_STATS + ["brl", "pull_air", "hard"]:
            g[c] = r[c] + LG_PRIOR_PA * prior[c]
        g["pa"] = r["pa"] + LG_PRIOR_PA
        g["gb"] = r["gb"] + 0.3 * LG_PRIOR_PA * prior["gb"]; g["bbe"] = r["bbe"] + 0.3 * LG_PRIOR_PA
        g["bs_sum"] = r["bs_sum"] + 3000 * prior["bat_speed"]; g["bs_n"] = r["bs_n"] + 3000
        rows.append(dict(season=season, game_date=gd, **_lg_from_counts(g.to_frame().T).iloc[0].to_dict()))
    out = pd.DataFrame(rows)
    # the morning of opening day: last season's line
    opening = []
    for season in sorted(d["season"].unique()):
        prior = full.loc[season - 1] if (season - 1) in full.index else full.loc[season]
        opening.append(dict(season=season, game_date=pd.Timestamp(f"{season}-01-01"), **prior.to_dict()))
    out = pd.concat([pd.DataFrame(opening), out], ignore_index=True).sort_values(["season", "game_date"])
    return out.reset_index(drop=True)


def attach_league(q: pd.DataFrame, lga: pd.DataFrame) -> pd.DataFrame:
    """League rates as of the morning of each row's game_date (rows need season, game_date). Index follows q."""
    qq = q[["season", "game_date"]].copy(); qq["_i"] = np.arange(len(qq))
    m = pd.merge_asof(qq.sort_values("game_date"), lga.rename(columns={"game_date": "_d"}).sort_values("_d"),
                      left_on="game_date", right_on="_d", by="season", allow_exact_matches=False, direction="backward")
    m = m.sort_values("_i")
    out = m[LG_COLS].copy(); out.index = q.index
    return out


def _rates(b: pd.DataFrame, L: pd.DataFrame, K: dict, extra: list[str], prefix: str) -> pd.DataFrame:
    out = pd.DataFrame(index=b.index)
    for c in RATE_STATS + [e for e in extra if e != "gb"]:
        out[f"{prefix}{c}"] = (b[c].to_numpy() + K[c] * L[c].to_numpy()) / (b["pa"].to_numpy() + K[c])
    if "gb" in extra:
        out[f"{prefix}gb"] = (b["gb"].to_numpy() + K["gb"] * L["gb"].to_numpy()) / (b["bbe"].to_numpy() + K["gb"])
    if "bs_n" in b:
        k = K.get("bat_speed", 40)
        out[f"{prefix}bat_speed"] = (b["bs_sum"].to_numpy() + k * L["bat_speed"].to_numpy()) / (b["bs_n"].to_numpy() + k)
    out[f"{prefix}pa_eff"] = b["pa"].to_numpy()
    return out


def _split_rates(s: pd.DataFrame, overall: pd.DataFrame, k: float, prefix: str, oprefix: str) -> pd.DataFrame:
    out = pd.DataFrame(index=s.index)
    for c in ["hr", "k", "bb", "s1", "xb"]:
        base = overall[f"{oprefix}{c}"].to_numpy()
        out[f"{prefix}{c}"] = (s[c].to_numpy() + k * base) / (s["pa"].to_numpy() + k)
    return out


class SkillBook:
    """Holds the as-of tables for one dataset so features can be attached to any (id, date) quickly."""

    def __init__(self, d: pd.DataFrame, K_bat=None, K_pit=None, k_split_bat=None, k_split_pit=None):
        self.d = d
        self.K_bat = dict(K_BAT, **(K_bat or {})); self.K_pit = dict(K_PIT, **(K_pit or {}))
        self.ksb = k_split_bat or K_SPLIT_BAT; self.ksp = k_split_pit or K_SPLIT_PIT
        self.lg = league_rates(d)
        self.lga = league_asof(d)
        self.b_cur = _asof(d, "batter", B_COUNTS); self.b_tot = _season_totals(d, "batter", B_COUNTS)
        self.p_cur = _asof(d, "pitcher", P_COUNTS); self.p_tot = _season_totals(d, "pitcher", P_COUNTS)
        self.bh_cur = _asof(d, "batter", B_COUNTS, "p_throws"); self.bh_tot = _season_totals(d, "batter", B_COUNTS, "p_throws")
        self.ps_cur = _asof(d, "pitcher", P_COUNTS, "stand"); self.ps_tot = _season_totals(d, "pitcher", P_COUNTS, "stand")

    def batter_features(self, q: pd.DataFrame) -> pd.DataFrame:
        """q: columns batter, season, game_date, p_throws. Returns rates overall and vs that hand."""
        b = self._attach(q, self.b_cur, self.b_tot, "batter", B_COUNTS)
        o = _rates(b, attach_league(q, self.lga), self.K_bat, B_EXTRA, "b_")
        bh = self._attach(q, self.bh_cur, self.bh_tot, "batter", B_COUNTS, "p_throws")
        s = _split_rates(bh, o, self.ksb, "bh_", "b_")
        return pd.concat([o, s], axis=1)

    def pitcher_features(self, q: pd.DataFrame) -> pd.DataFrame:
        """q: columns pitcher, season, game_date, stand."""
        p = self._attach(q, self.p_cur, self.p_tot, "pitcher", P_COUNTS)
        o = _rates(p, attach_league(q, self.lga), self.K_pit, P_EXTRA, "p_")
        ps = self._attach(q, self.ps_cur, self.ps_tot, "pitcher", P_COUNTS, "stand")
        s = _split_rates(ps, o, self.ksp, "ps_", "p_")
        return pd.concat([o, s], axis=1)

    def _attach(self, q, cur, tot, who, cols, split=None):
        """As-of counts for each row of q (this season before q.game_date, plus weighted prior seasons)."""
        keys = [who] + ([split] if split else []) + ["season"]
        qq = q[keys + ["game_date"]].copy()
        qq["_i"] = np.arange(len(qq))
        c = cur.rename(columns={"game_date": "_d"}).sort_values("_d")
        qq = qq.sort_values("game_date")
        m = pd.merge_asof(qq, c, left_on="game_date", right_on="_d", by=keys, allow_exact_matches=False, direction="backward")
        m = m.sort_values("_i").reset_index(drop=True)
        m[cols] = m[cols].fillna(0).astype("float32")
        m = _blend(m[keys + cols], tot, who, cols, split)
        m.index = q.index
        return m


def lg_logit(p):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))

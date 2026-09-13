"""The formula set. Everything here is league-aware: pass a scoring dict and it re-derives.

aging_curve(ps)             empirical delta-method aging curve of points/PA (what age does to a hitter's rate)
expected_points(ps, w)      xLP: league points per PA a hitter *deserved* from Statcast expected stats
lpar(ps, league)            League Points Above Replacement by position, for a 12-team (or N-team) roster
league_fit(ps, w, presets)  how much this league's scoring likes a hitter vs standard points / 5x5 roto
roto_values(ps)             classic z-score 5x5 roto value (R, HR, RBI, SB, AVG) for the roto branch
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .scoring import hitting_points

# ---------------------------------------------------------------------------------------------- scoring presets
PRESETS = {
    "philly_special": {"1B": 2, "2B": 3, "3B": 4, "HR": 5, "R": 1, "RBI": 1, "BB": 1, "HBP": 1, "SB": 3, "CS": 0, "K": 0,
                       "E": -1, "CSA": 2, "AOF": 2},
    "espn_points":    {"1B": 1, "2B": 2, "3B": 3, "HR": 4, "R": 1, "RBI": 1, "BB": 1, "HBP": 0, "SB": 1, "CS": 0, "K": -1},
    "yahoo_points":   {"1B": 2.6, "2B": 5.2, "3B": 7.8, "HR": 10.4, "R": 1.9, "RBI": 1.9, "BB": 2.6, "HBP": 2.6, "SB": 4.2, "CS": 0, "K": 0},
    "cbs_points":     {"1B": 1, "2B": 2, "3B": 3, "HR": 4, "R": 1, "RBI": 1, "BB": 1, "HBP": 1, "SB": 2, "CS": -1, "K": -0.5},
    "fantrax_default":{"1B": 1, "2B": 2, "3B": 3, "HR": 4, "R": 1, "RBI": 1, "BB": 1, "HBP": 1, "SB": 2, "CS": -1, "K": -0.5},
}


# ---------------------------------------------------------------------------------------------- aging curve
def aging_curve(ps: pd.DataFrame, min_pa=300, value="pts_pa") -> pd.DataFrame:
    """Delta method: mean change in points/PA from age a to a+1 for hitters with >= min_pa both years,
    PA-harmonic-weighted, then cumulated into a curve relative to age 27. Smoothed with a 3-age window."""
    d = ps[ps["PA"] >= min_pa][["mlbam_id", "season", "age", "PA", value]].copy()
    d["age_i"] = d["age"].round().astype(int)
    nxt = d.copy(); nxt["season"] -= 1
    m = d.merge(nxt, on=["mlbam_id", "season"], suffixes=("", "_n"))
    m["delta"] = m[f"{value}_n"] - m[value]
    m["w"] = 2 / (1 / m["PA"] + 1 / m["PA_n"])
    g = m.groupby("age_i").apply(lambda x: pd.Series({"delta": np.average(x["delta"], weights=x["w"]), "n": len(x)}))
    g = g[(g.index >= 21) & (g.index <= 38)]
    g["delta_s"] = g["delta"].rolling(3, center=True, min_periods=1).mean()
    # cumulate: curve(a) = sum of deltas from 27 up to a (or down)
    ages = list(range(21, 40)); curve = {}
    for a in ages:
        if a == 27: curve[a] = 0.0
    for a in range(28, 40):
        curve[a] = curve[a - 1] + float(g["delta_s"].get(a - 1, g["delta_s"].iloc[-1]))
    for a in range(26, 20, -1):
        curve[a] = curve[a + 1] - float(g["delta_s"].get(a, g["delta_s"].iloc[0]))
    out = pd.DataFrame({"age": ages, "rel_to_27": [curve[a] for a in ages]})
    out["yoy_delta"] = out["age"].map(lambda a: float(g["delta_s"].get(a, np.nan)))
    out["n_pairs"] = out["age"].map(lambda a: int(g["n"].get(a, 0)))
    return out


def age_adjustment(curve: pd.DataFrame, age: pd.Series) -> pd.Series:
    """Expected change in points/PA from this age to next age (one step along the curve)."""
    a = age.round().clip(21, 38).astype(int)
    lookup = curve.set_index("age")["yoy_delta"].ffill().bfill()
    return a.map(lookup).fillna(0)


# ---------------------------------------------------------------------------------------------- xLP
def expected_points(ps: pd.DataFrame, scoring: dict, min_pa=100) -> pd.DataFrame:
    """xLP: league points per PA a hitter deserved, built from Statcast expected stats.

    x1B, xXBH from xBA / xISO with the hitter's own XBH mix (regressed to league mix over 40 XBH);
    BB, HBP, SB are skills -> actual rates; R and RBI are modelled from xOBP/xISO/speed with a ridge fit
    (they depend on lineup context, so we take the league-average context for that profile).
    """
    d = ps.copy()
    ok = (d["PA"] >= min_pa) & d["xba"].notna()
    pa = d["PA"].replace(0, np.nan); ab = d["AB"].replace(0, np.nan)
    xbh = (d["2B"] + d["3B"] + d["HR"])
    lg_mix = (d.loc[ok, ["2B", "3B", "HR"]].sum() / xbh[ok].sum())
    k = 40  # regression weight in XBH
    r2 = (d["2B"] + k * lg_mix["2B"]) / (xbh + k); r3 = (d["3B"] + k * lg_mix["3B"]) / (xbh + k); rhr = (d["HR"] + k * lg_mix["HR"]) / (xbh + k)
    xH = d["xba"] * ab
    xXBH = (d["xiso"] * ab) / (r2 + 2 * r3 + 3 * rhr)
    xXBH = np.minimum(xXBH, xH * 0.9)
    d["x1B"] = (xH - xXBH); d["x2B"] = xXBH * r2; d["x3B"] = xXBH * r3; d["xHR"] = xXBH * rhr
    # R and RBI context model (fit on qualified seasons): per-PA rates vs expected profile
    feats = pd.DataFrame({"xobp": d["xobp"], "xiso": d["xiso"], "xba": d["xba"], "sb": d["SB"] / pa, "hr": d["xHR"] / pa,
                          "sprint": d["sprint_speed"].fillna(d["sprint_speed"].median())})
    fit = ok & (d["PA"] >= 300)
    rm = Ridge(alpha=1.0).fit(feats[fit], (d.loc[fit, "R"] / pa[fit]))
    rbim = Ridge(alpha=1.0).fit(feats[fit], (d.loc[fit, "RBI"] / pa[fit]))
    d["xR"] = rm.predict(feats.fillna(feats.median())) * pa
    d["xRBI"] = rbim.predict(feats.fillna(feats.median())) * pa
    w = scoring
    d["xLP"] = (w.get("1B", 0) * d["x1B"] + w.get("2B", 0) * d["x2B"] + w.get("3B", 0) * d["x3B"] + w.get("HR", 0) * d["xHR"]
                + w.get("R", 0) * d["xR"] + w.get("RBI", 0) * d["xRBI"] + w.get("BB", 0) * d["BB"] + w.get("HBP", 0) * d["HBP"]
                + w.get("SB", 0) * d["SB"] + w.get("CS", 0) * d["CS"] + w.get("K", 0) * d["K"]
                + w.get("E", 0) * d["E"] + w.get("CSA", 0) * d["CSA"] + w.get("AOF", 0) * d["AOF"])
    d.loc[~ok, "xLP"] = np.nan
    d["xLP_pa"] = d["xLP"] / pa
    d["luck_pts"] = d["pts"] - d["xLP"]           # + = out-produced the contact quality (regression risk)
    d["luck_pa"] = d["pts_pa"] - d["xLP_pa"]
    return d


# ---------------------------------------------------------------------------------------------- LPAR
def lpar(ps: pd.DataFrame, positions: dict, teams: int, bench_hitters: int = 2, value="pts") -> pd.DataFrame:
    """League Points Above Replacement. Replacement at a position = the value of the best hitter left once every
    team has filled that slot (plus a couple of bench bats league-wide). UT slots absorb the best leftovers."""
    d = ps.copy()
    out = []
    for season, s in d.groupby("season"):
        s = s.copy().sort_values(value, ascending=False)
        elig = s["elig"].fillna("UT").str.split("/")
        taken = set(); repl = {}
        for pos in ["C", "SS", "2B", "3B", "1B", "OF"]:
            n = teams * positions.get(pos, 0)
            pool = s[np.array([pos in e for e in elig]) & ~s["mlbam_id"].isin(taken).values]
            take = pool.head(n)
            taken |= set(take["mlbam_id"])
            rest = pool[~pool["mlbam_id"].isin(taken)]
            repl[pos] = float(rest[value].iloc[min(bench_hitters, len(rest) - 1)]) if len(rest) else 0.0
        # UT: best remaining hitters regardless of position
        n_ut = teams * positions.get("UT", 0)
        rest = s[~s["mlbam_id"].isin(taken)]
        ut_take = rest.head(n_ut); taken |= set(ut_take["mlbam_id"])
        rest = s[~s["mlbam_id"].isin(taken)]
        repl["UT"] = float(rest[value].iloc[min(bench_hitters * teams, len(rest) - 1)]) if len(rest) else 0.0
        # a hitter's replacement level is the lowest among the positions he qualifies at (his easiest slot to win)
        def rl(e):
            cands = [repl[p] for p in e if p in repl] or [repl["UT"]]
            return min(min(cands), repl["UT"]) if positions.get("UT", 0) else min(cands)
        s["repl_pts"] = [rl(e) for e in elig]
        s["LPAR"] = s[value] - s["repl_pts"]
        s["repl_levels"] = str({k: round(v) for k, v in repl.items()})
        out.append(s)
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------------------------- league fit
def league_fit(ps: pd.DataFrame, scoring: dict, min_pa=300) -> pd.DataFrame:
    """Percentile of a hitter's per-PA value in *this* league minus his percentile under standard points and under
    5x5 roto. Positive = this league's scoring likes his profile more than the rest of the fantasy world does."""
    d = ps.copy()
    d["lp_pa"] = hitting_points(d, scoring) / d["PA"].replace(0, np.nan)
    d["std_pa"] = hitting_points(d, PRESETS["espn_points"]) / d["PA"].replace(0, np.nan)
    rv = roto_values(d)
    d["roto_z"] = rv["roto_z"]
    out = []
    for season, s in d.groupby("season"):
        s = s.copy(); q = s["PA"] >= min_pa
        for col, name in [("lp_pa", "pct_league"), ("std_pa", "pct_std_points"), ("roto_z", "pct_roto")]:
            s[name] = np.nan
            s.loc[q, name] = s.loc[q, col].rank(pct=True) * 100
        s["fit_vs_points"] = s["pct_league"] - s["pct_std_points"]
        s["fit_vs_roto"] = s["pct_league"] - s["pct_roto"]
        out.append(s)
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------------------------- roto
def roto_values(ps: pd.DataFrame, top_n=180, cats=("R", "HR", "RBI", "SB", "AVG")) -> pd.DataFrame:
    """Standard z-score valuation for 5x5 roto: z of counting cats among the top-N hitters, AVG weighted by AB."""
    d = ps.copy(); out = []
    for season, s in d.groupby("season"):
        s = s.copy()
        pool = s[s["PA"] >= 200].sort_values("pts", ascending=False).head(top_n)
        z = pd.Series(0.0, index=s.index)
        for c in cats:
            if c == "AVG":
                lg = pool["H"].sum() / pool["AB"].sum()
                contrib = (s["H"] - lg * s["AB"])            # hits above average for his AB
                mu, sd = (pool["H"] - lg * pool["AB"]).mean(), (pool["H"] - lg * pool["AB"]).std(ddof=0) or 1
                z += (contrib - mu) / sd
            else:
                mu, sd = pool[c].mean(), pool[c].std(ddof=0) or 1
                z += (s[c] - mu) / sd
        s["roto_z"] = z
        out.append(s)
    return pd.concat(out).sort_index()

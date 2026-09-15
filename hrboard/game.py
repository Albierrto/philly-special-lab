"""From plate-appearance probabilities to a game: lineup slots, starter vs bullpen share, and the day's distributions.

For a batter in lineup slot s, his k-th plate appearance is roughly the team's (s + 9(k-1))-th. Whether he gets it at
all (q) and whether the starter is still in (w) both come from real games:
    P(HR in game) = 1 - prod_k [ 1 - q(s, home, k) * ( w_k * p_HR vs starter + (1 - w_k) * p_HR vs that bullpen ) ]
Hits and total bases use the same per-PA mixes; a starter's strikeouts run over the batters he is expected to face.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import features as F

CLS = ["out", "k", "bb", "s1", "xb", "hr"]
MAX_K = 7            # plate appearances per batter considered
LG_BF = 22.0         # a starter's batters faced when we know nothing about him


def batter_hands(d: pd.DataFrame) -> pd.DataFrame:
    """R, L or S (switch) from the sides a batter has actually hit from."""
    g = d.groupby("batter")["stand"].agg(lambda s: "S" if s.nunique() > 1 and s.value_counts(normalize=True).min() > 0.1 else s.mode().iat[0])
    return g.rename("bats").reset_index()


def pitcher_hands(d: pd.DataFrame) -> pd.DataFrame:
    return d.groupby("pitcher")["p_throws"].agg(lambda s: s.mode().iat[0]).rename("throws").reset_index()


def stand_vs(bats, throws):
    """Side a batter hits from against a pitcher's hand."""
    bats = np.asarray(bats); throws = np.asarray(throws)
    return np.where(bats == "S", np.where(throws == "R", "L", "R"), bats)


def starters(d: pd.DataFrame) -> pd.DataFrame:
    """Starting lineups (first nine batters to come up for each club) with what each did that game."""
    d = d.sort_values(["game_pk", "at_bat_number"])
    first = d.groupby(["game_pk", "bat_team", "batter"], sort=False)["at_bat_number"].min().reset_index()
    first["slot"] = first.groupby(["game_pk", "bat_team"])["at_bat_number"].rank(method="first").astype(int)
    first = first[first["slot"] <= 9]
    d = d.copy(); d["tb"] = d["s1"] + 2 * (d["events"] == "double") + 3 * (d["events"] == "triple") + 4 * d["hr"]
    d["h"] = d["s1"] + d["xb"] + d["hr"]
    agg = d.groupby(["game_pk", "batter"]).agg(PA=("pa", "sum"), HR=("hr", "sum"), H=("h", "sum"), TB=("tb", "sum"),
                                               K=("k", "sum"), BB=("bb", "sum"))
    meta = d.groupby(["game_pk", "bat_team"]).agg(season=("season", "first"), game_date=("game_date", "first"),
                                                  is_home=("is_home", "first"), opp_sp=("opp_sp", "first"),
                                                  fld_team=("fld_team", "first"), team_pa=("pa", "sum"))
    out = first.merge(agg, left_on=["game_pk", "batter"], right_index=True, how="left")
    out = out.merge(meta, left_on=["game_pk", "bat_team"], right_index=True, how="left")
    return out.drop(columns="at_bat_number")


def pa_reach_table(st: pd.DataFrame) -> np.ndarray:
    """q[home, slot-1, k-1] = share of starters in that slot who got at least k plate appearances."""
    q = np.zeros((2, 9, MAX_K))
    for h in (0, 1):
        for s in range(1, 10):
            pa = st.loc[(st["is_home"] == h) & (st["slot"] == s), "PA"].to_numpy()
            for k in range(1, MAX_K + 1):
                q[h, s - 1, k - 1] = (pa >= k).mean() if len(pa) else 0.0
    return q


def sp_workload(d: pd.DataFrame) -> pd.DataFrame:
    """Batters faced per start, one row per start, plus the leak-free expectation going in (this season's starts before
    today, half weight on last season's, shrunk toward LG_BF with three phantom starts)."""
    s = d[d["vs_sp"] == 1].groupby(["opp_sp", "game_pk"]).agg(bf=("pa", "sum"), season=("season", "first"),
                                                              game_date=("game_date", "first")).reset_index()
    s = s.rename(columns={"opp_sp": "pitcher"}).sort_values(["pitcher", "game_date", "game_pk"])
    g = s.groupby(["pitcher", "season"])
    s["n_before"] = g.cumcount()
    s["bf_before"] = g["bf"].cumsum() - s["bf"]
    tot = s.groupby(["pitcher", "season"]).agg(n_prev=("bf", "size"), bf_prev=("bf", "sum")).reset_index()
    tot["season"] += 1
    s = s.merge(tot, on=["pitcher", "season"], how="left").fillna({"n_prev": 0, "bf_prev": 0})
    s["bf_exp"] = (s["bf_before"] + 0.5 * s["bf_prev"] + 3 * LG_BF) / (s["n_before"] + 0.5 * s["n_prev"] + 3)
    # recent leash: the last three starts say more about a pitcher's current workload than April does
    s["bf_last3"] = g["bf"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    s["bf_exp"] = np.where(s["bf_last3"].notna(), 0.6 * s["bf_exp"] + 0.4 * s["bf_last3"], s["bf_exp"])
    return s


def vs_sp_curve(d: pd.DataFrame, work: pd.DataFrame) -> dict:
    """P(the starter is still pitching to the team's n-th batter), as a function of n - expected batters faced."""
    x = d[["game_pk", "bat_team", "at_bat_number", "opp_sp", "vs_sp"]].copy()
    x["idx"] = x.groupby(["game_pk", "bat_team"])["at_bat_number"].rank(method="first")
    x = x.merge(work[["pitcher", "game_pk", "bf_exp"]], left_on=["opp_sp", "game_pk"], right_on=["pitcher", "game_pk"], how="inner")
    x["gap"] = (x["idx"] - x["bf_exp"]).round().clip(-30, 25).astype(int)
    tab = x.groupby("gap")["vs_sp"].mean()
    return {int(k): float(v) for k, v in tab.items()}


def bf_residuals(work: pd.DataFrame) -> dict:
    r = (work["bf"] - work["bf_exp"]).round().clip(-20, 12).astype(int)
    p = r.value_counts(normalize=True).sort_index()
    return {int(k): float(v) for k, v in p.items()}


def w_share(curve: dict, idx, bf_exp):
    gap = np.clip(np.round(np.asarray(idx) - np.asarray(bf_exp)), -30, 25).astype(int)
    keys = np.array(sorted(curve)); vals = np.array([curve[k] for k in keys])
    return np.interp(gap, keys, vals)


def bullpen_frame(d: pd.DataFrame) -> pd.DataFrame:
    """Relief plate appearances, relabelled so a club's whole bullpen is one 'pitcher' for the skill book."""
    bp = d[d["vs_sp"] == 0].copy()
    bp["same"] = (bp["stand"] == bp["p_throws"]).astype("int8")
    bp["pitcher"] = bp["fld_team"]
    return bp


def bullpen_same_share(bp: pd.DataFrame) -> pd.DataFrame:
    """As-of share of a bullpen's plate appearances against each batter side that came from a same-handed arm."""
    g = bp.groupby(["fld_team", "stand", "season", "game_date"]).agg(pa=("pa", "sum"), same=("same", "sum")).reset_index()
    g = g.sort_values(["fld_team", "stand", "season", "game_date"])
    c = g.groupby(["fld_team", "stand", "season"])[["pa", "same"]].cumsum()
    g["same_share"] = (c["same"] + 30 * np.where(g["stand"] == "R", 0.62, 0.30)) / (c["pa"] + 30)
    return g[["fld_team", "stand", "season", "game_date", "same_share"]]


# --------------------------------------------------------------------------- per-game distributions
def batter_game(P_sp: np.ndarray, P_bp: np.ndarray, slot, home, bf_exp, q: np.ndarray, curve: dict, start_p=None) -> dict:
    """P_sp, P_bp: (n, 6) class probabilities in CLS order. Returns arrays for each batter-game."""
    n = len(P_sp); slot = np.asarray(slot).astype(int); home = np.asarray(home).astype(int)
    ihr, is1, ixb = CLS.index("hr"), CLS.index("s1"), CLS.index("xb")
    no_hr = np.ones(n); no_h = np.ones(n); e_hr = np.zeros(n); e_pa = np.zeros(n); e_k = np.zeros(n)
    tb = np.zeros((n, 3)); tb[:, 0] = 1.0            # P(total bases = 0, 1, 2+)
    w_all = []
    for k in range(1, MAX_K + 1):
        qk = q[home, slot - 1, k - 1]
        idx = slot + 9 * (k - 1)
        w = w_share(curve, idx, bf_exp); w_all.append(w)
        mix = w[:, None] * P_sp + (1 - w[:, None]) * P_bp
        p_hr = mix[:, ihr]; p_h = mix[:, is1] + mix[:, ixb] + p_hr; p_2 = mix[:, ixb] + p_hr; p_1 = mix[:, is1]
        no_hr *= 1 - qk * p_hr; no_h *= 1 - qk * p_h
        e_hr += qk * p_hr; e_pa += qk; e_k += qk * mix[:, CLS.index("k")]
        t0 = tb[:, 0] * (1 - qk * (p_1 + p_2))
        t1 = tb[:, 1] * (1 - qk * (p_1 + p_2)) + tb[:, 0] * qk * p_1
        t2 = tb[:, 2] + tb[:, 1] * qk * (p_1 + p_2) + tb[:, 0] * qk * p_2
        tb = np.c_[t0, t1, t2]
    sp = 1.0 if start_p is None else np.asarray(start_p, dtype=float)
    return dict(p_hr=sp * (1 - no_hr), p_hit=sp * (1 - no_h), p_tb2=sp * tb[:, 2], e_hr=sp * e_hr, e_pa=sp * e_pa,
                e_k=sp * e_k, sp_share=np.mean(w_all[:3], axis=0))


def pitcher_ks(pk_by_slot: np.ndarray, bf_exp: float, resid: dict, max_k: int = 16) -> dict:
    """pk_by_slot: P(strikeout) for lineup slots 1..9 against this starter. Returns expected K and P(K >= n)."""
    offs = np.array(sorted(resid)); pr = np.array([resid[o] for o in offs]); pr = pr / pr.sum()
    bfs = np.clip(np.round(bf_exp + offs).astype(int), 1, 45)
    p_bf = {}
    for b, p in zip(bfs, pr):
        p_bf[b] = p_bf.get(b, 0.0) + p
    dist = np.zeros(max_k + 1); dist[0] = 1.0
    mix = np.zeros(max_k + 1); e_bf = 0.0
    for j in range(1, max(p_bf) + 1):
        p = pk_by_slot[(j - 1) % 9]
        new = dist * (1 - p); new[1:] += dist[:-1] * p; new[-1] += dist[-1] * p
        dist = new
        if j in p_bf:
            mix += p_bf[j] * dist; e_bf += p_bf[j] * j
    ge = mix[::-1].cumsum()[::-1]
    return dict(e_k=float((np.arange(max_k + 1) * mix).sum()), p_ge={n: float(ge[n]) for n in range(1, max_k + 1)}, e_bf=e_bf, dist=mix)


# --------------------------------------------------------------------------- team runs and win probability
def team_runs_rate(mix_rates: np.ndarray) -> float:
    """BaseRuns per out from per-PA outcome rates (CLS order) -> runs per 27 outs."""
    out, k, bb, s1, xb, hr = mix_rates
    h = s1 + xb + hr; tb = s1 + 2.09 * xb + 4 * hr
    A = h + bb - hr; B = (1.4 * tb - 0.6 * h - 3 * hr + 0.1 * bb) * 1.02; C = out + k; D = hr
    runs = A * B / (B + C) + D
    return runs / (out + k) * 27.0


def nb_pmf(mu: float, r: float, n: int = 30) -> np.ndarray:
    from scipy.stats import nbinom
    p = r / (r + mu)
    x = np.arange(n + 1)
    pm = nbinom.pmf(x, r, p); pm[-1] += 1 - pm.sum()
    return pm


def win_prob(mu_home: float, mu_away: float, r: float = 4.0, home_extra: float = 0.52) -> dict:
    ph, pa = nb_pmf(mu_home, r), nb_pmf(mu_away, r)
    joint = np.outer(ph, pa)
    win = np.tril(joint, -1).sum(); tie = np.trace(joint)
    tot = np.add.outer(np.arange(len(ph)), np.arange(len(pa)))
    return dict(p_home=float(win + tie * home_extra), p_tie9=float(tie),
                total_pmf=np.bincount(tot.ravel(), weights=joint.ravel()))

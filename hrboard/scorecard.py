"""Log each day's board and grade it once the games are played.

The saved board for a game is the last one built BEFORE first pitch: once a game is live or final its entry is frozen.
A game first seen after it started is saved with late=True and left out of the market comparison (in-game prices).
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

from . import context as C

TOP_N = (1, 3, 5, 10)


def log_picks(slate: dict, folder: Path) -> None:
    f = folder / f"{slate['date']}.json"
    old = json.loads(f.read_text()) if f.exists() else {}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    games = {str(g["pk"]): g for g in slate["games"]}
    by_pk = {}
    for h in slate["hitters"]:
        if (h.get("hr") or 0) < 0.01 or ((h.get("ps") or 0) < 0.25 and not h.get("il")): continue
        mk = h.get("mk") or {}
        by_pk.setdefault(str(h["pk"]), {"hitters": [], "pitchers": []})["hitters"].append(dict(
            id=h["id"], n=h["n"], t=h["t"], hr=round(h["hr"], 4), hit=round(h["hit"], 4), tb2=round(h["tb2"], 4), sl=h["sl"], ps=h["ps"], il=h["il"],
            dk_hr=_mp(mk, "dk", "hr1", 1), ks_hr=_mp(mk, "ks", "hr1", 0),
            dk_hit=_mp(mk, "dk", "hit1", 1), ks_hit=_mp(mk, "ks", "hit1", 0),
            dk_tb2=_mp(mk, "dk", "tb2", 1), ks_tb2=_mp(mk, "ks", "tb2", 0)))
    for p in slate["pitchers"]:
        mk = p.get("mk") or {}
        by_pk.setdefault(str(p["pk"]), {"hitters": [], "pitchers": []})["pitchers"].append(dict(
            id=p["id"], n=p["n"], ek=p["ek"], ge=p["ge"],
            dk={k: v[1] for k, v in (mk.get("dk") or {}).items() if k.startswith("k")},
            ks={k: v[0] for k, v in (mk.get("ks") or {}).items() if k.startswith("k")}))
    from . import value as VAL
    vb = {}
    for v in VAL.value_bets(slate):
        if v["verdict"] < 1: continue
        vb.setdefault(str(v["pk"]), []).append({k: (round(x, 4) if isinstance(x, float) else x) for k, x in v.items()
                                                if k in ("kind", "key", "id", "name", "site", "cost", "ev", "verdict", "p_model", "fair", "push", "line", "confirmed", "capped", "price_edge", "ratio")})
    out = dict(old)
    for pk, g in games.items():
        prev = old.get(pk)
        started = g["state"] in ("Live", "Final")
        if prev and prev.get("frozen"):
            continue
        entry = by_pk.get(pk, {"hitters": [], "pitchers": []})
        entry["value"] = vb.get(pk, [])
        mk = g.get("mk") or {}; dk = mk.get("dk") or {}; ks = (mk.get("ks") or {}).get("win") or {}
        entry.update(saved=now, state=g["state"], frozen=started, late=bool(started and not prev), env=slate.get("env"),
                     home=g["home"]["abbr"], away=g["away"]["abbr"],
                     p_home=(g.get("model") or {}).get("p_home"), mu=[(g.get("model") or {}).get("mu_home"), (g.get("model") or {}).get("mu_away")],
                     dk_home=dk.get("home_p"), dk_total=dk.get("total"),
                     ks_home=(ks.get(g["home"]["abbr"]) or [None])[0])
        if started and prev and not prev.get("frozen"):
            prev.update(frozen=True, state=g["state"]); out[pk] = prev      # keep the last pre-game board
        else:
            out[pk] = entry
    f.write_text(json.dumps(out, separators=(",", ":")))


ENV = dict(days=14, prior=150.0, lo=0.85, hi=1.15)


def env_factors(d: pd.DataFrame, folder: Path, today: str, cfg: dict = ENV) -> dict:
    """How far the league is running from the model on homers and 2+ total bases over the last two weeks, as an odds
    multiplier for the model's calibration. Graded from the logged boards: hitters in a posted lineup who played, their
    pre-game chance with any multiplier that board already carried taken back out (so the factor does not chase its
    own tail), shrunk toward 1 with `prior` expected events and clamped.

    Why: the season backtest ran 10-15% hot on homers from July on (2026's second-half home run rate fell below 2024-25)
    and 12% hot on the first graded week. On the 2026 backtest a trailing 14-day factor improves log loss on homers and
    total bases beyond a season-level recalibration (hits: no gain, so hits are left alone)."""
    from datetime import date, timedelta
    t0 = (date.fromisoformat(today) - timedelta(days=cfg["days"])).isoformat()
    x = d[d["season"] == d["season"].max()]
    res = x.groupby(["game_pk", "batter"]).agg(HR=("hr", "sum"), s1=("s1", "sum"), ev=("events", lambda e: (2 * (e == "double") + 3 * (e == "triple")).sum()))
    res["TB"] = res["s1"] + res["ev"] + 4 * res["HR"]
    have = set(x["game_pk"].unique())
    a = dict(hr=[0.0, 0.0, 0], tb2=[0.0, 0.0, 0])
    for f in sorted(folder.glob("*.json")):
        if not (t0 <= f.stem < today): continue
        for pk, g in json.loads(f.read_text()).items():
            pk = int(pk)
            if pk not in have or g.get("late"): continue
            env = g.get("env") or {}
            for h in g["hitters"]:
                if not h.get("il") or (pk, h["id"]) not in res.index: continue
                r = res.loc[(pk, h["id"])]
                for k, y in (("hr", r["HR"] >= 1), ("tb2", r["TB"] >= 2)):
                    p = h.get(k)
                    if p is None or not (0 < p < 1): continue
                    m = env.get(k) or 1.0; o = p / (1 - p) / m; p0 = o / (1 + o)       # the chance before that day's multiplier
                    a[k][0] += float(y); a[k][1] += p0; a[k][2] += 1
    out = dict(days=cfg["days"], since=t0)
    for k, (yy, pp, n) in a.items():
        if n < 200:
            out[k] = 1.0; out[f"{k}_detail"] = dict(n=n); continue
        rate = min(cfg["hi"], max(cfg["lo"], (yy + cfg["prior"]) / (pp + cfg["prior"])))
        pbar = pp / n
        out[k] = round(rate ** (1 / (1 - pbar)), 4)                                   # rate ratio -> odds multiplier
        out[f"{k}_detail"] = dict(n=n, actual=int(yy), expected=round(pp, 1), rate=round(rate, 4))
    return out


def _mp(mk, src, key, i):
    v = (mk.get(src) or {}).get(key)
    return v[i] if v else None


def _ll(p, y):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4); y = np.asarray(y, dtype=float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def scorecard(d: pd.DataFrame, folder: Path, today: str) -> dict:
    files = sorted(p for p in folder.glob("*.json") if p.stem < today)
    if not files:
        return dict(days=0)
    x = d[d["season"] == d["season"].max()]
    res_b = x.groupby(["game_pk", "batter"]).agg(HR=("hr", "sum"), H=("s1", "sum"), XB=("xb", "sum"), PA=("pa", "sum"))
    res_b["H"] = res_b["H"] + res_b["XB"] + res_b["HR"]
    tb = x.assign(tb=x["s1"] + 2 * (x["events"] == "double") + 3 * (x["events"] == "triple") + 4 * x["hr"]).groupby(["game_pk", "batter"])["tb"].sum()
    res_b["TB"] = tb
    res_p = x[x["vs_sp"] == 1].groupby(["game_pk", "opp_sp"])["k"].sum()
    have = set(x["game_pk"].unique())
    H, P, Gm, daily, VB = [], [], [], [], []
    for f in files:
        day = f.stem
        j = json.loads(f.read_text())
        sch = C.schedule(day, day, hydrate="team").set_index("game_pk") if j else pd.DataFrame()
        dayrows = []
        for pk, g in j.items():
            pk = int(pk)
            if pk not in sch.index or sch.at[pk, "abstract"] != "Final" or pk not in have:
                continue
            for h in g["hitters"]:
                r = res_b.loc[(pk, h["id"])] if (pk, h["id"]) in res_b.index else None
                row = dict(day=day, pk=pk, late=g.get("late", False), **h, y_hr=int(r is not None and r["HR"] > 0),
                           y_hit=int(r is not None and r["H"] > 0), y_tb2=int(r is not None and r["TB"] >= 2),
                           played=int(r is not None), HRn=int(r["HR"]) if r is not None else 0)
                H.append(row); dayrows.append(row)
            for p in g["pitchers"]:
                k = res_p.get((pk, p["id"]))
                if k is None: continue
                P.append(dict(day=day, pk=pk, late=g.get("late", False), id=p["id"], n=p["n"], ek=p["ek"], K=int(k), ge=p["ge"], dk=p.get("dk", {}), ks=p.get("ks", {})))
            hr_, ar_ = sch.at[pk, "home_runs"], sch.at[pk, "away_runs"]
            for v in ([] if g.get("late") else g.get("value", [])):
                res = _grade_value(v, pk, res_b, res_p, hr_, ar_)
                if res is not None: VB.append(dict(day=day, **v, result=res))
            if g.get("p_home") is not None and pd.notna(hr_):
                Gm.append(dict(day=day, pk=pk, late=g.get("late", False), p=g["p_home"], dk=g.get("dk_home"), ks=g.get("ks_home"),
                               y=int(hr_ > ar_), total=int(hr_ + ar_), mu=sum(v or 0 for v in g.get("mu", [])), dk_total=g.get("dk_total"),
                               home=g["home"], away=g["away"], score=f"{int(ar_)}-{int(hr_)}"))
        if dayrows:
            t = pd.DataFrame(dayrows).sort_values("hr", ascending=False).head(10)
            daily.append(dict(day=day, top=[dict(id=int(r.id), n=r.n, t=r.t, p=r.hr, y=int(r.HRn), dk=r.dk_hr, ks=r.ks_hr, played=int(r.played))
                                            for r in t.itertuples()]))
    out = dict(days=len(daily), since=files[0].stem, daily=daily[-10:][::-1])
    if H:
        h = pd.DataFrame(H)
        h["rank"] = h.groupby("day")["hr"].rank(ascending=False, method="first")
        out["top"] = {str(n): dict(picks=int((h["rank"] <= n).sum()), hits=int(h.loc[h["rank"] <= n, "y_hr"].sum()),
                                   expected=round(float(h.loc[h["rank"] <= n, "hr"].sum()), 2)) for n in TOP_N}
        for k in ("hr", "hit", "tb2"):
            yk = h[f"y_{k}"]
            blk = dict(n=int(len(h)), actual=round(float(yk.mean()), 4), model=round(float(h[k].mean()), 4), ll_model=round(_ll(h[k], yk), 4))
            for m in ("dk", "ks"):
                c = f"{m}_{k}"
                sub = h[h[c].notna() & ~h["late"]] if c in h else h.iloc[0:0]
                if len(sub) >= 20:
                    blk[m] = dict(n=int(len(sub)), actual=round(float(sub[f"y_{k}"].mean()), 4), market=round(float(sub[c].mean()), 4),
                                  model=round(float(sub[k].mean()), 4), ll_market=round(_ll(sub[c], sub[f"y_{k}"]), 4),
                                  ll_model=round(_ll(sub[k], sub[f"y_{k}"]), 4))
            out[k] = blk
        bins = pd.cut(h["hr"], [0, .08, .12, .16, .2, .25, 1])
        out["hr_calib"] = [dict(bin=str(b), n=int(len(g)), model=round(float(g["hr"].mean()), 3), actual=round(float(g["y_hr"].mean()), 3))
                           for b, g in h.groupby(bins, observed=True)]
    if P:
        p = pd.DataFrame(P)
        out["k"] = dict(n=int(len(p)), mae=round(float((p["ek"] - p["K"]).abs().mean()), 2), mean_pred=round(float(p["ek"].mean()), 2),
                        mean_actual=round(float(p["K"].mean()), 2))
    if Gm:
        g = pd.DataFrame(Gm)
        fav = (g["p"] >= 0.5).astype(int)
        blk = dict(n=int(len(g)), right=int((fav == g["y"]).sum()), ll_model=round(_ll(g["p"], g["y"]), 4))
        for m in ("dk", "ks"):
            sub = g[g[m].notna() & ~g["late"]]
            if len(sub) >= 5:
                blk[m] = dict(n=int(len(sub)), right=int(((sub[m] >= .5).astype(int) == sub["y"]).sum()), ll_market=round(_ll(sub[m], sub["y"]), 4),
                              ll_model=round(_ll(sub["p"], sub["y"]), 4))
        out["games"] = blk
        out["recent_games"] = g.sort_values("day").tail(30)[["day", "home", "away", "p", "dk", "ks", "y", "score"]].iloc[::-1].to_dict(orient="records")
    if VB:
        v = pd.DataFrame(VB)
        v["profit"] = np.where(v["result"] == "win", 1 / v["cost"] - 1, np.where(v["result"] == "loss", -1.0, 0.0))
        def blk(x):
            settled = x[x["result"].isin(["win", "loss"])]
            return dict(n=int(len(settled)), wins=int((x["result"] == "win").sum()), losses=int((x["result"] == "loss").sum()),
                        pushes=int((x["result"] == "push").sum()), voids=int((x["result"] == "void").sum()),
                        units=round(float(x["profit"].sum()), 2), roi=round(float(settled["profit"].mean()), 4) if len(settled) else None,
                        expected=round(float(settled["ev"].mean()), 4) if len(settled) else None)
        out["value"] = dict(all=blk(v), value=blk(v[v["verdict"] == 2]), slight=blk(v[v["verdict"] == 1]),
                            by_kind={k: blk(x) for k, x in v.groupby("kind")},
                            recent=v.sort_values("day").tail(25).iloc[::-1][["day", "kind", "key", "name", "site", "cost", "ev", "verdict", "result"]]
                            .to_dict(orient="records"))
    return out


def _grade_value(v, pk, res_b, res_p, home_runs, away_runs):
    """win / loss / push / void for one logged value pick, or None if its result is not known yet."""
    kind = v["kind"]
    if kind in ("hr", "hit", "tb") or kind.startswith(("hit", "tb", "hr")):
        if (pk, v["id"]) not in res_b.index: return "void"          # did not play: DraftKings voids, Kalshi refunds
        r = res_b.loc[(pk, v["id"])]
        key = v.get("key") or {"hr": "hr1", "hit": "hit1", "tb": "tb2"}[kind]      # hr1, hit3, tb2 ... : stat and threshold
        n = int("".join(ch for ch in key if ch.isdigit()) or 1)
        stat = "HR" if key.startswith("hr") else "H" if key.startswith("hit") else "TB"
        return "win" if r[stat] >= n else "loss"
    if kind == "k":
        k = res_p.get((pk, v["id"]))
        if k is None: return "void"
        return "win" if k >= int(v["key"][1:]) else "loss"
    if pd.isna(home_runs) or pd.isna(away_runs): return None
    if kind == "ml":
        home_won = home_runs > away_runs
        return "win" if (home_won == (v["key"] == "home")) else "loss"
    if kind == "tot":
        t = home_runs + away_runs; line = v.get("line")
        if line is None: return None
        if t == line: return "push"
        return "win" if ((t > line) == (v["key"] == "over")) else "loss"
    return None


def slim_report(r: dict) -> dict:
    if not r: return {}
    keep = {}
    for k in ("hr", "hr_naive", "hit", "tb2", "k_ladder", "win", "win_home_only"):
        if k in r: keep[k] = {a: round(b, 4) for a, b in r[k].items() if a in ("n", "base", "mean_p", "logloss", "logloss_const", "auc")}
    for k in ("top", "sp_k", "totals", "hr_calib", "k_calib", "win_calib", "pa_2026"):
        if k in r: keep[k] = r[k]
    return keep

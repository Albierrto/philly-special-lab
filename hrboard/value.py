"""Model vs market at the best price: the same arithmetic the page's Best bets tab runs, in Python, so every pre-game
build can log its value picks and the scorecard can grade them (profit per $1 at the logged price).

fair chance  = average of the markets' own chances, margins removed (exchange midpoints when tight; DraftKings'
               one-sided player prices divided by how rich they run against Kalshi today; sportsbook two-way lines
               de-vigged side against side)
blend        = w x model + (1 - w) x fair          (w by bet type, VALUE_CFG; small, see below)
cost         = what a $1 payout costs at the best site, both exchanges' taker fees included
return / $1  = blend x (1 - push) / cost + push - 1
price edge   = fair  x (1 - push) / cost + push - 1   (the market's own number against the best price: line shopping)
Value        = price edge >= +2% and return >= +4%;  Slight = price edge >= 0 and return >= +1.5%
capped       = model odds >= 1.6 x market odds -> no verdict, whatever the return

Why the weights are small and the cap exists (fitted 2026-09-20 on the first four graded days, 842 hitter-games and 94
starts with model, DraftKings and Kalshi side by side): the weight that minimises log loss is 0.00-0.13 for every bet
type, with 80% bands reaching 0.3-0.5, so the model adds little the markets do not already know. And on the 222 settled
picks the return fell monotonically with how far the model sat above the market: under 1.15x the market's odds +62%,
1.15-1.3x +16%, 1.3-1.6x +9%, 1.6-2x -43%, over 2x -81%. Selecting on disagreement selects the model's mistakes. The
picks that paid were the ones where the best PRICE beat the market's own number.
"""
from __future__ import annotations
import math
from collections import Counter
from statistics import median

VALUE_CFG = dict(
    w=dict(hr=0.15, hit=0.10, tb=0.10, k=0.15, ml=0.05, tot=0.05),
    dk_ratio=dict(hr=1.2, hit=1.07, tb=1.11, k=1.06),
    good=0.04, lean=0.015, price_good=0.02, price_lean=0.0, cap_ratio=1.6,
    ks_tight=0.06, min_ratio_pairs=8, pm_fee=0.05,
)
KIND_KEY = dict(hr="hr1", hit="hit1", tb="tb2")


def series_of(key: str) -> str:
    return "hr" if key.startswith("hr") else "hit" if key.startswith("hit") else "tb" if key.startswith("tb") else "k"


def amer_p(o):
    try: v = float(str(o).replace("+", "").replace("−", "-"))
    except (TypeError, ValueError): return None
    if not math.isfinite(v) or v == 0: return None
    return 100 / (v + 100) if v > 0 else -v / (-v + 100)


def ks_fee(p, mult):
    if p is None or not (0 < p < 1): return 0.0
    return math.ceil(round(100 * 0.07 * mult * p * (1 - p) * 1e6) / 1e6) / 100


def ks_quote(entry, series, fees):
    if not entry: return None
    ask = entry[2]
    if ask is None or not (0 < ask < 1): return None
    fee = ks_fee(ask, fees.get(series, 1.0) if fees.get(series) is not None else 1.0)
    return dict(ask=ask, fee=fee, cost=min(ask + fee, 0.999))


def ks_tight_mid(entry, cfg):
    if not entry: return None
    bid, ask = entry[1], entry[2]
    return (bid + ask) / 2 if (bid and ask and bid > 0 and ask > 0 and ask - bid <= cfg["ks_tight"]) else None


def pm_mid(entry):
    if not entry: return None
    m = entry[0]
    return m if (m is not None and 0 < m < 1) else None


def pm_fee(p, rate=VALUE_CFG["pm_fee"]):
    """Polymarket's taker fee on sports, per share, charged when the order fills."""
    if p is None or not (0 < p < 1): return 0.0
    return math.floor(rate * p * (1 - p) * 1e5 + 0.5) / 1e5


def pm_cost(ask, cfg=VALUE_CFG):
    return None if (ask is None or not (0 < ask < 0.98)) else min(ask + pm_fee(ask, cfg["pm_fee"]), 0.999)


def mk(who, src, key):
    v = ((who.get("mk") or {}).get(src) or {}).get(key)
    return v if v else None


def player_offers(sl, who, key, games):
    g = games[who["pk"]]; gm = g.get("mk") or {}; series = series_of(key); fees = (sl.get("markets") or {}).get("ks_fee") or {}
    out = []
    dk = mk(who, "dk", key)
    if dk and dk[1] is not None:
        out.append(dict(site="DraftKings", cost=dk[1], label=dk[0]))
    kq = ks_quote(mk(who, "ks", key), series, fees)
    if kq:
        out.append(dict(site="Kalshi", cost=kq["cost"], label=f"{round(kq['cost'] * 100)}c"))
    pm = mk(who, "pm", key)
    pc = pm_cost(pm[1]) if pm else None
    if pc is not None:
        out.append(dict(site="Polymarket", cost=pc, label=f"{round(pc * 100)}c"))
    return sorted(out, key=lambda o: o["cost"])


def dk_ratios(sl, games, cfg):
    pre = {pk for pk, g in games.items() if g["state"] == "Preview"}
    pairs = {k: [] for k in ("hr", "hit", "tb", "k")}
    for w in sl["hitters"] + sl["pitchers"]:
        if w["pk"] not in pre: continue
        for key, d in ((w.get("mk") or {}).get("dk") or {}).items():
            mid = ks_tight_mid(mk(w, "ks", key), cfg)
            if mid and mid > 0.03 and d and d[1]:
                pairs[series_of(key)].append(d[1] / mid)
    return {k: (median(v) if len(v) >= cfg["min_ratio_pairs"] else cfg["dk_ratio"][k]) for k, v in pairs.items()}


def prop_fair(who, key, ratios, cfg):
    F = []
    km = ks_tight_mid(mk(who, "ks", key), cfg)
    if km is not None: F.append(km)
    pmm = pm_mid(mk(who, "pm", key))
    if pmm is not None: F.append(pmm)
    d = mk(who, "dk", key)
    if d and d[1]: F.append(min(0.99, d[1] / ratios[series_of(key)]))
    return sum(F) / len(F) if F else None


def odds(p):
    return p / (1 - p)


def assess(kind, p_model, fair, offers, cfg, push=0.0):
    if not offers or fair is None or p_model is None or not (0 < fair < 1) or not (0 < p_model < 1): return None
    w = cfg["w"][kind]; best = offers[0]
    blend = w * p_model + (1 - w) * fair
    ev = blend * (1 - push) / best["cost"] + push - 1
    price_edge = fair * (1 - push) / best["cost"] + push - 1
    ratio = odds(p_model) / odds(fair)
    capped = ratio >= cfg["cap_ratio"]
    if capped: verdict = 0
    elif price_edge >= cfg["price_good"] and ev >= cfg["good"]: verdict = 2
    elif price_edge >= cfg["price_lean"] and ev >= cfg["lean"]: verdict = 1
    else: verdict = 0
    return dict(p_model=p_model, fair=fair, blend=blend, site=best["site"], cost=best["cost"], label=best["label"], ev=ev,
                price_edge=price_edge, ratio=ratio, capped=capped, verdict=verdict, push=push, n_sites=len(offers))


def game_rows(sl, g):
    """Every site's moneyline and main-line total for one game, mirroring the page's gameOffers()."""
    gm = g.get("mk") or {}; fees = (sl.get("markets") or {}).get("ks_fee") or {}
    rows = []
    books = (gm.get("books") or {}).get("list") or []
    for b in books:
        r = dict(site=b.get("name"), ml={}, tot={}, ex=False)
        for side in ("away", "home"):
            v = (b.get("ml") or {}).get(side)
            if v and v[0] is not None: r["ml"][side] = amer_p(v[0])
        for side in ("over", "under"):
            v = (b.get("tot") or {}).get(side)
            if v and v[1] is not None: r["tot"][side] = (float(v[0]), amer_p(v[1]))
        rows.append(r)
    dk = gm.get("dk") or {}
    if dk.get("home_ml") and not any(b.get("key") == "draftkings" for b in books):
        r = dict(site="DraftKings", ml=dict(away=amer_p(dk.get("away_ml")), home=amer_p(dk.get("home_ml"))), tot={}, ex=False)
        if dk.get("total") is not None:
            if dk.get("over"): r["tot"]["over"] = (float(dk["total"]), amer_p(dk["over"]))
            if dk.get("under"): r["tot"]["under"] = (float(dk["total"]), amer_p(dk["under"]))
        rows.append(r)
    ks = gm.get("ks") or {}
    if ks.get("win") or ks.get("total"):
        r = dict(site="Kalshi", ml={}, tot={}, totals={}, ex=True)
        for side in ("away", "home"):
            e = (ks.get("win") or {}).get(g[side]["abbr"])
            q = ks_quote([e[0], e[1], e[2]], "game", fees) if e else None
            if q: r["ml"][side] = q["cost"]
        for line, e in (ks.get("total") or {}).items():
            q = ks_quote(e, "total", fees)
            bid = e[1]
            no_ask = 1 - bid if (bid is not None and bid > 0) else None
            nf = ks_fee(no_ask, fees.get("total", 1.0)) if no_ask else 0
            r["totals"][line] = dict(over=(float(line), q["cost"]) if q else None,
                                     under=(float(line), min(no_ask + nf, 0.999)) if (no_ask and no_ask < 0.99) else None)
        rows.append(r)
    pm = gm.get("pm") or {}
    if pm.get("ml") or pm.get("totals"):
        r = dict(site="Polymarket", ml={}, tot={}, totals={}, ex=True)
        for side in ("away", "home"):
            e = (pm.get("ml") or {}).get(side)
            c = pm_cost(e[1]) if e else None
            if c is not None: r["ml"][side] = c
        for line, t in (pm.get("totals") or {}).items():
            r["totals"][line] = {}
            for side in ("over", "under"):
                e = t.get(side)
                c = pm_cost(e[1]) if e else None
                if c is not None: r["totals"][line][side] = (float(line), c)
        rows.append(r)
    lines = [(r["tot"].get("over") or r["tot"].get("under"))[0] for r in rows if (r["tot"].get("over") or r["tot"].get("under"))]
    if dk.get("total") is not None:
        main = float(dk["total"])
    elif lines:
        main = float(Counter(lines).most_common(1)[0][0])
    else:
        main = None
    for r in rows:
        if "totals" in r:
            t = r["totals"].get(f"{main:g}" if main is not None else "", {}) or {}
            r["tot"] = {k: v for k, v in (("over", t.get("over")), ("under", t.get("under"))) if v}
    return rows, main


def game_fair(g, rows, main, kind, side, cfg):
    gm = g.get("mk") or {}; F = []
    if kind == "ml":
        for r in rows:
            if r["ex"] or not r["ml"].get("away") or not r["ml"].get("home"): continue
            ph = r["ml"]["home"] / (r["ml"]["home"] + r["ml"]["away"])
            F.append(ph if side == "home" else 1 - ph)
        e = ((gm.get("ks") or {}).get("win") or {}).get(g[side]["abbr"])
        km = ks_tight_mid([e[0], e[1], e[2]], cfg) if e else None
        if km is not None: F.append(km)
        pl = ((gm.get("pm") or {}).get("ml") or {}).get(side)
        if pl and pl[0]: F.append(pl[0])
    else:
        if main is None: return None
        for r in rows:
            if r["ex"]: continue
            o, u = r["tot"].get("over"), r["tot"].get("under")
            if not o or not u or o[0] != main or u[0] != main: continue
            po = o[1] / (o[1] + u[1]); F.append(po if side == "over" else 1 - po)
        kt = ((gm.get("ks") or {}).get("total") or {}).get(f"{main:g}")
        km = ks_tight_mid(kt, cfg) if kt else None
        if km is not None: F.append(km if side == "over" else 1 - km)
        pt = ((gm.get("pm") or {}).get("totals") or {}).get(f"{main:g}")
        pe = (pt or {}).get(side)
        if pe and pe[0]: F.append(pe[0])
    return sum(F) / len(F) if F else None


def value_bets(sl: dict, cfg: dict = VALUE_CFG) -> list[dict]:
    games = {g["pk"]: g for g in sl["games"]}
    ratios = dk_ratios(sl, games, cfg)
    out = []
    for h in sl["hitters"]:
        if games[h["pk"]]["state"] != "Preview" or h.get("zsp") is None: continue
        confirmed = h["lp"] and h["il"]
        if not confirmed and (h["lp"] or (h["ps"] or 0) < 0.6): continue
        for kind in ("hr", "hit", "tb"):
            key = KIND_KEY[kind]; p = h.get({"hr": "hri", "hit": "hiti", "tb": "tbi"}[kind])
            offers = player_offers(sl, h, key, games)
            v = assess(kind, p, prop_fair(h, key, ratios, cfg), offers, cfg)
            if v: out.append(dict(v, kind=kind, key=key, id=h["id"], pk=h["pk"], name=h["n"], confirmed=confirmed))
    for p in sl["pitchers"]:
        if games[p["pk"]]["state"] != "Preview": continue
        for n in range(3, 11):
            key = f"k{n}"; offers = player_offers(sl, p, key, games)
            v = assess("k", (p.get("ge") or {}).get(str(n)), prop_fair(p, key, ratios, cfg), offers, cfg)
            if v: out.append(dict(v, kind="k", key=key, id=p["id"], pk=p["pk"], name=p["n"], confirmed=True))
    for g in sl["games"]:
        m = g.get("model")
        if g["state"] != "Preview" or not m: continue
        rows, main = game_rows(sl, g)
        for side in ("away", "home"):
            offers = sorted([dict(site=r["site"], cost=r["ml"][side], label="") for r in rows if r["ml"].get(side)], key=lambda o: o["cost"])
            pm_ = m["p_home"] if side == "home" else 1 - m["p_home"]
            v = assess("ml", pm_, game_fair(g, rows, main, "ml", side, cfg), offers, cfg)
            if v: out.append(dict(v, kind="ml", key=side, id=g["pk"], pk=g["pk"], name=g[side]["abbr"], confirmed=True))
        if main is not None:
            pmf = m.get("total_pmf") or []
            p_over = sum(v for t, v in enumerate(pmf) if t > main); p_push = sum(v for t, v in enumerate(pmf) if t == main)
            p_under = 1 - p_over - p_push
            for side in ("over", "under"):
                offers = sorted([dict(site=r["site"], cost=r["tot"][side][1], label="") for r in rows
                                 if r["tot"].get(side) and r["tot"][side][0] == main], key=lambda o: o["cost"])
                pm_ = (p_over if side == "over" else p_under) / max(1e-6, 1 - p_push)
                v = assess("tot", pm_, game_fair(g, rows, main, "tot", side, cfg), offers, cfg, p_push)
                if v: out.append(dict(v, kind="tot", key=side, line=main, id=g["pk"], pk=g["pk"], name=f"{g['away']['abbr']}@{g['home']['abbr']}", confirmed=True))
    return sorted(out, key=lambda v: -v["ev"])

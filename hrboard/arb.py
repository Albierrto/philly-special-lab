"""Arbitrage: the same bet priced on two sites, where buying both sides costs less than the dollar it pays.

Kalshi and Polymarket are exchanges, so every market has two sides you can buy outright. Buy one side on one site and
the other side on the other, and one of them has to win: exactly $1 comes back per contract. If the two together cost
less than $1 once the sites take their cut, the difference is locked the moment both orders fill.

What a side costs:
  Kalshi      yes = the ask; the other side = 1 - the bid (that is what "No" costs)
  Polymarket  yes = the ask; the other side = 1 - the bid (the opposite token's ask)
  Sportsbook  the price as a chance: -150 costs 60c per $1, +300 costs 25c

The cut:
  Kalshi      taker fee = 0.07 x (series multiplier, 0.5 on baseball) x contracts x p x (1 - p), rounded UP to the
              next cent on the whole order. At one contract that rounding alone is a cent, which is why a real
              arbitrage needs size.
  Polymarket  taker fee = 0.05 x shares x p x (1 - p) on sports, in USDC, taken when the order fills.
  Sportsbook  nothing on top; the cut is already inside the price.

None of this is free money. Kalshi refunds a scratched hitter at a fair price while Polymarket settles him Under, a
rained-out game settles differently at each site, prices move between the two clicks, and a book fills less than it
shows. Every row carries what can break it.
"""
from __future__ import annotations
import math

CFG = dict(
    ks_rate=0.07,        # Kalshi taker rate, times the series fee_multiplier (0.5 on the baseball series)
    pm_rate=0.05,        # Polymarket taker rate on sports markets
    pm_min=5,            # Polymarket will not take an order under 5 shares
    min_roi=0.003,       # under 0.3% back on the money it is noise, not an edge
    max_size=1000,       # never suggest more size than this, whatever the book shows
)
KIND_NAME = dict(hr="home run", hit="hit", tb="total bases", k="strikeouts")


def _r(x, n=4):
    """Round the way the page's JavaScript does, so the two agree to the cent."""
    return math.floor(x * 10 ** n + 0.5) / 10 ** n


# ------------------------------------------------------------------ the money
def ks_fee(p, contracts, mult=0.5, rate=CFG["ks_rate"]):
    """Kalshi rounds the fee up to the next cent on the order, not per contract."""
    if not (0 < p < 1) or contracts <= 0: return 0.0
    return math.ceil(round(100 * rate * mult * contracts * p * (1 - p), 6)) / 100


def pm_fee(p, shares, rate=CFG["pm_rate"]):
    if not (0 < p < 1) or shares <= 0: return 0.0
    return _r(rate * shares * p * (1 - p), 5)


def leg_fee(leg, n, cfg=CFG, price=None):
    p = leg["price"] if price is None else price
    if leg["venue"] == "ks": return ks_fee(p, n, leg.get("mult", 0.5), cfg["ks_rate"])
    if leg["venue"] == "pm": return pm_fee(p, n, cfg["pm_rate"])
    return 0.0


def size_of(a, b, cfg=CFG):
    """(contracts, is that number a guess). None means the pair cannot be done at these prices: Kalshi is showing less
    than Polymarket's five-share minimum. Sportsbooks and Polymarket's gamma feed publish no size at all, so a pair that
    leans on either is priced at the smallest ticket and flagged."""
    pm = any(l["venue"] == "pm" for l in (a, b))
    known = [l["size"] for l in (a, b) if l.get("size")]
    if not known: return (cfg["pm_min"] if pm else 1), True
    n = math.floor(min(min(known), cfg["max_size"]))
    if n < 1 or (pm and n < cfg["pm_min"]): return None, False
    return n, len(known) < 2


def pair(a, b, cfg=CFG, size=None):
    """Buy both sides at these prices: what it costs, what it pays, what is left."""
    if a is None or b is None: return None
    unknown = False
    if size is None:
        size, unknown = size_of(a, b, cfg)
        if size is None: return None
    n = math.floor(size)
    if n < 1: return None
    fa, fb = leg_fee(a, n, cfg), leg_fee(b, n, cfg)
    cost = n * (a["price"] + b["price"]) + fa + fb
    if cost <= 0: return None
    return dict(size=n, size_unknown=unknown, cost=_r(cost), profit=_r(n - cost), roi=(n - cost) / cost,
                fees=_r(fa + fb), legs=[dict(a, fee=_r(fa)), dict(b, fee=_r(fb))])


def amer_p(o):
    try: v = float(str(o).replace("+", "").replace("−", "-"))
    except (TypeError, ValueError): return None
    if not math.isfinite(v) or v == 0: return None
    return 100 / (v + 100) if v > 0 else -v / (-v + 100)


# ------------------------------------------------------------------ reading the payload
def _n(x):
    return x if isinstance(x, (int, float)) and x == x else None


def ks_legs(entry, mult, link, yes_label, no_label):
    """Both sides of one Kalshi market, entry [mid, bid, ask, ticker, bid size, ask size]."""
    if not entry: return {}
    bid, ask = _n(entry[1]), _n(entry[2])
    tick = entry[3] if len(entry) > 3 else None
    bsz = _n(entry[4]) if len(entry) > 4 else None
    asz = _n(entry[5]) if len(entry) > 5 else None
    out = {}
    if ask is not None and 0 < ask < 1 and asz:
        out["yes"] = dict(venue="ks", site="Kalshi", side="yes", price=ask, size=asz, mult=mult, link=link, what=yes_label, id=tick)
    if bid is not None and 0 < bid < 1 and bsz:
        out["no"] = dict(venue="ks", site="Kalshi", side="no", price=_r(1 - bid), size=bsz, mult=mult, link=link,
                         what=no_label, id=tick)
    return out


def pm_legs(entry, link, yes_label, no_label, slug=None):
    """Both sides of one Polymarket market, entry [market price, ask, bid, slug?]. Gamma quotes the first outcome; the
    other outcome's ask is one minus that bid. Gamma does not publish size, so depth is checked separately."""
    if not entry: return {}
    ask, bid = _n(entry[1]), (_n(entry[2]) if len(entry) > 2 else None)
    out = {}
    if ask is not None and 0 < ask < 0.98:
        out["yes"] = dict(venue="pm", site="Polymarket", side="yes", price=ask, size=None, link=link, what=yes_label, id=slug)
    if bid is not None and 0 < bid < 0.98:
        out["no"] = dict(venue="pm", site="Polymarket", side="no", price=_r(1 - bid), size=None, link=link,
                         what=no_label, id=slug)
    return out


def book_leg(name, odds, link, what):
    p = amer_p(odds)
    return None if not p else dict(venue="book", site=name, side="yes", price=_r(p), size=None, link=link,
                                   what=f"{what} {odds}")


def best_pair(A, B, cfg=CFG):
    """The cheapest way to own both sides. The two cheapest prices are not always a pair: the cheapest of each often
    turns out to be the two sides of one Kalshi market, and a book never crosses itself."""
    A = sorted([o for o in A if o and 0 < o["price"] < 1], key=lambda o: (o["price"], -(o["size"] or 0)))[:6]
    B = sorted([o for o in B if o and 0 < o["price"] < 1], key=lambda o: (o["price"], -(o["size"] or 0)))[:6]
    best = None
    for a in A:
        for b in B:
            if a.get("id") and a.get("id") == b.get("id"): continue
            p = pair(a, b, cfg)
            if p and (best is None or p["roi"] > best["roi"]): best = p
    return best


def _dk_row(gm):
    """DraftKings from ESPN, only when the books feed has not already got it."""
    dk = gm.get("dk") or {}
    books = (gm.get("books") or {}).get("list") or []
    return dk if dk.get("home_ml") and not any(b.get("key") == "draftkings" for b in books) else {}


def game_sides(sl, g):
    """Every two-sided market on one game: (title, sub, kind, offers for A, offers for B, risk)."""
    gm = g.get("mk") or {}
    ks, pm = gm.get("ks") or {}, gm.get("pm") or {}
    fees = (sl.get("markets") or {}).get("ks_fee") or {}
    kurl = ks.get("urls") or {}
    books = (gm.get("books") or {}).get("list") or []
    A, H = g["away"]["abbr"], g["home"]["abbr"]
    out = []

    a_off, h_off = [], []
    for side, abbr, other, mine, theirs in (("away", A, H, a_off, h_off), ("home", H, A, h_off, a_off)):
        e = (ks.get("win") or {}).get(abbr)
        kl = ks_legs([e[0], e[1], e[2], e[4], e[5], e[6]] if e and len(e) > 6 else None,
                     fees.get("game", 0.5), kurl.get("game"), f"{abbr} wins", f"{other} wins (No on {abbr})")
        if kl.get("yes"): mine.append(kl["yes"])
        if kl.get("no"): theirs.append(kl["no"])
        pl = pm_legs((pm.get("ml") or {}).get(side), pm.get("url"), f"{abbr} wins", f"{other} wins", slug=pm.get("slug"))
        if pl.get("yes"): mine.append(pl["yes"])
        if pl.get("no"): theirs.append(pl["no"])
    for b in books:
        for side, abbr, mine in (("away", A, a_off), ("home", H, h_off)):
            v = (b.get("ml") or {}).get(side)
            o = book_leg(b.get("name"), v[0] if v else None, (v[1] if v and len(v) > 1 else None) or b.get("link"), f"{abbr} moneyline")
            if o: mine.append(o)
    dk = _dk_row(gm); L = dk.get("links") or {}
    for side, abbr, odds, mine in (("away", A, dk.get("away_ml"), a_off), ("home", H, dk.get("home_ml"), h_off)):
        o = book_leg("DraftKings", odds, L.get(f"{side}_ml") or dk.get("url"), f"{abbr} moneyline")
        if o: mine.append(o)
    if a_off and h_off:
        out.append((f"{A} @ {H}", "moneyline", "ml", a_off, h_off, "game"))

    for line in sorted(set(ks.get("total") or {}) | set(pm.get("totals") or {}), key=float):
        o_off, u_off = [], []
        kl = ks_legs((ks.get("total") or {}).get(line), fees.get("total", 0.5), kurl.get("total") or kurl.get("game"),
                     f"over {line} runs", f"under {line} runs")
        if kl.get("yes"): o_off.append(kl["yes"])
        if kl.get("no"): u_off.append(kl["no"])
        t = (pm.get("totals") or {}).get(line) or {}
        for side, mine, theirs in (("over", o_off, u_off), ("under", u_off, o_off)):
            flip = "under" if side == "over" else "over"
            pl = pm_legs(t.get(side), pm.get("url"), f"{side} {line} runs", f"{flip} {line} runs", slug=t.get("slug"))
            if pl.get("yes"): mine.append(pl["yes"])
            if pl.get("no"): theirs.append(pl["no"])
        for b in books:
            for side, mine in (("over", o_off), ("under", u_off)):
                v = (b.get("tot") or {}).get(side)
                if not v or v[1] is None or f"{float(v[0]):g}" != line: continue
                o = book_leg(b.get("name"), v[1], (v[2] if len(v) > 2 else None) or b.get("link"), f"{side} {line}")
                if o: mine.append(o)
        if dk.get("total") is not None and f"{float(dk['total']):g}" == line:
            for side, odds, mine in (("over", dk.get("over"), o_off), ("under", dk.get("under"), u_off)):
                o = book_leg("DraftKings", odds, L.get(side) or dk.get("url"), f"{side} {line}")
                if o: mine.append(o)
        if o_off and u_off:
            out.append((f"{A} @ {H}", f"total runs {line}", "tot", o_off, u_off, "game"))
    return out


def prop_sides(sl, who, g):
    gm = g.get("mk") or {}
    ks, pm = gm.get("ks") or {}, gm.get("pm") or {}
    fees = (sl.get("markets") or {}).get("ks_fee") or {}
    mkw = who.get("mk") or {}
    out = []
    for key in sorted(set(mkw.get("ks") or {}) | set(mkw.get("pm") or {})):
        series = "hr" if key.startswith("hr") else "hit" if key.startswith("hit") else "tb" if key.startswith("tb") else "k"
        n = "".join(c for c in key if c.isdigit()) or "1"
        yes_l, no_l = f"{n}+ {KIND_NAME[series]}", f"under {n} {KIND_NAME[series]}"
        yes_off, no_off = [], []
        kl = ks_legs((mkw.get("ks") or {}).get(key), fees.get(series, 0.5), (ks.get("urls") or {}).get(series), yes_l, no_l)
        if kl.get("yes"): yes_off.append(kl["yes"])
        if kl.get("no"): no_off.append(kl["no"])
        pe = (mkw.get("pm") or {}).get(key)
        pl = pm_legs(pe, pm.get("props_url") or pm.get("url"), yes_l, no_l, slug=(pe[3] if pe and len(pe) > 3 else None))
        if pl.get("yes"): yes_off.append(pl["yes"])
        if pl.get("no"): no_off.append(pl["no"])
        d = (mkw.get("dk") or {}).get(key)
        if d and d[1] is not None and 0 < d[1] < 1:
            yes_off.append(dict(venue="book", site="DraftKings", side="yes", price=_r(d[1]), size=None,
                                link=(gm.get("dk") or {}).get("url"), what=f"{yes_l} {d[0]}"))
        if yes_off and no_off:
            where = (f"{who['t']} {'vs' if who.get('h') else '@'} {who['o']}" if who.get("t") and who.get("o")
                     else f"{g['away']['abbr']} @ {g['home']['abbr']}")
            out.append((who["n"], f"{yes_l} · {where}", series, yes_off, no_off, "player"))
    return out


RISK = dict(
    player=["Kalshi settles a scratched player at a fair price and Polymarket settles him Under, so this only truly "
            "locks if he starts and bats.",
            "A sportsbook leg can be voided or cut to pennies; an exchange leg will not be."],
    game=["A game called off with no make-up pays 50-50 on Polymarket and a fair price on Kalshi, so a washout can "
          "leave a gap.",
          "A sportsbook leg can be voided or limited; an exchange leg will not be."],
)


def scan(sl: dict, cfg: dict = CFG, exchanges_only: bool = False) -> list[dict]:
    """Every market on the slate whose two sides together cost less than the dollar they pay."""
    games = {g["pk"]: g for g in sl["games"]}
    skip = {pk for pk, g in games.items() if g["state"] != "Preview" or (g.get("mk") or {}).get("pregame_as_of")
            or (g.get("mk") or {}).get("no_pregame")}
    jobs = []
    for g in sl["games"]:
        if g["pk"] not in skip: jobs += [(g, s) for s in game_sides(sl, g)]
    for who in sl.get("hitters", []) + sl.get("pitchers", []):
        g = games.get(who.get("pk"))
        if g and g["pk"] not in skip: jobs += [(g, s) for s in prop_sides(sl, who, g)]
    rows = []
    for g, (title, sub, kind, A, B, risk) in jobs:
        if exchanges_only:
            A = [o for o in A if o["venue"] != "book"]; B = [o for o in B if o["venue"] != "book"]
        p = best_pair(A, B, cfg)
        if not p or p["roi"] < cfg["min_roi"]: continue
        a, b = p["legs"]
        ex = all(l["venue"] != "book" for l in (a, b))
        rows.append(dict(p, kind=kind, pk=g["pk"], title=title, sub=sub,
                         risk=[t for t in RISK[risk] if not (ex and "sportsbook" in t)],
                         venues=" + ".join(sorted({a["site"], b["site"]})), exchange=ex,
                         gap=_r(1 - a["price"] - b["price"])))
    return sorted(rows, key=lambda r: -r["roi"])


# ------------------------------------------------------------------ the real book, not just its top
def fill(levels, n):
    """What n contracts actually cost walking a book of (price, size) levels, cheapest first."""
    left, paid = n, 0.0
    for price, size in levels:
        take = min(left, size)
        paid += take * price; left -= take
        if left <= 1e-9: return _r(paid, 6)
    return None


def _levels(leg, tok, O):
    if leg["venue"] == "ks" and leg.get("id"):
        return (O.kalshi_book(leg["id"]) or {}).get(leg["side"]) or []
    if leg["venue"] == "pm" and leg.get("id"):
        t = tok.get(leg["id"]) or []
        if len(t) > 1: return O.pm_book(t[0] if leg["side"] == "yes" else t[1]) or []
    return []


def deepen(rows, tok, cfg=CFG, log=print):
    """For the handful of markets that look arbitrable, read both order books and work out the most that could really
    go on at a profit. The top of a book is often a token size; this is the honest number."""
    from . import odds as O
    for r in rows:
        try:
            legs = r["legs"]
            books = [_levels(l, tok or {}, O) for l in legs]
            if not all(books): continue
            caps = [sum(s for _, s in b) for b in books]
            best = None
            for n in sorted({math.floor(c) for b in books for c in _cum(b)} | {math.floor(min(caps))}):
                if n < 1 or n > cfg["max_size"] or n > min(caps): continue
                if any(l["venue"] == "pm" for l in legs) and n < cfg["pm_min"]: continue
                paid = [fill(b, n) for b in books]
                if any(v is None for v in paid): continue
                fees = sum(leg_fee(l, n, cfg, price=paid[i] / n) for i, l in enumerate(legs))
                cost = sum(paid) + fees
                if n - cost <= 0: continue
                cand = dict(size=n, cost=_r(cost), profit=_r(n - cost), roi=(n - cost) / cost,
                            fees=_r(fees), avg=[_r(paid[i] / n) for i in range(2)])
                if best is None or cand["profit"] > best["profit"]: best = cand
            r["depth"] = best
        except Exception as e:
            log(f"arb depth {r.get('title')}: {e}")
    return rows


def _cum(levels):
    t = 0
    for _, s in levels:
        t += s; yield t


def describe(r) -> str:
    a, b = r["legs"]
    d = r.get("depth")
    size = d["size"] if d else r["size"]
    prof = d["profit"] if d else r["profit"]
    return (f"{r['title']} {r['sub']}: {a['site']} {a['what']} {a['price']*100:.0f}c + {b['site']} {b['what']} "
            f"{b['price']*100:.0f}c -> {100*(d['roi'] if d else r['roi']):.1f}% on ${size} for ${prof:.2f}")

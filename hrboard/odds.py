"""Market prices: DraftKings (through ESPN's public odds feed) and Kalshi (public market data API).

Both are read-only and need no key. ESPN sends CORS headers, so the page refreshes DraftKings itself; Kalshi does not,
so its prices are whatever the last build saw (the page says when that was).
"""
from __future__ import annotations
import re, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import numpy as np
import pandas as pd
import requests

from breakout.names import key as name_key

_S = requests.Session()   # default requests UA: ESPN's edge refuses a bare "Mozilla/5.0"
ESPN = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb"
CORE = "https://sports.core.api.espn.com/v2/sports/baseball/leagues/mlb"
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"

# ESPN / Kalshi club codes -> MLB Stats API abbreviations
ABBR = {"CHW": "CWS", "WAS": "WSH", "ARI": "AZ", "OAK": "ATH", "SA": "SD", "SFG": "SF", "SDP": "SD", "TBR": "TB",
        "KCR": "KC", "WSN": "WSH", "LA": "LAD", "ANA": "LAA", "LV": "ATH"}


def mlb_abbr(a: str) -> str:
    a = (a or "").upper(); return ABBR.get(a, a)


def _get(url, **params):
    for i in range(4):
        try:
            r = _S.get(url, params=params, timeout=45)
            if r.status_code == 429:
                time.sleep(2 + 3 * i); continue
            if r.status_code in (400, 404):
                return {}
            r.raise_for_status(); return r.json()
        except Exception:
            time.sleep(1 + i)
    return {}


def dk_link(link) -> str | None:
    """ESPN wraps DraftKings bet-slip links in a tracking gateway with unfilled placeholders; the real target is its
    `preurl` (the DraftKings event page, with the outcome pre-selected when there is one)."""
    from urllib.parse import urlparse, parse_qs
    href = (link or {}).get("href") if isinstance(link, dict) else link
    if not href: return None
    q = parse_qs(urlparse(href).query).get("preurl")
    return q[0] if q else (href if "__" not in href else None)


def american_to_prob(o) -> float:
    try: o = float(str(o).replace("+", ""))
    except (TypeError, ValueError): return np.nan
    if o == 0 or np.isnan(o): return np.nan
    return 100 / (o + 100) if o > 0 else -o / (-o + 100)


# ------------------------------------------------------------------ DraftKings via ESPN
def espn_games(day: str) -> pd.DataFrame:
    j = _get(f"{ESPN}/scoreboard", dates=day.replace("-", ""))
    rows = []
    for e in j.get("events", []):
        c = e["competitions"][0]
        teams = {t["homeAway"]: t for t in c["competitors"]}
        o = (c.get("odds") or [{}])[0]
        ml = o.get("moneyline") or {}; tot = o.get("total") or {}; ps = o.get("pointSpread") or {}
        hml = (ml.get("home") or {}).get("close", {}).get("odds"); aml = (ml.get("away") or {}).get("close", {}).get("odds")
        ph, pa = american_to_prob(hml), american_to_prob(aml)
        L = lambda side, part: dk_link(((((o.get(part) or {}).get(side) or {}).get("close") or {}).get("link")))
        rows.append(dict(dk_links=dict(event=dk_link(o.get("link")), home_ml=L("home", "moneyline"), away_ml=L("away", "moneyline"),
                                       over=L("over", "total"), under=L("under", "total"), home_rl=L("home", "pointSpread"),
                                       away_rl=L("away", "pointSpread")),
                         espn_id=e["id"], date=e["date"], home_abbr=mlb_abbr(teams["home"]["team"]["abbreviation"]),
                         away_abbr=mlb_abbr(teams["away"]["team"]["abbreviation"]), espn_home_team=teams["home"]["team"]["id"],
                         espn_away_team=teams["away"]["team"]["id"], dk_home_ml=hml, dk_away_ml=aml,
                         dk_home_p=(ph / (ph + pa) if ph == ph and pa == pa else np.nan),
                         dk_home_ml_open=(ml.get("home") or {}).get("open", {}).get("odds"),
                         dk_total=o.get("overUnder"), dk_over=(tot.get("over") or {}).get("close", {}).get("odds"),
                         dk_under=(tot.get("under") or {}).get("close", {}).get("odds"),
                         dk_home_rl=(ps.get("home") or {}).get("close", {}).get("line"),
                         dk_home_rl_odds=(ps.get("home") or {}).get("close", {}).get("odds"),
                         dk_away_rl_odds=(ps.get("away") or {}).get("close", {}).get("odds")))
    return pd.DataFrame(rows)


PROP_TYPES = {"Home Runs Milestones": "hr", "Hits Milestones": "hit", "Total Bases Milestones": "tb",
              "Strikeouts Thrown Milestones": "k"}


def espn_props(event_ids) -> pd.DataFrame:
    def one(eid):
        u = f"{CORE}/events/{eid}/competitions/{eid}/odds/100/propBets"
        j = _get(u, limit=1000); items = list(j.get("items", []))
        for p in range(2, (j.get("pageCount") or 1) + 1):
            items += _get(u, limit=1000, page=p).get("items", [])
        out = []
        for i in items:
            mk = PROP_TYPES.get((i.get("type") or {}).get("name"))
            if not mk: continue
            tgt = ((i.get("current") or {}).get("target") or {}).get("value")
            aid = re.search(r"/athletes/(\d+)", (i.get("athlete") or {}).get("$ref", ""))
            if tgt is None or not aid: continue
            am = ((i.get("odds") or {}).get("american") or {})
            out.append(dict(espn_id=str(eid), espn_athlete=aid.group(1), market=mk, line=float(tgt), dk_odds=am.get("value"),
                            dk_odds_open=am.get("open")))
        return out
    with ThreadPoolExecutor(6) as ex:
        rows = [r for part in ex.map(one, list(event_ids)) for r in part]
    d = pd.DataFrame(rows, columns=["espn_id", "espn_athlete", "market", "line", "dk_odds", "dk_odds_open"])
    if len(d):
        d["dk_p"] = d["dk_odds"].map(american_to_prob)
        d = d.drop_duplicates(["espn_athlete", "market", "line"], keep="last")
    return d


def espn_rosters() -> pd.DataFrame:
    teams = _get(f"{ESPN}/teams").get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", [])
    ids = [(t["team"]["id"], mlb_abbr(t["team"]["abbreviation"])) for t in teams]
    def one(t):
        tid, ab = t
        j = _get(f"{ESPN}/teams/{tid}/roster")
        return [dict(espn_athlete=str(p["id"]), espn_name=p.get("fullName") or p.get("displayName"), team_abbr=ab)
                for grp in j.get("athletes", []) for p in grp.get("items", [])]
    with ThreadPoolExecutor(6) as ex:
        rows = [r for part in ex.map(one, ids) for r in part]
    d = pd.DataFrame(rows)
    if len(d): d["nkey"] = d["espn_name"].map(name_key)
    return d


# ------------------------------------------------------------------ Kalshi
KALSHI_SERIES = {"KXMLBGAME": "game", "KXMLBTOTAL": "total", "KXMLBHR": "hr", "KXMLBHIT": "hit", "KXMLBTB": "tb", "KXMLBKS": "k"}
EVT = re.compile(r"-(\d{2})([A-Z]{3})(\d{2})(\d{4})([A-Z]+)$")
MON = {m: i for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}


def _dollars(x):
    try: return float(x)
    except (TypeError, ValueError): return np.nan


def kalshi_markets(day: str, series=tuple(KALSHI_SERIES)) -> pd.DataFrame:
    """Open markets for games on `day` (ET date), one row per market with bid / ask / last and a usable price."""
    ymd = datetime.fromisoformat(day)
    tag = f"{ymd:%y}{list(MON)[ymd.month - 1]}{ymd:%d}"
    rows = []
    for s in series:
        for m in _kalshi_series(s):
            if True:
                ev = m.get("event_ticker", "")
                if f"-{tag}" not in ev: continue
                mm = EVT.search(ev)
                bid, ask, last = _dollars(m.get("yes_bid_dollars")), _dollars(m.get("yes_ask_dollars")), _dollars(m.get("last_price_dollars"))
                if bid == bid and ask == ask and ask > 0 and ask - bid <= 0.10:
                    price = (bid + ask) / 2
                elif last == last and last > 0:
                    price = last
                else:
                    price = np.nan
                rows.append(dict(series=KALSHI_SERIES[s], series_ticker=s, url=kalshi_url(s, ev), ticker=m["ticker"], event=ev, title=m.get("title"), sub=m.get("yes_sub_title"),
                                 et_time=(mm.group(4) if mm else None), teams=(mm.group(5) if mm else None),
                                 bid=bid, ask=ask, last=last, price=price, volume=_dollars(m.get("volume_fp")),
                                 bid_sz=_dollars(m.get("yes_bid_size_fp")), ask_sz=_dollars(m.get("yes_ask_size_fp")),
                                 oi=_dollars(m.get("open_interest_fp"))))
    return pd.DataFrame(rows)


_KCACHE: dict = {}


def kalshi_url(series: str, event: str) -> str:
    """Kalshi's own event pages are /markets/<series>/<slug>/<event>; '-' works as the slug."""
    return f"https://kalshi.com/markets/{series.lower()}/-/{event.lower()}"


def kalshi_fees(series=tuple(KALSHI_SERIES)) -> dict:
    """fee_multiplier per series (taker fee = round up to the cent of 0.07 x multiplier x contracts x P x (1 - P))."""
    out = {}
    for s in series:
        j = _get(f"{KALSHI}/series/{s}")
        out[KALSHI_SERIES[s]] = float((j.get("series") or {}).get("fee_multiplier") or 1.0)
    return out


def kalshi_fee(p: float, mult: float, contracts: int = 1) -> float:
    import math as _m
    if p is None or not (0 < p < 1): return 0.0
    return _m.ceil(round(100 * 0.07 * mult * contracts * p * (1 - p), 6)) / 100


def kalshi_prices(tickers) -> dict:
    """Current bid / ask / last and the size resting at each for a list of market tickers (100 per request)."""
    tickers = list(dict.fromkeys(t for t in tickers if t))
    out = {}
    for i in range(0, len(tickers), 100):
        j = _get(f"{KALSHI}/markets", tickers=",".join(tickers[i:i + 100]), limit=100)
        for m in j.get("markets", []):
            out[m["ticker"]] = [_dollars(m.get("yes_bid_dollars")), _dollars(m.get("yes_ask_dollars")), _dollars(m.get("last_price_dollars")),
                                _dollars(m.get("yes_bid_size_fp")), _dollars(m.get("yes_ask_size_fp"))]
        time.sleep(0.2)
    return out


def kalshi_book(ticker: str) -> dict:
    """What it costs to buy each side, level by level. Kalshi keeps two books of resting bids: to buy Yes you match the
    people bidding No, so a No bid at 57c is a Yes contract at 43c."""
    j = _get(f"{KALSHI}/markets/{ticker}/orderbook")
    ob = j.get("orderbook_fp") or j.get("orderbook") or {}
    out = {}
    for side, other in (("yes", "no_dollars"), ("no", "yes_dollars")):
        lv = [(round(1 - float(p), 4), float(q)) for p, q in (ob.get(other) or []) if float(q) > 0]
        out[side] = sorted(lv)          # cheapest first
    return out


def pm_book(token_id: str) -> list:
    """Polymarket ask levels for one outcome token: [(cost, shares)] cheapest first."""
    j = _get(f"{CLOB}/book", token_id=token_id)
    lv = [(float(a["price"]), float(a["size"])) for a in (j.get("asks") or []) if float(a.get("size", 0)) > 0]
    return sorted(lv)


def _kalshi_series(s: str) -> list[dict]:
    """Every open market in a series (paged), fetched once per process."""
    if s in _KCACHE: return _KCACHE[s]
    out, cursor = [], None
    for _ in range(40):
        params = dict(series_ticker=s, status="open", limit=1000)
        if cursor: params["cursor"] = cursor
        j = _get(f"{KALSHI}/markets", **params)
        out += j.get("markets", [])
        cursor = j.get("cursor")
        if not cursor or not j.get("markets"): break
        time.sleep(0.25)
    _KCACHE[s] = out
    return out


PLAYER_T = re.compile(r"^(.*?):\s*(\d+)\+")


def kalshi_split(km: pd.DataFrame, games: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match Kalshi events to today's games (by the away+home code in the ticker) and split into game-level and
    player-level rows. games needs game_pk, away_abbr, home_abbr."""
    if not len(km): return pd.DataFrame(), pd.DataFrame()
    codes = {}
    for g in games.itertuples():
        for a in {g.away_abbr, _k(g.away_abbr)}:
            for h in {g.home_abbr, _k(g.home_abbr)}:
                codes.setdefault(a + h, []).append(g.game_pk)
    km = km.copy()
    km["game_pk"] = km["teams"].map(lambda t: codes.get(t, [None])[0] if t else None)
    # doubleheaders share a code; the ticker's start time picks the game
    km = km.dropna(subset=["game_pk"])
    game = km[km["series"].isin(["game", "total"])].copy()
    ply = km[~km["series"].isin(["game", "total"])].copy()
    if len(game):
        game["team"] = game["ticker"].str.rsplit("-", n=1).str[-1].map(mlb_abbr)
        game["line"] = np.where(game["series"] == "total", game["title"].str.extract(r"Over ([\d.]+)")[0].astype(float), np.nan)
    if not len(ply):
        return game, pd.DataFrame(columns=list(ply.columns) + ["kname", "line", "nkey"])
    m = ply["title"].astype(str).str.extract(PLAYER_T)
    ply["kname"] = m[0].str.strip(); ply["line"] = pd.to_numeric(m[1], errors="coerce")
    ply["nkey"] = ply["kname"].map(lambda v: name_key(v) if isinstance(v, str) else "")
    return game, ply


KAL = {"AZ": "ARI", "CWS": "CWS", "WSH": "WSH", "ATH": "ATH"}


def _k(a):
    return KAL.get(a, a)


# ------------------------------------------------------------------ Polymarket (public, CORS-open, so the page also reads it live)
GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
PM_CODE = {"AZ": ["ari", "az"], "ATH": ["oak", "ath"], "CWS": ["cws", "chw"], "WSH": ["wsh", "was"], "KC": ["kc"], "SD": ["sd"],
           "SF": ["sf"], "TB": ["tb"], "LAA": ["laa"], "LAD": ["lad"]}
PM_PROP = {"baseball_player_home_runs": "hr", "baseball_player_hits": "hit", "baseball_player_total_bases": "tb",
           "baseball_player_strikeouts": "k"}


def _pm_event(slug):
    j = _get(f"{GAMMA}/events", slug=slug)
    return j[0] if isinstance(j, list) and j else None


def _jl(x):
    import json as _j
    try: return _j.loads(x) if isinstance(x, str) else (x or [])
    except ValueError: return []


def pm_side_prices(m) -> list:
    """[price, ask, bid] for each outcome of a two-outcome market. Gamma's bestBid/bestAsk describe the first outcome;
    the second outcome's ask is one minus the first outcome's bid (and its bid one minus the first's ask)."""
    px = [float(v) for v in _jl(m.get("outcomePrices"))] + [None, None]
    f = lambda v: float(v) if v not in (None, "") else None
    b0, a0 = f(m.get("bestBid")), f(m.get("bestAsk"))
    return [[px[0], a0, b0], [px[1], round(1 - b0, 4) if b0 is not None else None, round(1 - a0, 4) if a0 is not None else None]]


def polymarket_game(day: str, away: str, home: str) -> dict | None:
    """Moneyline, totals and run line from the game event, and player props from its '-player-props' twin."""
    ev = slug = None
    for a in PM_CODE.get(away, [away.lower()]):
        for h in PM_CODE.get(home, [home.lower()]):
            cand = f"mlb-{a}-{h}-{day}"
            ev = _pm_event(cand)
            if ev: slug = cand; break
        if ev: break
    if not ev: return None
    out = dict(slug=slug, url=f"https://polymarket.com/event/{slug}", ml={}, totals={}, rl={}, props={}, props_slug=None, tok={})
    for m in ev.get("markets", []):
        t = m.get("sportsMarketType"); sp = pm_side_prices(m)
        if m.get("closed"): continue
        if t == "moneyline":
            out["ml"] = dict(names=_jl(m.get("outcomes")), prices=sp, slug=m.get("slug"))
            out["tok"][m.get("slug")] = _jl(m.get("clobTokenIds"))
        elif t == "totals" and m.get("line") is not None:
            out["totals"][f"{float(m['line']):g}"] = dict(over=sp[0], under=sp[1], slug=m.get("slug"))
            out["tok"][m.get("slug")] = _jl(m.get("clobTokenIds"))
        elif t == "spreads" and m.get("line") is not None:
            out["rl"][m.get("slug")] = dict(line=float(m["line"]), first=_jl(m.get("outcomes"))[:1], prices=sp)
    pe = _pm_event(f"{slug}-player-props")
    if pe:
        out["props_slug"] = f"{slug}-player-props"; out["props_url"] = f"https://polymarket.com/event/{slug}-player-props"
        for m in pe.get("markets", []):
            k = PM_PROP.get(m.get("sportsMarketType"))
            if not k or m.get("closed") or m.get("line") is None: continue
            name = str(m.get("question", "")).split(":")[0].strip()
            n = int(float(m["line"]) + 0.5)
            out["props"].setdefault(name, {})[f"{k}{n}"] = pm_side_prices(m)[0] + [m.get("slug")]
            out["tok"][m.get("slug")] = _jl(m.get("clobTokenIds"))
    return out


def polymarket_games(day: str, games: pd.DataFrame) -> dict:
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(lambda g: (g.game_pk, polymarket_game(day, g.away_abbr, g.home_abbr)), list(games.itertuples())))
    return {int(pk): v for pk, v in res if v}


# ------------------------------------------------------------------ The Odds API (key in the ODDS_API_KEY secret; build-time only)
ODDS_API = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
BOOKS = ["fanduel", "draftkings", "betmgm", "williamhill_us", "fanatics", "betrivers", "espnbet", "hardrockbet", "novig", "prophetx"]
BOOK_NAMES = {"fanduel": "FanDuel", "draftkings": "DraftKings", "betmgm": "BetMGM", "williamhill_us": "Caesars", "fanatics": "Fanatics",
              "betrivers": "BetRivers", "espnbet": "theScore Bet", "hardrockbet": "Hard Rock", "novig": "Novig", "prophetx": "ProphetX"}
TEAM_FULL = {}   # filled by the caller from the MLB schedule: "Seattle Mariners" -> "SEA"


def odds_api_lines(key: str, markets=("h2h", "spreads", "totals")) -> tuple[list, dict]:
    """One request for every upcoming game: ten books (counts as one region) x three markets = 3 credits."""
    r = _S.get(ODDS_API, params=dict(apiKey=key, bookmakers=",".join(BOOKS), markets=",".join(markets), oddsFormat="american",
                                     dateFormat="iso", includeLinks="true"), timeout=45)
    usage = dict(remaining=r.headers.get("x-requests-remaining"), used=r.headers.get("x-requests-used"), last=r.headers.get("x-requests-last"),
                 status=r.status_code)
    if not r.ok:
        return [], usage
    return r.json(), usage


def odds_api_by_game(events: list, games: pd.DataFrame) -> dict:
    """Match The Odds API events to MLB games by team names and start time; shape each book's prices."""
    out = {}
    g = games.copy(); g["t"] = pd.to_datetime(g["time"], utc=True)
    for ev in events:
        h, a = TEAM_FULL.get(ev.get("home_team")), TEAM_FULL.get(ev.get("away_team"))
        cand = g[(g["home_abbr"] == h) & (g["away_abbr"] == a)]
        if not len(cand): continue
        t = pd.Timestamp(ev["commence_time"])
        cand = cand.assign(dt=(cand["t"] - t).abs()).sort_values("dt")
        if cand["dt"].iat[0] > pd.Timedelta(hours=6): continue
        pk = int(cand["game_pk"].iat[0])
        books = []
        for b in ev.get("bookmakers", []):
            row = dict(key=b["key"], name=BOOK_NAMES.get(b["key"], b.get("title")), link=b.get("link"), updated=b.get("last_update"))
            for mk in b.get("markets", []):
                oc = mk.get("outcomes", [])
                if mk["key"] == "h2h":
                    for o in oc:
                        side = "home" if o["name"] == ev["home_team"] else "away"
                        row.setdefault("ml", {})[side] = [o.get("price"), o.get("link") or mk.get("link")]
                elif mk["key"] == "totals":
                    for o in oc:
                        row.setdefault("tot", {})[o["name"].lower()] = [o.get("point"), o.get("price"), o.get("link") or mk.get("link")]
                elif mk["key"] == "spreads":
                    for o in oc:
                        side = "home" if o["name"] == ev["home_team"] else "away"
                        row.setdefault("rl", {})[side] = [o.get("point"), o.get("price"), o.get("link") or mk.get("link")]
            books.append(row)
        out[pk] = books
    return out

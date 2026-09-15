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
        rows.append(dict(espn_id=e["id"], date=e["date"], home_abbr=mlb_abbr(teams["home"]["team"]["abbreviation"]),
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
                rows.append(dict(series=KALSHI_SERIES[s], ticker=m["ticker"], event=ev, title=m.get("title"), sub=m.get("yes_sub_title"),
                                 et_time=(mm.group(4) if mm else None), teams=(mm.group(5) if mm else None),
                                 bid=bid, ask=ask, last=last, price=price, volume=_dollars(m.get("volume_fp")),
                                 oi=_dollars(m.get("open_interest_fp"))))
    return pd.DataFrame(rows)


_KCACHE: dict = {}


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

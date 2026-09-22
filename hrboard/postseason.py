"""NFBC Postseason Holdem board: who plays how many games in October, what each player is worth in the contest's
points with the round multipliers, and the roster that maximises it.   python -m hrboard.postseason

The contest (nfc.shgn.com/rules/2618): 16 spots (C 1B 2B 3B SS OF OF OF OF UT and six pitchers), picked from the eight
Division Series clubs with at most 3 and at least 1 from each. Between rounds you may only replace players whose club
was eliminated. A player kept from the Division Series scores double in the Championship Series and triple in the World
Series; one added in the LCS scores double in the WS; one added for the WS scores single. LCS roster: 2 to 6 per club.
WS: 6 to 10 per club. Scoring: 1B 1, 2B 2, 3B 3, HR 4, R 1, RBI 1, SB 1, BB 1, HBP 1, out (AB - H) -0.25;
IP 1, K 1, W 4, SV 4, ER -1.

Where the numbers come from
  team strength   the exchanges' World Series and pennant prices (Kalshi and Polymarket, margins removed), turned into
                  a per-game rating for each club so that a full bracket simulation reproduces them; from that, the
                  chance each club plays each round and how many games it plays there
  hitters         this season's rate of each scoring event per plate appearance, shrunk toward the league, times the
                  plate appearances a lineup slot gets in a playoff game, times 0.86 (what playoff pitching did to
                  regular-season rates in 2024-25), times the club's expected games
  starters        rotation order from September usage; expected starts per series from where a slot falls in a best-of-5
                  or best-of-7 (game 1 and 5, game 2 and 6, ...); playoff length 0.8 of the season's innings per start
                  (4.4 vs 5.6 in 2024-25), season strikeout rate, earned runs 1.25x the season rate, a win in 51% of
                  the games the club wins (2022-25 postseason)
  relievers       the closer gets the save in 48% of his club's wins (2022-25 postseason) plus an inning; setup men are
                  an inning and a strikeout in half the games
"""
from __future__ import annotations
import argparse, json, math, time
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from breakout.config import DATA
from . import odds as O, mlbstats as MS

SITE = Path(__file__).resolve().parents[1] / "site"
HOME_LOGIT = 0.16                    # ~54% for the home side at equal strength
HIT_DEFLATE = 0.86                    # playoff pitching vs regular-season rates, 2024-25 postseasons
SP_IP_SCALE, SP_ER_SCALE, SP_WIN_SHARE = 0.80, 1.25, 0.51
CLOSER_SAVE_SHARE = 0.48              # saves per club win, 2022-25 postseason
PA_BY_SLOT = {1: 4.45, 2: 4.25, 3: 4.31, 4: 4.21, 5: 3.97, 6: 3.81, 7: 3.47, 8: 3.07, 9: 3.05}
ROUNDS = ("DS", "LCS", "WS")
MULT = dict(held=(1, 2, 3), add_lcs=(0, 1, 2), add_ws=(0, 0, 1))
LIMITS = dict(DS=(1, 3), LCS=(2, 6), WS=(6, 10))
SLOTS = ["C", "1B", "2B", "3B", "SS", "OF", "OF", "OF", "OF", "UT", "P", "P", "P", "P", "P", "P"]


def log(*a):
    print(datetime.now().strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------ the field
def standings(season: int) -> pd.DataFrame:
    j = MS._get(f"{MS.API}/standings?leagueId=103,104&season={season}&standingsTypes=regularSeason&hydrate=team")
    rows = []
    for rec in j.get("records", []):
        lg = rec.get("league", {}).get("id")
        for t in rec["teamRecords"]:
            rows.append(dict(id=t["team"]["id"], abbr=t["team"].get("abbreviation"), name=t["team"].get("name"), league="AL" if lg == 103 else "NL",
                             w=t["wins"], l=t["losses"], pct=float(t.get("winningPercentage") or 0), div_rank=int(t.get("divisionRank") or 9),
                             wc_rank=int(t.get("wildCardRank") or 9), clinch=t.get("clinchIndicator") or "",
                             elim=t.get("eliminationNumber"), wc_elim=t.get("wildCardEliminationNumber"),
                             gp=t.get("gamesPlayed") or (t["wins"] + t["losses"])))
    return pd.DataFrame(rows)


def field(season: int) -> tuple[pd.DataFrame, bool]:
    """Six seeds per league: division leaders by record, then the wild cards. `settled` is True once every spot is
    clinched; until then the field is the standings as they sit."""
    st = standings(season)
    out = []
    for lg, g in st.groupby("league"):
        div = g[g["div_rank"] == 1].sort_values(["pct", "w"], ascending=False)
        wc = g[g["div_rank"] != 1].sort_values(["wc_rank", "pct"], ascending=[True, False]).head(3)
        for i, r in enumerate(list(div.itertuples()) + list(wc.itertuples()), 1):
            out.append(dict(seed=i, league=lg, id=r.id, abbr=r.abbr, name=r.name, w=r.w, l=r.l, clinch=r.clinch, bye=i <= 2))
    f = pd.DataFrame(out)
    settled = bool((f["clinch"].isin(["x", "y", "z", "w"])).all()) and st[~st["id"].isin(f["id"])]["elim"].fillna("E").astype(str).eq("E").all() \
        and st[~st["id"].isin(f["id"])]["wc_elim"].fillna("E").astype(str).eq("E").all()
    return f, settled


def playoff_schedule(season: int) -> pd.DataFrame:
    """Postseason games already scheduled or played (once the bracket is set, this is the truth)."""
    j = MS._get(f"{MS.API}/schedule?sportId=1&season={season}&gameTypes=F,D,L,W&hydrate=team")
    rows = []
    for d in j.get("dates", []):
        for g in d["games"]:
            h, a = g["teams"]["home"], g["teams"]["away"]
            rows.append(dict(pk=g["gamePk"], gt=g["gameType"], date=g["officialDate"], home=h["team"].get("abbreviation"), away=a["team"].get("abbreviation"),
                             home_id=h["team"]["id"], away_id=a["team"]["id"], gno=g.get("seriesGameNumber"), state=g["status"]["abstractGameState"],
                             home_won=h.get("isWinner"), away_won=a.get("isWinner"), hs=h.get("score"), as_=a.get("score")))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ what the markets think
NAME_TO_ABBR = {"Los Angeles D": "LAD", "Los Angeles A": "LAA", "New York Y": "NYY", "New York M": "NYM", "Chicago C": "CHC", "Chicago WS": "CWS",
                "Chicago W": "CWS", "Tampa Bay": "TB", "San Diego": "SD", "San Francisco": "SF", "St. Louis": "STL", "Kansas City": "KC",
                "Milwaukee": "MIL", "Atlanta": "ATL", "Boston": "BOS", "Philadelphia": "PHI", "Cleveland": "CLE", "Texas": "TEX", "Houston": "HOU",
                "Toronto": "TOR", "Seattle": "SEA", "Baltimore": "BAL", "Arizona": "AZ", "Detroit": "DET", "Minnesota": "MIN", "Cincinnati": "CIN",
                "Pittsburgh": "PIT", "Miami": "MIA", "Washington": "WSH", "Colorado": "COL", "A's": "ATH", "Athletics": "ATH"}
FULL_TO_ABBR = {"Los Angeles Dodgers": "LAD", "Milwaukee Brewers": "MIL", "New York Yankees": "NYY", "Tampa Bay Rays": "TB", "Atlanta Braves": "ATL",
                "San Diego Padres": "SD", "Boston Red Sox": "BOS", "Chicago Cubs": "CHC", "Philadelphia Phillies": "PHI", "Cleveland Guardians": "CLE",
                "Texas Rangers": "TEX", "Chicago White Sox": "CWS", "Houston Astros": "HOU", "Toronto Blue Jays": "TOR", "Arizona Diamondbacks": "AZ",
                "Seattle Mariners": "SEA", "Baltimore Orioles": "BAL", "Detroit Tigers": "DET", "Minnesota Twins": "MIN", "Kansas City Royals": "KC",
                "New York Mets": "NYM", "Miami Marlins": "MIA", "Washington Nationals": "WSH", "St. Louis Cardinals": "STL", "Cincinnati Reds": "CIN",
                "Pittsburgh Pirates": "PIT", "San Francisco Giants": "SF", "Colorado Rockies": "COL", "Los Angeles Angels": "LAA", "Athletics": "ATH"}


def _kalshi_book(series: str) -> dict:
    out = {}
    j = O._get(f"{O.KALSHI}/markets", series_ticker=series, status="open", limit=200)
    for m in j.get("markets", []):
        ab = NAME_TO_ABBR.get(m.get("yes_sub_title") or "", None)
        if not ab: continue
        bid, ask = O._dollars(m.get("yes_bid_dollars")), O._dollars(m.get("yes_ask_dollars"))
        if bid == bid and ask == ask and ask > 0: out[ab] = dict(mid=(bid + ask) / 2, bid=bid, ask=ask, ticker=m["ticker"])
    return out


def _poly_book(slug: str) -> dict:
    out = {}
    ev = O._get(f"{O.GAMMA}/events", slug=slug)
    if not ev: return out
    for m in ev[0].get("markets", []):
        if m.get("closed"): continue
        ab = FULL_TO_ABBR.get(m.get("groupItemTitle") or "")
        px = O._jl(m.get("outcomePrices"))
        if ab and px: out[ab] = dict(mid=float(px[0]), bid=O._dollars(m.get("bestBid")), ask=O._dollars(m.get("bestAsk")))
    return out


def market_odds(season: int) -> dict:
    """De-vigged chance to win the World Series and each pennant, per club, averaged across the exchanges."""
    yy = str(season)[2:]
    books = dict(ws=[_kalshi_book(f"KXMLB"), _poly_book(f"mlb-world-series-champion-{season}")],
                 AL=[_kalshi_book("KXMLBAL"), _poly_book(f"mlb-{season}-american-league-champion")],
                 NL=[_kalshi_book("KXMLBNL"), _poly_book(f"mlb-{season}-national-league-champion")])
    out = {}
    for k, bs in books.items():
        acc = {}
        for b in bs:
            tot = sum(v["mid"] for v in b.values())
            if tot <= 0: continue
            for ab, v in b.items(): acc.setdefault(ab, []).append(v["mid"] / tot)       # one book's margin out
        out[k] = {ab: float(np.mean(v)) for ab, v in acc.items()}
    out["sources"] = dict(kalshi=bool(books["ws"][0]), polymarket=bool(books["ws"][1]))
    out["raw"] = {k: [{ab: round(v["mid"], 4) for ab, v in b.items()} for b in bs] for k, bs in books.items()}
    return out


# ------------------------------------------------------------------ series arithmetic
def _series(p_home, p_away, best_of: int, pattern: str):
    """Distribution over outcomes of a series for the side that owns home field. Returns (P(win), P(reaches game g)
    for g = 1..N, P(length = k)) with the home side's per-game chances p_home at home and p_away away."""
    need = best_of // 2 + 1
    states = {(0, 0): 1.0}                          # (wins for the home side, wins for the other)
    reach = np.zeros(best_of + 1); length = np.zeros(best_of + 1); p_win = 0.0
    for g in range(1, best_of + 1):
        home = pattern[g - 1] == "H"
        p = p_home if home else p_away
        nxt = {}
        for (a, b), pr in states.items():
            reach[g] += pr
            for won, q in ((True, p), (False, 1 - p)):
                a2, b2 = a + won, b + (not won)
                if a2 == need: p_win += pr * q; length[g] += pr * q
                elif b2 == need: length[g] += pr * q
                else: nxt[(a2, b2)] = nxt.get((a2, b2), 0.0) + pr * q
        states = nxt
    return p_win, reach[1:], length[1:]


PATTERN = {3: "HHH", 5: "HHAAH", 7: "HHAAAHH"}


def game_p(s_a, s_b, home_a: bool):
    return 1 / (1 + math.exp(-(s_a - s_b + (HOME_LOGIT if home_a else -HOME_LOGIT))))


def series_p(s_home, s_away, best_of):
    return _series(game_p(s_home, s_away, True), game_p(s_home, s_away, False), best_of, PATTERN[best_of])


# ------------------------------------------------------------------ the bracket
class Bracket:
    """Six seeds a league. 3 v 6 and 4 v 5 in a best-of-3 at the higher seed; 1 v (4/5) and 2 v (3/6) in a best-of-5;
    a best-of-7 LCS; a best-of-7 World Series with home field to the better record."""

    def __init__(self, f: pd.DataFrame, strength: dict, played: pd.DataFrame | None = None, records: dict | None = None):
        self.f = f; self.s = strength; self.rec = records or {}
        self.seed = {(r.league, r.seed): r.abbr for r in f.itertuples()}
        self.league = {r.abbr: r.league for r in f.itertuples()}
        self.seedno = {r.abbr: r.seed for r in f.itertuples()}
        self.played = played if played is not None and len(played) else pd.DataFrame()

    def _known(self, gt, a, b):
        """Series already under way: (wins a, wins b) so far, else (0, 0)."""
        if not len(self.played): return 0, 0
        p = self.played[(self.played["gt"] == gt) & (self.played["state"] == "Final")]
        p = p[((p["home"] == a) & (p["away"] == b)) | ((p["home"] == b) & (p["away"] == a))]
        wa = int(((p["home"] == a) & (p["home_won"] == True)).sum() + ((p["away"] == a) & (p["away_won"] == True)).sum())
        wb = int(len(p) - wa)
        return wa, wb

    def _series_from(self, gt, hi, lo, best_of):
        """Given the games already played: P(hi wins), expected remaining games, P(each remaining game number is
        played) as a vector over game numbers 1..N, and expected wins for hi and lo over the remaining games."""
        need = best_of // 2 + 1
        wa, wb = self._known(gt, hi, lo)
        reach = np.zeros(best_of); wins = np.zeros(2)
        if wa >= need or wb >= need: return (1.0 if wa >= need else 0.0), 0.0, reach, wins
        ph, pa = game_p(self.s[hi], self.s[lo], True), game_p(self.s[hi], self.s[lo], False)
        pat = PATTERN[best_of][wa + wb:]
        states = {(wa, wb): 1.0}; p_win = 0.0; exp_games = 0.0
        for g, side in enumerate(pat, wa + wb + 1):
            p = ph if side == "H" else pa
            nxt = {}
            for (a, b), pr in states.items():
                exp_games += pr; reach[g - 1] += pr; wins += pr * np.array([p, 1 - p])
                for won, q in ((True, p), (False, 1 - p)):
                    a2, b2 = a + won, b + (not won)
                    if a2 == need: p_win += pr * q
                    elif b2 == need: pass
                    else: nxt[(a2, b2)] = nxt.get((a2, b2), 0.0) + pr * q
            states = nxt
        return p_win, exp_games, reach, wins

    def run(self) -> dict:
        """Per club: chance to play each round, expected games in it, chance of each game number being played, expected
        wins, chance of the pennant and the title."""
        f = self.f
        teams = list(f["abbr"])
        P = {t: dict(WC=0.0, DS=0.0, LCS=0.0, WS=0.0, pennant=0.0, title=0.0, gWC=0.0, gDS=0.0, gLCS=0.0, gWS=0.0,
                     rDS=np.zeros(5), rLCS=np.zeros(7), rWS=np.zeros(7), wDS=0.0, wLCS=0.0, wWS=0.0, opp={}) for t in teams}
        pennant = {}

        def add(t, rnd, pr, reach, w, opp):
            P[t][rnd] += pr; P[t]["g" + rnd] += pr * reach.sum(); P[t]["r" + rnd] = P[t]["r" + rnd] + pr * reach
            P[t]["w" + rnd] += pr * w; P[t]["opp"].setdefault(rnd, {}); P[t]["opp"][rnd][opp] = P[t]["opp"][rnd].get(opp, 0.0) + pr

        for lg in ("AL", "NL"):
            S = {i: self.seed[(lg, i)] for i in range(1, 7)}
            wc = {}
            for hi, lo in ((3, 6), (4, 5)):
                a, b = S[hi], S[lo]
                p, _, reach, w = self._series_from("F", a, b, 3)
                wc[(hi, lo)] = (a, b, p)
                P[a]["WC"] = P[b]["WC"] = 1.0; P[a]["gWC"] = P[b]["gWC"] = reach.sum()
            lcs_entrants = {}
            for top, pair in ((1, (4, 5)), (2, (3, 6))):
                a, b, pab = wc[pair]
                for opp, popp in ((a, pab), (b, 1 - pab)):
                    hi = S[top]
                    pw, _, reach, w = self._series_from("D", hi, opp, 5)
                    add(hi, "DS", popp, reach, w[0], opp); add(opp, "DS", popp, reach, w[1], hi)
                    lcs_entrants[(top, hi)] = lcs_entrants.get((top, hi), 0.0) + popp * pw
                    lcs_entrants[(pair, opp)] = lcs_entrants.get((pair, opp), 0.0) + popp * (1 - pw)
            side1 = {k: v for k, v in lcs_entrants.items() if k[0] == 1 or k[0] == (4, 5)}
            side2 = {k: v for k, v in lcs_entrants.items() if k[0] == 2 or k[0] == (3, 6)}
            for (k1, t1), p1 in side1.items():
                for (k2, t2), p2 in side2.items():
                    pr = p1 * p2
                    if pr <= 0: continue
                    hi, lo = (t1, t2) if self.seedno[t1] < self.seedno[t2] else (t2, t1)
                    pw, _, reach, w = self._series_from("L", hi, lo, 7)
                    add(hi, "LCS", pr, reach, w[0], lo); add(lo, "LCS", pr, reach, w[1], hi)
                    pennant[hi] = pennant.get(hi, 0.0) + pr * pw
                    pennant[lo] = pennant.get(lo, 0.0) + pr * (1 - pw)
        for t, p in pennant.items(): P[t]["pennant"] = p
        al = {t: p for t, p in pennant.items() if self.league[t] == "AL"}
        nl = {t: p for t, p in pennant.items() if self.league[t] == "NL"}
        for ta, pa in al.items():
            for tn, pn in nl.items():
                pr = pa * pn
                if pr <= 0: continue
                hi, lo = (ta, tn) if self.rec.get(ta, 0) >= self.rec.get(tn, 0) else (tn, ta)
                pw, _, reach, w = self._series_from("W", hi, lo, 7)
                add(hi, "WS", pr, reach, w[0], lo); add(lo, "WS", pr, reach, w[1], hi)
                P[hi]["title"] += pr * pw; P[lo]["title"] += pr * (1 - pw)
        for t in teams:
            for r in ("DS", "LCS", "WS"):
                if P[t][r] > 0:
                    P[t]["g" + r] /= P[t][r]; P[t]["r" + r] = P[t]["r" + r] / P[t][r]; P[t]["w" + r] /= P[t][r]
                    P[t]["opp"][r] = {k: v / P[t][r] for k, v in P[t]["opp"].get(r, {}).items()}
                P[t]["r" + r] = [round(float(v), 4) for v in P[t]["r" + r]]
        return P


def fit_strength(f: pd.DataFrame, mk: dict, played=None, records=None) -> dict:
    """Per-game ratings that make the bracket reproduce the market's title and pennant chances."""
    teams = list(f["abbr"])
    target = []
    for t in teams:
        if t in mk.get("ws", {}): target.append(("title", t, mk["ws"][t]))
        lg = f.loc[f["abbr"] == t, "league"].iat[0]
        if t in mk.get(lg, {}): target.append(("pennant", t, mk[lg][t]))
    x0 = np.zeros(len(teams))

    def loss(x):
        s = dict(zip(teams, x - x.mean()))
        P = Bracket(f, s, played, records).run()
        err = 0.0
        for kind, t, v in target:
            m = P[t][kind]
            err += (math.log((m + 1e-4) / (1 - m + 1e-4)) - math.log((v + 1e-4) / (1 - v + 1e-4))) ** 2 * (0.5 + v)
        return err + 0.01 * float((x ** 2).sum())
    r = minimize(loss, x0, method="L-BFGS-B", options=dict(maxiter=300))
    return dict(zip(teams, (r.x - r.x.mean()).round(4)))


# ------------------------------------------------------------------ the players
LG_K_PA = 200            # shrink a hitter's per-PA rates toward the league with this many phantom plate appearances
POS_SLOT = {"C": "C", "1B": "1B", "2B": "2B", "3B": "3B", "SS": "SS", "LF": "OF", "CF": "OF", "RF": "OF", "OF": "OF", "DH": "UT"}
def elig_slots(pos) -> list:
    """Slot types a player can fill: his positions plus UT for any hitter; pitchers only P."""
    pos = [pos] if isinstance(pos, str) else list(pos)
    if pos == ["P"]: return ["P"]
    return sorted({p for p in pos if p in ("C", "1B", "2B", "3B", "SS", "OF")} | {"UT"})
SP_GAME = {5: {1: [1, 5], 2: [2], 3: [3], 4: [4]}, 7: {1: [1, 5], 2: [2, 6], 3: [3, 7], 4: [4]}}     # which games each rotation slot starts


def _rates(h: pd.DataFrame, lg: pd.Series) -> pd.DataFrame:
    """Per-PA rate of each scoring event, shrunk toward the league."""
    out = h.copy()
    out["X1"] = out["H"] - out["X2"] - out["X3"] - out["HR"]; out["OUT"] = out["AB"] - out["H"]
    for c in ("X1", "X2", "X3", "HR", "R", "RBI", "SB", "BB", "HBP", "OUT"):
        out[f"r_{c}"] = (out[c] + LG_K_PA * lg[c]) / (out["PA"] + LG_K_PA)
    out["ppa"] = sum(MS.HOLDEM_H[c] * out[f"r_{c}"] for c in MS.HOLDEM_H)
    return out


IL_AVAIL = {"D10": 0.55, "D15": 0.45, "D7": 0.6, "D60": 0.0, "BRV": 0.9, "PL": 0.8, "SUS": 0.5}


def roster40(f: pd.DataFrame, season: int) -> pd.DataFrame:
    """The 40-man of every playoff club with each man's status, plus the injury text and date for anyone on the IL.
    A player on the 10- or 15-day list in late September may well be back for the Division Series; he is carried
    with a reduced chance and flagged, so the page can count him in or out."""
    from breakout import injuries as INJ
    rows = []
    for t in f.itertuples():
        j = MS._get(f"{MS.API}/teams/{t.id}/roster?rosterType=40Man&hydrate=person")
        for r in j.get("roster", []):
            st = (r.get("status") or {}).get("code") or "A"
            rows.append(dict(team_id=t.id, pid=r["person"]["id"], name=r["person"]["fullName"], pos=r["position"]["abbreviation"],
                             ptype=r["position"]["type"], bats=(r["person"].get("batSide") or {}).get("code", "R"),
                             throws=(r["person"].get("pitchHand") or {}).get("code", "R"), status=st))
    ros = pd.DataFrame(rows)
    ros = ros[ros["status"].isin(["A"] + list(IL_AVAIL))].copy()
    ros["avail"] = ros["status"].map(lambda c: IL_AVAIL.get(c, 1.0) if c != "A" else 1.0)
    ros["il"] = ""
    try:
        tr = INJ.transactions(season, refresh=True)
        last = {}
        for x in tr:
            if x.get("typeDesc") != "Status Change": continue
            pid = (x.get("person") or {}).get("id"); dsc = x.get("description") or ""
            if pid and "injured list" in dsc.lower() and " placed " in dsc.lower():
                if pid not in last or x.get("date", "") > last[pid][0]: last[pid] = (x.get("date", ""), dsc)
        for i, r in ros[ros["status"] != "A"].iterrows():
            if r["pid"] in last:
                dt, dsc = last[r["pid"]]; why = dsc.split(".")[-2].strip() if dsc.count(".") >= 2 else ""
                ros.at[i, "il"] = f"{r['status']} since {dt}" + (f": {why}" if why else "")
            else:
                ros.at[i, "il"] = r["status"]
    except Exception as e:
        log("injury text skipped:", e)
    return ros


def hitters(f: pd.DataFrame, season: int, d: pd.DataFrame) -> pd.DataFrame:
    """Every hitter on a playoff club: points per plate appearance, chance to start, usual slot, position."""
    from . import build as B
    H = MS.season_hitting(season)
    lg = {c: H[c].sum() / max(H["PA"].sum(), 1) for c in ("X2", "X3", "HR", "R", "RBI", "SB", "BB", "HBP")}
    lg["X1"] = (H["H"] - H["X2"] - H["X3"] - H["HR"]).sum() / H["PA"].sum(); lg["OUT"] = (H["AB"] - H["H"]).sum() / H["PA"].sum()
    H = _rates(H, pd.Series(lg))
    ros = roster40(f, season)
    stl = B.recent_lineups(d)
    elig = MS.season_fielding(season)
    rows = []
    for t in f.itertuples():
        act = ros[(ros["team_id"] == t.id) & ((ros["ptype"] != "Pitcher") | (ros["pos"] == "TWP")) & (ros["avail"] > 0)]
        pr = B.projected(stl, t.abbr, "R", set(act["pid"])).set_index("batter"); pl = B.projected(stl, t.abbr, "L", set(act["pid"])).set_index("batter")
        proj = pr[["p_start", "proj_slot"]].join(pl[["p_start", "proj_slot"]], how="outer", lsuffix="_r", rsuffix="_l").fillna(0)
        proj["p_start"] = 0.7 * proj["p_start_r"] + 0.3 * proj["p_start_l"]
        proj["slot"] = np.where(proj["proj_slot_r"] > 0, proj["proj_slot_r"], proj["proj_slot_l"])
        for r in act.itertuples():
            hs = H[H["pid"] == r.pid]
            if not len(hs): continue
            hs = hs.iloc[0]; pp = proj.loc[r.pid] if r.pid in proj.index else None
            p_start = float(pp["p_start"]) if pp is not None else 0.0
            slot = int(round(pp["slot"])) if pp is not None and pp["slot"] > 0 else 9
            if r.status != "A" and hs["PA"] >= 150:
                # a regular on the short IL: his usual slot from earlier in the year, at the reduced chance he is back
                seen = stl[(stl["bat_team"] == t.abbr) & (stl["batter"] == r.pid)]
                if len(seen) >= 10: p_start = 0.9; slot = int(round(seen.tail(20)["slot"].mean()))
            if hs["PA"] < 40 and p_start < 0.3: continue
            prim = POS_SLOT.get(r.pos, "UT")
            pos = sorted({prim} | {p for p in elig.get(r.pid, []) if p in ("C", "1B", "2B", "3B", "SS", "OF")}) if prim != "UT" or elig.get(r.pid) else [prim]
            pos = [p for p in pos if p != "UT"] or ["UT"]
            rows.append(dict(pid=r.pid, name=r.name, team=t.abbr, pos=pos, bats=r.bats,
                             PA=int(hs["PA"]), ppa=round(float(hs["ppa"]), 4), p_start=round(p_start, 3), slot=slot, avail=float(r.avail), il=r.il,
                             ppg=round(float(hs["ppa"]) * PA_BY_SLOT.get(min(max(slot, 1), 9), 3.5) * HIT_DEFLATE * p_start, 3),
                             line=dict(HR=int(hs["HR"]), R=int(hs["R"]), RBI=int(hs["RBI"]), SB=int(hs["SB"]), BB=int(hs["BB"]),
                                       AVG=round(hs["H"] / max(hs["AB"], 1), 3), pts=round(float(MS.holdem_hit_points(hs.to_frame().T).iat[0]), 1))))
    return pd.DataFrame(rows)


def pitchers(f: pd.DataFrame, season: int, d: pd.DataFrame) -> pd.DataFrame:
    """Every pitcher on a playoff club: rotation slot or bullpen role, and points per start or per appearance."""
    from . import build as B
    from . import game as G
    Pz = MS.season_pitching(season)
    ros = roster40(f, season)
    x = d[d["season"] == season]
    last = x.groupby("opp_sp")["game_date"].max()
    # batters faced per start from the plate-appearance cache, so a swingman's relief innings do not inflate his starts
    work = G.sp_workload(d); work = work[work["season"] == season]
    bf_start = work.groupby("pitcher")["bf"].apply(lambda v: v.tail(6).mean())
    rows = []
    for t in f.itertuples():
        act = ros[(ros["team_id"] == t.id) & ((ros["ptype"] == "Pitcher") | (ros["pos"] == "TWP")) & (ros["avail"] > 0)]
        ps = Pz[Pz["pid"].isin(act["pid"]) & (Pz["IP"] > 0)].copy()
        if not len(ps): continue
        ps["avail"] = ps["pid"].map(dict(zip(act["pid"], act["avail"]))); ps["il"] = ps["pid"].map(dict(zip(act["pid"], act["il"])))
        ps["ip_gs"] = (ps["pid"].map(bf_start) / 4.3).fillna(ps["IP"] / ps["GS"].clip(lower=1)).clip(upper=7.5)
        ps["k_ip"] = ps["K"] / ps["IP"]; ps["era"] = 9 * ps["ER"] / ps["IP"]
        ps["last"] = ps["pid"].map(last)
        recent = (ps["last"] >= (x["game_date"].max() - pd.Timedelta(days=12))) | ((ps["avail"] < 1) & (ps["GS"] >= 15))
        sp = ps[(ps["GS"] >= 8) & recent].copy()
        # rotation order: rate quality over a standard five innings (a manager starts his best arms, not his longest)
        sp["ps"] = 5 + 5 * sp["k_ip"] - 5 * sp["era"] / 9 * SP_ER_SCALE
        sp = sp.sort_values("ps", ascending=False)
        for i, r in enumerate(sp.head(5).itertuples(), 1):
            ip = min(SP_IP_SCALE * r.ip_gs, 6.5)
            rows.append(dict(pid=r.pid, name=r.name, team=t.abbr, role=f"SP{i}", rot=i, ip=round(ip, 2), k_ip=round(r.k_ip, 3), era=round(r.era, 2), avail=float(r.avail), il=r.il,
                             pps=round(ip + ip * r.k_ip - ip * r.era / 9 * SP_ER_SCALE, 3),     # per start before the win
                             line=dict(GS=int(r.GS), IP=round(r.IP, 1), K=int(r.K), W=int(r.W), ERA=round(r.era, 2), pts=round(float(MS.holdem_pit_points(pd.DataFrame([r._asdict()])).iat[0]), 1))))
        rp = ps[~ps["pid"].isin(sp.head(5)["pid"]) & (ps["G"] >= 15)].copy()
        rp["ip_g"] = rp["IP"] / rp["G"].clip(lower=1)
        closer = rp.sort_values(["SV", "HLD"], ascending=False).head(1)
        setup = rp[~rp["pid"].isin(closer["pid"])].sort_values(["HLD", "K"], ascending=False).head(2)
        for r in closer.itertuples():
            app = 0.55; ip = min(r.ip_g, 1.2)
            rows.append(dict(pid=r.pid, name=r.name, team=t.abbr, role="CL", rot=0, ip=round(ip, 2), k_ip=round(r.k_ip, 3), era=round(r.era, 2), app=app, avail=float(r.avail), il=r.il,
                             ppg=round(app * (ip + ip * r.k_ip - ip * r.era / 9 * SP_ER_SCALE), 3),
                             line=dict(G=int(r.G), SV=int(r.SV), IP=round(r.IP, 1), K=int(r.K), ERA=round(r.era, 2), pts=round(float(MS.holdem_pit_points(pd.DataFrame([r._asdict()])).iat[0]), 1))))
        for r in setup.itertuples():
            app = 0.5; ip = min(r.ip_g, 1.1)
            rows.append(dict(pid=r.pid, name=r.name, team=t.abbr, role="RP", rot=0, ip=round(ip, 2), k_ip=round(r.k_ip, 3), era=round(r.era, 2), app=app, avail=float(r.avail), il=r.il,
                             ppg=round(app * (ip + ip * r.k_ip - ip * r.era / 9 * SP_ER_SCALE), 3),
                             line=dict(G=int(r.G), HLD=int(r.HLD), IP=round(r.IP, 1), K=int(r.K), ERA=round(r.era, 2), pts=round(float(MS.holdem_pit_points(pd.DataFrame([r._asdict()])).iat[0]), 1))))
    return pd.DataFrame(rows)


def expected(hit: pd.DataFrame, pit: pd.DataFrame, P: dict) -> list[dict]:
    """Per player, expected points in each round given the club plays it (and the chance it does), plus the value of
    holding him from the Division Series, adding him for the LCS, or adding him for the World Series."""
    out = []
    for r in hit.itertuples():
        p = P[r.team]
        by = {rnd: round(p["g" + rnd] * r.ppg * r.avail, 2) for rnd in ROUNDS}
        out.append(dict(id=int(r.pid), n=r.name, t=r.team, pos=list(r.pos), kind="H", slot=r.slot, ps=r.p_start, ppg=r.ppg, avail=r.avail, il=r.il or "", by=by, line=r.line))
    for r in pit.itertuples():
        p = P[r.team]; by = {}
        for rnd, n in (("DS", 5), ("LCS", 7), ("WS", 7)):
            reach = p["r" + rnd]; games = p["g" + rnd]; wins = p["w" + rnd]
            pwin = wins / games if games > 0 else 0.5
            if r.role.startswith("SP"):
                starts = sum(reach[g - 1] for g in SP_GAME[n].get(r.rot, [])) if r.rot <= 4 else 0.0
                by[rnd] = round(starts * (r.pps + 4 * pwin * SP_WIN_SHARE) * r.avail, 2)
            elif r.role == "CL":
                by[rnd] = round(games * (r.ppg + 4 * pwin * CLOSER_SAVE_SHARE) * r.avail, 2)
            else:
                by[rnd] = round(games * r.ppg * r.avail, 2)
        out.append(dict(id=int(r.pid), n=r.name, t=r.team, pos=["P"], kind="P", role=r.role, avail=r.avail, il=r.il or "", by=by, line=r.line))
    for o in out:
        p = P[o["t"]]; b = o["by"]
        o["p"] = dict(DS=round(p["DS"], 4), LCS=round(p["LCS"], 4), WS=round(p["WS"], 4))
        o["ev"] = round(p["DS"] * b["DS"] + 2 * p["LCS"] * b["LCS"] + 3 * p["WS"] * b["WS"], 1)       # held from the start
        o["ev_lcs"] = round(p["LCS"] * b["LCS"] + 2 * p["WS"] * b["WS"], 1)
        o["ev_ws"] = round(p["WS"] * b["WS"], 1)
    return out


# ------------------------------------------------------------------ the roster, for one bracket
def path_values(players: list[dict], path: dict) -> None:
    """Under a chosen bracket (which clubs win the DS and the LCS), what each player is worth held, added at the LCS,
    or added at the WS. Written onto each player as v_held / v_lcs / v_ws. A club's survival is 0/1 on the path;
    series lengths stay random."""
    ds_w, lcs_w = set(path["ds"]), set(path["lcs"])
    for o in players:
        b = o["by"]; t = o["t"]
        in_lcs, in_ws = t in ds_w, t in lcs_w
        o["v_held"] = round(b["DS"] + (2 * b["LCS"] if in_lcs else 0) + (3 * b["WS"] if in_ws else 0), 1)
        o["v_lcs"] = round((b["LCS"] if in_lcs else 0) + (2 * b["WS"] if in_ws else 0), 1)
        o["v_ws"] = round(b["WS"] if in_ws else 0, 1)


SLOT_TYPES = ["C", "1B", "2B", "3B", "SS", "OF", "UT", "P"]
SLOT_CAP = {"C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "OF": 4, "UT": 1, "P": 6}


def _assign(players, key, cap: dict, team_lo: dict, team_hi: dict, team_exact: dict | None = None):
    """Best assignment of players to open slot types by value `key`: each open slot filled, per-club counts within
    bounds (or exact), a player at most once and only where eligible. scipy's MILP (HiGHS); exact."""
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import lil_matrix
    types = [k for k in SLOT_TYPES if cap.get(k, 0) > 0]
    pairs = [(o, k) for o in players if o[key] > 0 for k in types if k in elig_slots(o["pos"])]
    if not pairs or sum(cap.values()) == 0: return []
    n = len(pairs); teams = sorted({o["t"] for o, _ in pairs})
    A = lil_matrix((len(players) + len(types) + len(teams), n)); lo = []; hi = []
    pidx = {id(o): i for i, o in enumerate(players)}
    for j, (o, k) in enumerate(pairs):
        A[pidx[id(o)], j] = 1
        A[len(players) + types.index(k), j] = 1
        A[len(players) + len(types) + teams.index(o["t"]), j] = 1
    lo += [0] * len(players); hi += [1] * len(players)
    for k in types: lo.append(cap[k]); hi.append(cap[k])
    for t in teams:
        if team_exact is not None: lo.append(team_exact.get(t, 0)); hi.append(team_exact.get(t, 0))
        else: lo.append(team_lo.get(t, 0)); hi.append(team_hi.get(t, 99))
    c = -np.array([o[key] for o, _ in pairs], dtype=float)
    res = milp(c, constraints=LinearConstraint(A.tocsr(), lo, hi), integrality=np.ones(n), bounds=Bounds(0, 1))
    if res.x is None: return []
    return [(pairs[j][0], pairs[j][1]) for j in range(n) if res.x[j] > 0.5]


def build_roster(players: list[dict], path: dict, teams: list[str] | None = None) -> dict:
    """The best Division Series roster for a chosen bracket and the replacements each later round, each stage an exact
    assignment. Quotas follow the bracket: three from each club expected to win its DS, one from each expected to lose;
    the LCS adds go where the WS multiplier will be; the WS adds fill what the LCS losers vacate."""
    teams = list(path.get("ds8") or teams)
    players = [o for o in players if o["t"] in teams]
    path_values(players, path)
    ds_w, lcs_w = list(path["ds"]), list(path["lcs"])
    quota = {t: (3 if t in ds_w else 1) for t in teams}
    ds = _assign(players, "v_held", dict(SLOT_CAP), {}, {}, team_exact=quota)
    if len(ds) < 16:                                   # a club short of eligible bodies at some position: relax to bounds
        ds = _assign(players, "v_held", dict(SLOT_CAP), {t: 1 for t in teams}, {t: 3 for t in teams})
    used = {o["id"] for o, _ in ds}
    kept = [(o, k) for o, k in ds if o["t"] in ds_w]
    freed = {}; [freed.__setitem__(k, freed.get(k, 0) + 1) for o, k in ds if o["t"] not in ds_w]
    have = {}; [have.__setitem__(o["t"], have.get(o["t"], 0) + 1) for o, k in kept]
    pool = [o for o in players if o["t"] in ds_w and o["id"] not in used]
    lcs_add = _assign(pool, "v_lcs", freed, {t: max(0, 2 - have.get(t, 0)) for t in ds_w}, {t: 6 - have.get(t, 0) for t in ds_w})
    used |= {o["id"] for o, _ in lcs_add}
    lcs_roster = kept + lcs_add
    kept2 = [(o, k) for o, k in lcs_roster if o["t"] in lcs_w]
    freed2 = {}; [freed2.__setitem__(k, freed2.get(k, 0) + 1) for o, k in lcs_roster if o["t"] not in lcs_w]
    have2 = {}; [have2.__setitem__(o["t"], have2.get(o["t"], 0) + 1) for o, k in kept2]
    pool2 = [o for o in players if o["t"] in lcs_w and o["id"] not in used]
    ws_add = _assign(pool2, "v_ws", freed2, {t: max(0, 6 - have2.get(t, 0)) for t in lcs_w}, {t: 10 - have2.get(t, 0) for t in lcs_w})
    D = [dict(id=o["id"], slot=k, v=o["v_held"]) for o, k in ds]
    L = [dict(id=o["id"], slot=k, v=o["v_lcs"]) for o, k in lcs_add]
    W = [dict(id=o["id"], slot=k, v=o["v_ws"]) for o, k in ws_add]
    total = sum(x["v"] for x in D + L + W)
    by_id = {o["id"]: o for o in players}
    return dict(path=path, ds=D, lcs_add=L, ws_add=W, total=round(total, 1),
                held_share=round(sum(x["v"] for x in D if by_id[x["id"]]["t"] in lcs_w) / max(total, 1), 3))


def ev_roster(players: list[dict], teams: list[str]) -> dict:
    """The Division Series roster that maximises expected held value over the whole bracket distribution, within the
    contest's 1-to-3 per club. What a single entry should look like if you refuse to pick a bracket."""
    for o in players: o["v_ev"] = o["ev"]
    ds = _assign(players, "v_ev", dict(SLOT_CAP), {t: 1 for t in teams}, {t: 3 for t in teams})
    return dict(ds=[dict(id=o["id"], slot=k, v=o["ev"]) for o, k in ds], total=round(sum(o["ev"] for o, _ in ds), 1))


def top_brackets(B: "Bracket", n: int = 24) -> list[dict]:
    """One bracket per World Series pair, the likeliest pairs first. Every path is walked as a tree (Wild Card winners,
    Division Series winners, pennant winners, each step priced from the ratings and from the games already played
    once a series is under way); a pair's chance is the sum over every path that ends in it, and the bracket carries
    the single likeliest path to it: ds8 (the Division Series field), ds (its four winners), lcs (the pair), p (that
    path's chance), p_pair (the pair's chance) and p_field (the chance of that Division Series field).

    Ranked by pair rather than by path because that is what an entry is: two thirds of a winning roster's points come
    from the two World Series clubs, so two paths to the same pair are nearly the same entry, and a list of the
    likeliest paths was twenty-two versions of one World Series."""
    from itertools import product
    f = B.f; S = {(r.league, r.seed): r.abbr for r in f.itertuples()}; seedno = B.seedno
    per_league = {}
    for L in ("AL", "NL"):
        paths = []
        wc = {}
        for hi, lo in ((3, 6), (4, 5)):
            a, b = S[(L, hi)], S[(L, lo)]
            pw = B._series_from("F", a, b, 3)[0]
            wc[(hi, lo)] = [(a, pw), (b, 1 - pw)]
        for (w45, p45), (w36, p36) in product(wc[(4, 5)], wc[(3, 6)]):
            ds_field = [S[(L, 1)], S[(L, 2)], w45, w36]
            d1 = B._series_from("D", S[(L, 1)], w45, 5)[0]; d2 = B._series_from("D", S[(L, 2)], w36, 5)[0]
            for (x, px), (y, py) in product([(S[(L, 1)], d1), (w45, 1 - d1)], [(S[(L, 2)], d2), (w36, 1 - d2)]):
                hi, lo = (x, y) if seedno[x] < seedno[y] else (y, x)
                pl = B._series_from("L", hi, lo, 7)[0]
                for (z, pz) in ((hi, pl), (lo, 1 - pl)):
                    paths.append(dict(ds8=ds_field, ds=[x, y], lcs=z, p=p45 * p36 * px * py * pz, p_field=p45 * p36))
        best, tot = {}, {}
        for q in paths:
            tot[q["lcs"]] = tot.get(q["lcs"], 0.0) + q["p"]
            if q["lcs"] not in best or q["p"] > best[q["lcs"]]["p"]: best[q["lcs"]] = q
        per_league[L] = (best, tot)
    out = []
    for a, pa in per_league["AL"][1].items():
        for b, pb in per_league["NL"][1].items():
            qa, qb = per_league["AL"][0][a], per_league["NL"][0][b]
            out.append(dict(ds8=qa["ds8"] + qb["ds8"], ds=qa["ds"] + qb["ds"], lcs=[a, b], p=round(qa["p"] * qb["p"], 5),
                            p_pair=round(pa * pb, 5), p_field=round(qa["p_field"] * qb["p_field"], 5)))
    out.sort(key=lambda x: -x["p_pair"])
    return out[:n]


# ------------------------------------------------------------------ what the record says (2022-25 postseasons, boxes in data/postseason)
def history_notes() -> dict:
    """The shape of a winning entry, from the perfect-bracket roster each of the last four Octobers."""
    p = DATA / "postseason" / "boxes_2022_2025.parquet"
    if not p.exists(): return {}
    d = pd.read_parquet(p); d = d[d["gt"].isin(["D", "L", "W"])].copy(); d["pts"] = d["hit_pts"] + d["pit_pts"]
    reached = d.groupby(["season", "team"])["gt"].agg(lambda s: "W" if "W" in set(s) else "L" if "L" in set(s) else "D")
    out = []
    for season, x in d.groupby("season"):
        x2 = x.assign(D=np.where(x["gt"] == "D", x["pts"], 0.0), L=np.where(x["gt"] == "L", x["pts"], 0.0), W=np.where(x["gt"] == "W", x["pts"], 0.0))
        P = x2.groupby(["pid", "name", "team"])[["D", "L", "W"]].sum().reset_index()
        P["reached"] = [reached[(season, t)] for t in P["team"]]
        ws = P[P.reached == "W"]; lcs = P[P.reached == "L"]; ds = P[P.reached == "D"]
        total = 0.0; from_ws = 0.0; held = 0.0; names = []
        for t, g in ws.groupby("team"):
            g = g.assign(hv=g.D + 2 * g.L + 3 * g.W, lv=g.L + 2 * g.W)
            h = g.nlargest(3, "hv"); r = g.drop(h.index); a = r.nlargest(3, "lv"); w = r.drop(a.index).nlargest(2, "W")
            v = h.hv.sum() + a.lv.sum() + w.W.sum(); total += v; from_ws += v; held += h.hv.sum(); names += list(h.name)
        for t, g in lcs.groupby("team"):
            g = g.assign(v=g.D + 2 * g.L); total += g.nlargest(3, "v").v.sum()
        for t, g in ds.groupby("team"):
            total += g.nlargest(1, "D").D.sum()
        out.append(dict(season=int(season), ws=sorted(ws.team.unique().tolist()), lcs_losers=sorted(lcs.team.unique().tolist()),
                        perfect=round(total), from_ws_pct=round(100 * from_ws / total), held_pct=round(100 * held / total), held_names=names))
    usage = d[d["started"]].groupby("pid").size()
    return dict(years=out, starter_ip=round(float(d[d["started"]]["ip"].mean()), 2), saves_per_game=round(float(d["sv"].sum() / d["pk"].nunique()), 3),
                pts_split=dict(hitters=round(float(d["hit_pts"].sum())), pitchers=round(float(d["pit_pts"].sum()))))


STRATEGY = [
    dict(h="It is a bracket contest first",
         p="Sixty-four to seventy-seven percent of a perfect entry's points in each of the last four Octobers came from the two World Series clubs, and about half from the six men held on them from the Division Series at triple points. The four Division Series losers together were worth one to two percent. Pick the two pennant winners; everything else is detail."),
    dict(h="Three from each club you believe in, one from each you do not",
         p="With eight clubs, a minimum of one each and a maximum of three, the roster is 3-3-3-3-1-1-1-1: three from each of the four clubs you expect to win the Division Series, one throwaway from each you expect to lose. The throwaways are where to park the catcher and the weakest infield spot (Pullhitter's punt; Zola's Dillon Dingler warning cuts the other way: a throwaway is still a real player for one series)."),
    dict(h="Aces and top-of-the-order bats hold; the multiplier is what you are buying",
         p="The held names on the perfect rosters were Valdez, Suarez, Eovaldi, Yamamoto, Snell and Yesavage; Schwarber, Harper, Seager, Betts, Freeman, Soto, Ohtani and Guerrero. A number-one starter throws game 1 and game 5 of a five-game series and games 1 and 5 (and maybe 7) of a seven, so his expected starts scale with series length the way a leadoff hitter's plate appearances do. Playoff starts are short (4.4 innings on average since 2022), so the innings point is worth less than it looks and the strikeout and win points more."),
    dict(h="Closers are a hold, setup men are not",
         p="A save is four points and the closer gets one in about 48% of his club's wins. On a club that wins twelve games in October that is 23 points before innings and strikeouts, and it triples in the Series. Middle relievers are two points a night and not worth a slot the multiplier could be on."),
    dict(h="Replacements are worth less than they look",
         p="A player added for the LCS scores single there and double in the Series; added for the Series, single. The perfect rosters got about a third as much from replacements as from held players. Do not plan to fix the roster later; the entry is decided at 1 PM on October 3."),
    dict(h="Multiple entries mean different brackets, not different players",
         p="Zola's point: with up to twelve entries the way to be different is to back a different pair of pennant winners, not to swap in a lesser player on the same bracket. Two entries on the same bracket mostly duplicate each other. The Brackets table lists the likely ones with the chance of each; spend entries down that list."),
    dict(h="Byes are worth less than they used to be, but a bye club still cannot lose the Wild Card",
         p="Since 2022 the two bye clubs in each league have reached the World Series in six of sixteen chances, and a Wild Card club did in five of eight Series. The markets already price this; the point is that a bye is certainty about the Division Series, not about October."),
    dict(h="What the model does not know",
         p="Lineups and rotations in October are set by managers who have not decided yet: the rotation order here is the season's rate quality among September starters, and a club can go to a three-man rotation or an opener. Injured players on the 10- or 15-day list are carried at a reduced chance and flagged; count them in or out yourself as news breaks. Series lengths are random even on a chosen bracket, so the numbers are averages."),
]


def write(season: int, out: Path | None = None) -> dict:
    from . import build as B
    t0 = time.time()
    f, settled = field(season)
    sched = playoff_schedule(season)
    mk = market_odds(season)
    rec = {r.abbr: r.w for r in f.itertuples()}
    s = fit_strength(f, mk, played=sched, records=rec)
    P = Bracket(f, s, played=sched, records=rec).run()
    log(f"field {'settled' if settled else 'projected'}, markets {mk['sources']}, ratings fit ({time.time()-t0:.0f}s)")
    d = B.load_history(datetime.now().date().isoformat())
    hit = hitters(f, season, d); pit = pitchers(f, season, d)
    players = expected(hit, pit, P)
    players.sort(key=lambda o: -o["ev"])
    log(f"{len(hit)} hitters, {len(pit)} pitchers ({time.time()-t0:.0f}s)")
    B = Bracket(f, s, played=sched, records=rec)
    brackets = top_brackets(B, n=24)
    rosters = []
    for b in brackets[:12]:
        r = build_roster([dict(o) for o in players], b); r["p"] = b["p"]; rosters.append(r)
    ds8 = max(brackets, key=lambda b: b["p_field"])["ds8"]   # the likeliest Division Series field (the field itself, once the Wild Cards are done)
    evr = ev_roster([dict(o) for o in players if o["t"] in ds8], ds8); evr["ds8"] = ds8
    log(f"{len(brackets)} brackets, {len(rosters)} rosters, EV roster {evr['total']} ({time.time()-t0:.0f}s)")
    team_rows = []
    for t in f.itertuples():
        p = P[t.abbr]
        team_rows.append(dict(abbr=t.abbr, name=t.name, league=t.league, seed=int(t.seed), w=int(t.w), l=int(t.l), bye=bool(t.bye), clinch=t.clinch, rating=round(float(s[t.abbr]), 3),
                              p=dict(DS=round(p["DS"], 4), LCS=round(p["LCS"], 4), WS=round(p["WS"], 4), pennant=round(p["pennant"], 4), title=round(p["title"], 4)),
                              g=dict(DS=round(p["gDS"], 2), LCS=round(p["gLCS"], 2), WS=round(p["gWS"], 2)),
                              reach=dict(DS=p["rDS"], LCS=p["rLCS"], WS=p["rWS"]),
                              opp=dict(DS={k: round(v, 3) for k, v in p["opp"].get("DS", {}).items()}),
                              mk=dict(title=round(mk["ws"].get(t.abbr, 0), 4), pennant=round(mk[t.league].get(t.abbr, 0), 4))))
    payload = dict(built=datetime.now(timezone.utc).isoformat(timespec="seconds"), season=season, settled=settled,
                   deadline="2026-10-03T17:00:00Z", rules=dict(slots=SLOTS, limits=LIMITS, mult=MULT, scoring=dict(hit=MS.HOLDEM_H, pit=MS.HOLDEM_P)),
                   teams=team_rows, players=players, brackets=brackets, rosters=rosters, ev_roster=evr, history=history_notes(), strategy=STRATEGY,
                   assumptions=dict(hit_deflate=HIT_DEFLATE, sp_ip_scale=SP_IP_SCALE, sp_er_scale=SP_ER_SCALE, sp_win_share=SP_WIN_SHARE,
                                    closer_save_share=CLOSER_SAVE_SHARE, home_logit=HOME_LOGIT, pa_by_slot=PA_BY_SLOT, il_avail=IL_AVAIL),
                   market_raw=mk.get("raw"))
    out = out or (SITE / "data" / "postseason.js")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("window.POST=" + json.dumps(payload, separators=(",", ":"), default=_json) + ";\n")
    log(f"wrote {out} ({out.stat().st_size/1e3:.0f} kB) in {time.time()-t0:.0f}s")
    return payload


def _json(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return None if math.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (pd.Timestamp, datetime)): return o.isoformat()
    return str(o)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--season", type=int, default=datetime.now().year)
    a = ap.parse_args()
    write(a.season)

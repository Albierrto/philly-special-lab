"""Fresh Kalshi prices for every Kalshi market on the board -> site/hr/data/kalshi.json.   python -m hrboard.kalshi_live

Kalshi refuses requests from web pages, so the page cannot ask it directly. This runs on its own every ~20 minutes
(the hrlive workflow) and with every board build. Standard library only, so the job needs no installs.
"""
from __future__ import annotations
import json, time, urllib.parse, urllib.request
from datetime import datetime, timezone
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "site" / "hr" / "data"
API = "https://api.elections.kalshi.com/trade-api/v2/markets"


def board() -> dict:
    s = (DATA / "hr.js").read_text()
    return json.loads(s[s.index("=") + 1:s.rstrip().rindex(";")])


def entries(payload: dict):
    """(ticker, [bid, ask, last, bid size, ask size]) for every Kalshi price in the payload, skipping finished games."""
    for sl in payload.get("slates", []):
        live = {g["pk"] for g in sl["games"] if g.get("state") != "Final"}
        for g in sl["games"]:
            if g["pk"] not in live: continue
            ks = (g.get("mk") or {}).get("ks") or {}
            for v in (ks.get("win") or {}).values():
                if len(v) > 4: yield v[4], [v[1], v[2], v[0], v[5] if len(v) > 5 else None, v[6] if len(v) > 6 else None]
            for v in (ks.get("total") or {}).values():
                if len(v) > 3: yield v[3], [v[1], v[2], v[0], v[4] if len(v) > 4 else None, v[5] if len(v) > 5 else None]
        for who in sl.get("hitters", []) + sl.get("pitchers", []):
            if who.get("pk") not in live: continue
            for v in ((who.get("mk") or {}).get("ks") or {}).values():
                if len(v) > 3: yield v[3], [v[1], v[2], v[0], v[4] if len(v) > 4 else None, v[5] if len(v) > 5 else None]


def _get(url):
    for i in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"Accept": "application/json"}), timeout=30) as r:
                return json.load(r)
        except Exception:
            time.sleep(2 + 3 * i)
    return {}


def _num(x):
    try: return round(float(x), 4)
    except (TypeError, ValueError): return None


def fetch(tickers) -> dict:
    tickers = list(dict.fromkeys(t for t in tickers if t))
    out = {}
    for i in range(0, len(tickers), 100):
        q = urllib.parse.urlencode(dict(tickers=",".join(tickers[i:i + 100]), limit=100))
        for m in _get(f"{API}?{q}").get("markets", []):
            out[m["ticker"]] = [_num(m.get("yes_bid_dollars")), _num(m.get("yes_ask_dollars")), _num(m.get("last_price_dollars")),
                                _num(m.get("yes_bid_size_fp")), _num(m.get("yes_ask_size_fp"))]
        time.sleep(0.2)
    return out


def write(payload: dict | None = None, live: bool = False) -> int:
    payload = payload or board()
    prices = dict(entries(payload))
    if live:
        fresh = fetch(list(prices))
        if not fresh:
            print("kalshi: no response, keeping the last file"); return 0
        prices = fresh
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "kalshi.json").write_text(json.dumps(dict(as_of=datetime.now(timezone.utc).isoformat(timespec="seconds"), p=prices),
                                                 separators=(",", ":")))
    print(f"kalshi: {len(prices)} prices")
    return len(prices)


if __name__ == "__main__":
    import sys
    n = write(live=True)
    sys.exit(0 if n else 3)          # 3 = nothing on the board to price (the workflow then skips the deploy)

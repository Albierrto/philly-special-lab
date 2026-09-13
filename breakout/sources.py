"""Data fetchers with on-disk caching. Everything here is a plain GET/POST to public pages — no logins.

    savant_custom(year)      -> list[dict]   ~600 batters/yr, 196 Statcast fields (xwOBA, barrels, bat speed, sprint...)
    mlb_season(group, year)  -> list[dict]   every player, MLB Stats API season totals (hitting or fielding)
    fantasypros_adp(year)    -> DataFrame    preseason composite ADP archive (NFBC / Fantrax / CBS / Yahoo / RTS / ESPN)
    nfbc_adp(from,to)        -> DataFrame    live NFBC ADP for a date window (current draft season only)
"""
from __future__ import annotations
import json, re, time
from pathlib import Path
import pandas as pd
import requests

from .config import DATA

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36"}
_S = requests.Session(); _S.headers.update(UA)


def _cached(path: Path, fetch, refresh=False):
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    data = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data


# ----------------------------------------------------------------------------- Baseball Savant
def savant_custom(year: int, min_pa: int = 25, refresh=False) -> list[dict]:
    """The custom leaderboard page embeds `var data = [...]` with every metric for every batter >= min_pa."""
    def fetch():
        u = (f"https://baseballsavant.mlb.com/leaderboard/custom?year={year}&type=batter&filter=&min={min_pa}"
             f"&selections=xwoba&chart=false&x=xwoba&y=xwoba&r=no&chartType=beeswarm&sort=xwoba&sortDir=desc")
        h = _S.get(u, timeout=120).text
        i = h.find("var data = ["); j = h.find("];", i)
        if i < 0:
            raise RuntimeError(f"Savant custom leaderboard: could not find embedded data for {year}")
        return json.loads(h[i + 11:j + 1])
    return _cached(DATA / "savant" / f"custom_{year}.json", fetch, refresh)


# ----------------------------------------------------------------------------- MLB Stats API
def mlb_season(group: str, year: int, refresh=False) -> list[dict]:
    """group = 'hitting' | 'fielding'. Season totals across all teams, person hydrated (birthDate, primaryPosition)."""
    def fetch():
        out, offset = [], 0
        while True:
            u = (f"https://statsapi.mlb.com/api/v1/stats?stats=season&group={group}&season={year}&sportId=1"
                 f"&playerPool=all&limit=1000&offset={offset}&hydrate=person")
            st = _S.get(u, timeout=120).json()["stats"][0]
            sp = st.get("splits", [])
            out += sp
            offset += len(sp)
            if not sp or offset >= st.get("totalSplits", 0):
                break
        return out
    return _cached(DATA / "mlb" / f"{group}_{year}.json", fetch, refresh)


# ----------------------------------------------------------------------------- FantasyPros ADP archive
def fantasypros_adp(year: int, refresh=False) -> pd.DataFrame:
    p = DATA / "adp" / f"fantasypros_overall_{year}.csv"
    if p.exists() and not refresh:
        return pd.read_csv(p, dtype=str)
    h = _S.get(f"https://www.fantasypros.com/mlb/adp/overall.php?year={year}", timeout=90).text
    tbl = re.search(r'<table[^>]*id="data"[^>]*>.*?</table>', h, re.S).group(0)
    heads = [re.sub("<[^>]+>", "", x).strip() for x in
             re.findall(r"<th[^>]*>(.*?)</th>", re.search(r"<thead>.*?</thead>", tbl, re.S).group(0), re.S)]
    body = re.search(r"<tbody>(.*?)</tbody>", tbl, re.S).group(1)
    rows = []
    for b in re.split(r'<tr class="mpb-player-', body)[1:]:
        fpid = re.match(r"(\d+)", b).group(1)
        name = re.search(r'fp-player-name="([^"]+)"', b).group(1)
        plain = re.sub("<[^>]+>", "", b)
        tp = re.search(r"\(\s*([A-Z]+)\s*-\s*([^)]+)\)", plain)
        team = tp.group(1) if tp else ""; pos = tp.group(2).strip() if tp else ""
        tds = [re.sub("<[^>]+>", "", t).replace("&nbsp;", "").strip() for t in re.findall(r"<td[^>]*>(.*?)</td>", b, re.S)]
        rows.append([tds[0], fpid, name, team, pos] + tds[2:])
    df = pd.DataFrame(rows, columns=["rank", "fp_id", "player", "team", "pos"] + heads[2:])
    p.parent.mkdir(parents=True, exist_ok=True); df.to_csv(p, index=False)
    return df


# ----------------------------------------------------------------------------- NFBC live ADP
def nfbc_adp(from_date: str, to_date: str, draft_type: int = 0, num_teams: int = 0, refresh=False) -> pd.DataFrame:
    p = DATA / "adp" / f"nfbc_{from_date}_{to_date}_{draft_type}_{num_teams}.csv"
    if p.exists() and not refresh:
        return pd.read_csv(p)
    s = requests.Session(); s.headers.update(UA)
    s.headers.update({"X-Requested-With": "XMLHttpRequest", "Referer": "https://nfc.shgn.com/adp/baseball"})
    s.get("https://nfc.shgn.com/adp/baseball", timeout=60)
    r = s.post("https://nfc.shgn.com/adp.data.php", timeout=90, data={
        "team_id": 0, "from_date": from_date, "to_date": to_date, "num_teams": num_teams,
        "draft_type": draft_type, "sport": "baseball", "position": "", "league_teams": 0, "as_board": ""})
    rows = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", r.text, re.S):
        c = [re.sub("<[^>]+>", "", x).strip() for x in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
        if len(c) >= 9 and c[0].isdigit():
            rows.append(dict(rank=int(c[0]), player=c[1], team=c[2], pos=c[3], adp=float(c[4]),
                             min=c[5], max=c[6], picks=c[8]))
    df = pd.DataFrame(rows)
    p.parent.mkdir(parents=True, exist_ok=True); df.to_csv(p, index=False)
    return df

"""Fantrax, read-only, through its official external API (no login, no keys: the league allows API reads).

    python -m breakout.fantrax_api            # refresh data/fantrax/{rosters,standings,teams}_<season>.csv/json and the owner tables

Endpoints (https://www.fantrax.com/fxea/general/...): getLeagueInfo, getTeamRosters, getStandings, getDraftResults,
getPlayerIds. Nothing here can change a roster, make a claim, propose a trade or post a message; it only reads.
"""
from __future__ import annotations
import json, re, sys, time
import pandas as pd
import requests
from . import config as C

BASE = "https://www.fantrax.com/fxea/general"
UA = {"User-Agent": "philly-special-lab/1.0 (read-only league sync)"}
DRAFT_RULE = "bottom4_then_reverse"  # 9th-12th place pick 1-4 in that order, then 8th..1st pick 5..12 (fixed order, no snake)


def _get(path: str, **params) -> dict | list:
    for attempt in range(3):
        r = requests.get(f"{BASE}/{path}", params=params, headers=UA, timeout=60)
        if r.ok:
            try:
                return json.loads(r.content.decode("utf-8"))
            except ValueError:
                pass
        time.sleep(2 + 3 * attempt)
    r.raise_for_status(); return {}


def league_id(season: int | None = None) -> str:
    cfg = json.loads((C.DATA / "fantrax" / "league_config.json").read_text())
    ids = cfg.get("league_ids", {}); season = season or C.CURRENT_SEASON
    return ids.get(str(season)) or cfg.get("league_id") or next(iter(ids.values()))


def player_ids() -> dict:
    return _get("getPlayerIds", sport="MLB")


def flip_name(n: str) -> str:
    """'Henderson, Gunnar' -> 'Gunnar Henderson'; suffixes stay ('Witt Jr., Bobby' -> 'Bobby Witt Jr.')."""
    if "," in n:
        last, first = [x.strip() for x in n.split(",", 1)]
        return f"{first} {last}".strip()
    return n.strip()


def teams(lid: str) -> pd.DataFrame:
    info = _get("getLeagueInfo", leagueId=lid)
    seen = {}
    for m in info.get("matchups", []):
        for g in m.get("matchupList", []):
            for side in ("away", "home"):
                t = g.get(side) or {}
                if t.get("id"): seen[t["id"]] = dict(team_id=t["id"], team=t.get("name"), abbrev=t.get("shortName"))
    for tid, t in (info.get("teamInfo") or {}).items():
        seen.setdefault(tid, dict(team_id=tid, team=t.get("name"), abbrev=t.get("shortName")))
    return pd.DataFrame(list(seen.values()))


def standings(lid: str, tm: pd.DataFrame) -> pd.DataFrame:
    st = _get("getStandings", leagueId=lid)
    rows = []
    for r in st if isinstance(st, list) else st.get("standings", []):
        w, l, t = (r.get("points") or "0-0-0").split("-")[:3]
        rows.append(dict(rank=int(r.get("rank", 0)), team=r.get("teamName"), team_id=r.get("teamId"), W=int(w), L=int(l), T=int(t),
                         win_pct=r.get("winPercentage"), points_for=r.get("totalPointsFor"), games_back=r.get("gamesBack")))
    df = pd.DataFrame(rows).sort_values("rank")
    return df.merge(tm[["team_id", "abbrev"]], on="team_id", how="left")


def draft_slots_from_standings(st: pd.DataFrame, rule: str = DRAFT_RULE) -> pd.DataFrame:
    """Next year's fixed (non-snake) draft order from this year's regular-season finish."""
    n = len(st); order = st.sort_values("rank")["abbrev"].tolist()  # index 0 = 1st place
    if rule == "bottom4_then_reverse" and n >= 6:
        seq = order[n - 4:] + order[: n - 4][::-1]   # 9th,10th,11th,12th then 8th ... 1st
    else:
        seq = order[::-1]                            # plain reverse standings
    return pd.DataFrame([dict(slot=i + 1, abbrev=a, team=st.loc[st["abbrev"] == a, "team"].iloc[0]) for i, a in enumerate(seq)])


def rosters(lid: str, tm: pd.DataFrame, ids: dict) -> pd.DataFrame:
    ro = _get("getTeamRosters", leagueId=lid)
    ab = dict(zip(tm["team_id"], tm["abbrev"])); rows = []
    for tid, t in (ro.get("rosters") or {}).items():
        for it in t.get("rosterItems", []):
            p = ids.get(it.get("id"), {})
            rows.append(dict(fantrax_id=it.get("id"), player=flip_name(p.get("name", "")), mlb=p.get("team"), pos=p.get("position"),
                             slot=it.get("position"), status=it.get("status"), owner=ab.get(tid, tid), team=t.get("teamName")))
    return pd.DataFrame(rows)


def draft_results(lid: str, tm: pd.DataFrame, ids: dict) -> pd.DataFrame:
    d = _get("getDraftResults", leagueId=lid); nm = dict(zip(tm["team_id"], tm["team"])); rows = []
    for p in d.get("draftPicks", []):
        pl = ids.get(p.get("playerId"), {})
        rows.append(dict(overall=p.get("pick"), round=p.get("round"), pick=p.get("pickInRound"), team=nm.get(p.get("teamId"), p.get("teamId")),
                         player=flip_name(pl.get("name", "")), pos=pl.get("position"), mlb=pl.get("team")))
    return pd.DataFrame(rows).sort_values("overall")


def schedule(lid: str) -> dict:
    """The league's scoring periods and its head-to-head schedule, straight from getLeagueInfo.

    Regular-season periods name the two teams. Playoff periods name only SEEDS, because who fills them depends on
    results Fantrax has not played yet, so the matchup simulator falls back to a team picker once the bracket starts.
    """
    info = _get("getLeagueInfo", leagueId=lid)
    periods = [dict(n=int(p["number"]), start=str(p["startDate"])[:10], end=str(p["endDate"])[:10]) for p in info.get("scoringPeriods", [])]
    games = []
    for grp in info.get("matchups", []):
        for g in grp.get("matchupList", []):
            a, h = g.get("away") or {}, g.get("home") or {}
            games.append(dict(period=int(grp["period"]), away=a.get("shortName"), home=h.get("shortName"),
                              away_seed=a.get("seed"), home_seed=h.get("seed")))
    pl = info.get("playoffs") or {}
    return dict(periods=periods, games=games, first_playoff_period=pl.get("firstPlayoffPeriod"),
                last_regular_period=pl.get("lastRegularSeasonPeriod"), playoff_teams=pl.get("numPlayoffTeams"))


def sync(season: int | None = None) -> dict:
    season = season or C.CURRENT_SEASON; lid = league_id(season); out = C.DATA / "fantrax"; out.mkdir(parents=True, exist_ok=True)
    ids = player_ids(); tm = teams(lid); st = standings(lid, tm); ro = rosters(lid, tm, ids); slots = draft_slots_from_standings(st)
    tm.to_csv(out / f"teams_{season}.csv", index=False); st.to_csv(out / f"standings_{season}.csv", index=False)
    ro.to_csv(out / f"rosters_{season}.csv", index=False); slots.to_csv(out / f"draft_slots_{season + 1}.csv", index=False)
    # owner tables the pipelines read (player, owner, mlb): every rostered player in both files, so a two-way player
    # (Ohtani is listed as "DH" by Fantrax) is found from either side; the pipelines only keep names they know. Free agents are absent.
    cols = ["player", "pos", "mlb", "owner", "status"]
    ro[cols].to_csv(out / f"hitters_{season}.csv", index=False); ro[cols].to_csv(out / f"pitchers_{season}.csv", index=False)
    # keep the league config's team names in step
    cfg_p = out / "league_config.json"; cfg = json.loads(cfg_p.read_text())
    names = cfg.get("fantasy_team_abbrevs", {}); names.update({a: n for a, n in zip(tm["abbrev"], tm["team"]) if a and n})
    if cfg.get("user_team_abbrev") in names: names[cfg["user_team_abbrev"]] = f"{names[cfg['user_team_abbrev']].split(' (')[0]} (Bort)"
    cfg["fantasy_team_abbrevs"] = names; cfg[f"standings_{season}"] = st[["rank", "team", "abbrev", "W", "L", "points_for"]].to_dict(orient="records")
    cfg[f"draft_order_{season + 1}"] = slots["team"].tolist(); cfg["draft_order_rule"] = DRAFT_RULE
    try: cfg["schedule"] = schedule(lid)          # scoring periods + head-to-head, for the matchup simulator
    except Exception as e: print("schedule sync failed, keeping what is on file:", e)
    cfg_p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    res = dict(teams=len(tm), rostered=len(ro), standings=len(st), synced_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), season=season)
    (out / "sync_meta.json").write_text(json.dumps(res, indent=2))
    return res


def main(argv=None):
    r = sync(); print(r)
    st = pd.read_csv(C.DATA / "fantrax" / f"standings_{C.CURRENT_SEASON}.csv"); print(st[["rank", "abbrev", "team", "W", "L"]].to_string(index=False))
    print(pd.read_csv(C.DATA / "fantrax" / f"draft_slots_{C.CURRENT_SEASON + 1}.csv").to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

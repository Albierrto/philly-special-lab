"""Draft-pick values from the league's own drafts.

For every pick in the league's 2024-2026 drafts (180 picks a year, keepers are not picks) we look up what the player
actually returned that season in league points, hitters and pitchers alike, and express it as points above
replacement on the same footing the Trades and Keepers pages use: a hitter is worth what he beat the free-agent rate
by on the days he played (points minus replacement points per game x games), a starter what he beat a streamed start
by (points minus 7.2 x starts). Replacement per game is that season's 150th-best regular by points per game (12 teams x
10 hitters x 1.25), which replayed at 3.62, 3.70 and 3.54 in 2024-26 against the 3.6 a season simulation in this
league's format measured. A pick that busted counts as zero, not negative, because you drop him and stream the spot. Pooled over three drafts and smoothed into a curve
that only goes down as the pick number goes up, that is the "what picks at this slot have actually returned" half of
the pick value used on the Trades page; the other half is what the 2027 board says is left at that pick.

    python -m breakout.picks           # prints the curve
    picks.augment(data)                # adds pick_hist / pick_points / draft_slots to explorer_data
"""
from __future__ import annotations
import json, re, sys, unicodedata
import numpy as np
import pandas as pd
from . import config as C

START_REPL = 7.2  # points a streamed start scores, from the 2026 season replayed under the 10-start cap


def nk(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s or "")).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z ]", "", s); s = re.sub(r"\b(jr|sr|ii|iii)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def load_drafts() -> pd.DataFrame:
    out = []
    for p in sorted((C.DATA / "fantrax").glob("draft_*.psv")):
        y = int(p.stem.split("_")[1]); d = pd.read_csv(p, sep="|"); d["season"] = y; out.append(d)
    return pd.concat(out, ignore_index=True)


def realized(drafts: pd.DataFrame, teams: int = 12) -> pd.DataFrame:
    h = pd.read_csv(C.OUT / "v2" / "hitter_seasons_full.csv")[["name", "season", "pts", "PA", "G"]]
    p = pd.read_csv(C.OUT / "v3" / "pitcher_seasons_full.csv")[["name", "season", "pts", "GS", "IP"]]
    h["nkey"] = h["name"].map(nk); p["nkey"] = p["name"].map(nk)
    # one line per player-season (a traded player can have two rows): keep the larger
    h = h.sort_values("pts", ascending=False).drop_duplicates(["nkey", "season"])
    p = p.sort_values("pts", ascending=False).drop_duplicates(["nkey", "season"])
    repl = {}
    for y in sorted(drafts["season"].unique()):
        hy = h[(h.season == y) & (h.G >= 100)].assign(ppg=lambda x: x.pts / x.G).sort_values("ppg", ascending=False)
        repl[y] = dict(h=float(hy["ppg"].iloc[min(round(1.25 * teams * 10), len(hy)) - 1]) if len(hy) else 3.6, p=START_REPL)
    d = drafts.copy(); d["nkey"] = d["player"].map(nk)
    hm = h.set_index(["nkey", "season"])[["pts", "G"]]; pm = p.set_index(["nkey", "season"])[["pts", "GS"]]
    rows = []
    HIT = {"C", "1B", "2B", "3B", "SS", "OF", "LF", "CF", "RF", "DH", "UT", "IF", "MI", "CI"}
    for r in d.itertuples():
        pos = {x.strip() for x in str(r.pos).upper().replace(",", "/").split("/")}
        # 2024 was drafted under a format with relief slots; relief-only picks say nothing about what a slot returns in
        # today's SP-only league (a reliever cannot be started), so they are left out of the curve rather than scored as
        # a starter with zero starts, which counted his whole season as value (19 picks, 1,606 points).
        if pos & {"RP"} and not pos & {"SP", "P"} and not pos & HIT:
            continue
        is_p, is_h = bool(pos & {"SP", "P"}), bool(pos & HIT)
        k = (r.nkey, r.season); hv = hm.loc[k] if k in hm.index else None; pv = pm.loc[k] if k in pm.index else None
        h_par = float(hv["pts"] - repl[r.season]["h"] * hv["G"]) if hv is not None else None
        p_par = float(pv["pts"] - START_REPL * pv["GS"]) if pv is not None else None
        # the drafted position decides which line is his; only a two-way or unlabeled pick takes the larger. Matching by
        # name alone once handed Luis Garcia Jr. (2B) the reliever Luis Garcia's season.
        if is_p and not is_h: kind = "p"
        elif is_h and not is_p: kind = "h"
        else: kind = "p" if (p_par is not None and (h_par is None or p_par > h_par)) else "h"
        par = (p_par if kind == "p" else h_par) or 0.0
        pts = float((pv if kind == "p" else hv)["pts"]) if (pv if kind == "p" else hv) is not None else 0.0
        rows.append(dict(season=r.season, overall=int(r.overall), round=int(r.round), team=r.team, player=r.player, pos=r.pos, kind=kind,
                         pts=round(pts, 1), repl=round(repl[r.season][kind], 2), par=round(max(0.0, par), 1)))
    return pd.DataFrame(rows), repl


def curve(real: pd.DataFrame, n_picks: int = 180) -> dict:
    """Monotone-decreasing expected PAR by overall pick, pooled over seasons, then lightly smoothed."""
    from sklearn.isotonic import IsotonicRegression
    x = real["overall"].to_numpy(float); y = real["par"].to_numpy(float)
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(x, y)
    ks = np.arange(1, n_picks + 1); v = iso.predict(ks.astype(float))
    v = pd.Series(v).rolling(9, center=True, min_periods=1).mean().to_numpy()
    v = np.maximum.accumulate(v[::-1])[::-1]  # keep it monotone after smoothing
    return {int(k): round(float(val), 1) for k, val in zip(ks, v)}


def draft_slots() -> list[dict]:
    """Next year's fixed draft order. Preferred source: data/fantrax/draft_slots_<next>.csv, built by fantrax_api from the
    regular-season standings (9th-12th place pick 1-4 in that order, then 8th ... 1st). Fallback: last year's order."""
    nxt = C.DATA / "fantrax" / f"draft_slots_{C.CURRENT_SEASON + 1}.csv"
    if nxt.exists():
        df = pd.read_csv(nxt)
        return [dict(team=r.team, abbrev=r.abbrev, slot=int(r.slot), source="standings") for r in df.itertuples()]
    cfg = json.loads((C.DATA / "fantrax" / "league_config.json").read_text())
    names = cfg.get("fantasy_team_abbrevs", {}); inv = {re.sub(r"\s*\(.*\)$", "", v).strip(): k for k, v in names.items() if v not in ("?", "Free Agent", "Waivers")}
    order = cfg.get("draft_order_2026", [])
    return [dict(team=t, abbrev=inv.get(t), slot=i + 1) for i, t in enumerate(order)]


def augment(data: dict) -> dict:
    drafts = load_drafts(); real, repl = realized(drafts, teams=int(data.get("meta", {}).get("teams", 12)))
    data["pick_hist"] = curve(real)
    data["pick_points"] = json.loads(real[["season", "overall", "round", "team", "player", "kind", "pts", "par"]].to_json(orient="records"))
    data["pick_repl"] = {str(k): v for k, v in repl.items()}
    data["draft_slots"] = draft_slots()
    return data


def main(argv=None):
    drafts = load_drafts(); real, repl = realized(drafts)
    print("replacement levels:", repl)
    c = curve(real)
    for k in (1, 2, 5, 12, 13, 24, 36, 60, 96, 120, 180): print(f"pick {k:3d}: {c[k]:6.1f} PAR")
    print(real.groupby("round")["par"].mean().round(1).to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())

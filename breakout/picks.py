"""Draft-pick values from the league's own drafts.

For every pick in the league's 2024-2026 drafts (180 picks a year, keepers are not picks) we look up what the player
actually returned that season in league points, hitters and pitchers alike, and express it as points above a
replacement-level player (the best bat or arm you could have grabbed off waivers instead). A pick that busted counts
as zero, not negative, because you drop him and stream the spot. Pooled over three drafts and smoothed into a curve
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

ROSTERED_H = 13   # hitters rostered per team (10 starters + ~3 bench)
ROSTERED_P = 7    # starting pitchers rostered per team (5 starters + ~2 bench)


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
    h = pd.read_csv(C.OUT / "v2" / "hitter_seasons_full.csv")[["name", "season", "pts", "PA"]]
    p = pd.read_csv(C.OUT / "v3" / "pitcher_seasons_full.csv")[["name", "season", "pts", "GS", "IP"]]
    h["nkey"] = h["name"].map(nk); p["nkey"] = p["name"].map(nk)
    # one line per player-season (a traded player can have two rows): keep the larger
    h = h.sort_values("pts", ascending=False).drop_duplicates(["nkey", "season"])
    p = p.sort_values("pts", ascending=False).drop_duplicates(["nkey", "season"])
    repl = {}
    for y in sorted(drafts["season"].unique()):
        hy = h[h.season == y].sort_values("pts", ascending=False); py = p[(p.season == y) & (p.GS >= 5)].sort_values("pts", ascending=False)
        repl[y] = dict(h=float(hy["pts"].iloc[min(teams * ROSTERED_H, len(hy) - 1)]), p=float(py["pts"].iloc[min(teams * ROSTERED_P, len(py) - 1)]))
    d = drafts.copy(); d["nkey"] = d["player"].map(nk)
    hm = h.set_index(["nkey", "season"])["pts"]; pm = p.set_index(["nkey", "season"])["pts"]
    rows = []
    for r in d.itertuples():
        is_p = str(r.pos).upper() in ("SP", "P", "RP") or "SP" in str(r.pos).upper().split("/")
        hp = hm.get((r.nkey, r.season)); pp = pm.get((r.nkey, r.season))
        # two-way or ambiguous: take the larger of his hitting / pitching lines
        pts = max([v for v in (hp, pp) if v is not None and not np.isnan(v)] or [0.0])
        kind = "p" if (is_p or (pp is not None and (hp is None or pp > hp))) else "h"
        rp = repl[r.season][kind]
        rows.append(dict(season=r.season, overall=int(r.overall), round=int(r.round), team=r.team, player=r.player, pos=r.pos, kind=kind,
                         pts=round(float(pts), 1), repl=round(rp, 1), par=round(max(0.0, float(pts) - rp), 1)))
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

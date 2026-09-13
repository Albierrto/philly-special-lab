"""Injured-list history from the MLB Stats API transactions feed.

For every hitter-season: IL stints, IL days (placement -> activation, or season end), 60-day flag, and the injury
text MLB publishes ("Left hamstring strain"). Rolled into durability features the projection can use.
"""
from __future__ import annotations
import json, re
from pathlib import Path
import pandas as pd
import requests

from .config import DATA

_S = requests.Session(); _S.headers.update({"User-Agent": "Mozilla/5.0"})
SEASON_END = {2021: "2021-10-03", 2022: "2022-10-05", 2023: "2023-10-01", 2024: "2024-09-29", 2025: "2025-09-28", 2026: "2026-09-27"}


def transactions(year: int, refresh=False) -> list[dict]:
    p = DATA / "injuries" / f"transactions_{year}.json"
    if p.exists() and not refresh:
        return json.loads(p.read_text())
    u = f"https://statsapi.mlb.com/api/v1/transactions?startDate={year}-02-15&endDate={year}-10-05&sportId=1"
    tr = _S.get(u, timeout=180).json().get("transactions", [])
    p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(tr))
    return tr


def il_table(year: int) -> pd.DataFrame:
    """One row per IL stint for `year`."""
    rows = []
    placements, activations = {}, {}
    for t in transactions(year):
        if t.get("typeDesc") != "Status Change":
            continue
        d = (t.get("description") or "")
        pid = (t.get("person") or {}).get("id")
        if pid is None:
            continue
        dl = d.lower()
        if "injured list" in dl and " placed " in dl:
            m = re.search(r"(\d+)-day injured list", dl)
            kind = int(m.group(1)) if m else 10
            reason = d.split("injured list.")[-1].strip().rstrip(".") if "injured list." in d else ""
            placements.setdefault(pid, []).append((t["date"], kind, reason))
        elif "activated" in dl and "injured list" in dl:
            activations.setdefault(pid, []).append(t["date"])
    end = pd.Timestamp(SEASON_END.get(year, f"{year}-10-01"))
    for pid, pl in placements.items():
        acts = sorted(pd.Timestamp(a) for a in activations.get(pid, []))
        cur_stop = None
        for date, kind, reason in sorted(pl):
            start = pd.Timestamp(date)
            if cur_stop is not None and start <= cur_stop:      # transfer 10-day -> 60-day: same stint
                if kind == 60 and rows:
                    rows[-1]["il_kind"] = 60
                continue
            nxt = [a for a in acts if a > start]
            stop = nxt[0] if nxt else end
            cur_stop = stop
            days = max(0, (min(stop, end) - start).days)
            rows.append(dict(mlbam_id=pid, season=year, il_start=date, il_kind=kind, il_days=days, il_reason=reason))
    return pd.DataFrame(rows)


def injury_features(seasons) -> pd.DataFrame:
    """Per hitter-season: il_stints, il_days, il_60, il_reasons; plus trailing 3-season totals."""
    frames = [il_table(y) for y in seasons]
    il = pd.concat([f for f in frames if len(f)], ignore_index=True)
    g = il.groupby(["mlbam_id", "season"]).agg(il_stints=("il_days", "size"), il_days=("il_days", "sum"),
                                                il_60=("il_kind", lambda s: int((s == 60).any())),
                                                il_reasons=("il_reason", lambda s: "; ".join(x for x in s if x)[:160])).reset_index()
    # trailing windows (this season and two before)
    g = g.sort_values(["mlbam_id", "season"])
    full = []
    for pid, d in g.groupby("mlbam_id"):
        d = d.set_index("season").reindex(range(min(seasons), max(seasons) + 1)).fillna({"il_stints": 0, "il_days": 0, "il_60": 0})
        d["il_days_3yr"] = d["il_days"].rolling(3, min_periods=1).sum()
        d["il_stints_3yr"] = d["il_stints"].rolling(3, min_periods=1).sum()
        d["mlbam_id"] = pid
        full.append(d.reset_index().rename(columns={"index": "season"}))
    out = pd.concat(full, ignore_index=True)
    out = out[out["il_stints"].notna()]
    out["il_reasons"] = out["il_reasons"].fillna("")
    return out[["mlbam_id", "season", "il_stints", "il_days", "il_60", "il_reasons", "il_days_3yr", "il_stints_3yr"]]

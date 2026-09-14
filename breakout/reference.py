"""The two lists no API gives us: PitcherList's weekly starter tiers and CBS (Scott White) sleeper / two-start lists.

Everything else the lab uses comes from an API every morning. These two are written by people, so a scheduled Claude
session reads the week's articles and drops a dated CSV in data/reference/; the pipeline always picks up the newest file
of each kind, and if one is missing or stale it simply carries on without it.

    python -m breakout.reference                     # what is on file, and how old
    python -m breakout.reference --check new.csv     # validate a file the scheduled task just wrote

File names (the date is the day the list was published, not the day it was fetched):
    data/reference/pitcherlist_tiers_<YYYY-MM-DD>.csv   pitcher,team,tier,note
    data/reference/cbs_week<NN>_<YYYY-MM-DD>.csv        player,list,tier_or_rank,note
"""
from __future__ import annotations
import re, sys
from datetime import date
import pandas as pd
from . import config as C

REF = C.DATA / "reference"
STALE_DAYS = 9            # a weekly list older than this is not worth showing
TIERS = ["Auto", "Solid", "Tobys & Bombs", "This Week Streamer", "Desperate Streamer", "Later Streamer", "Do Not Start", "Not listed"]
LISTS = ["sleeper_hitter", "sleeper_pitcher", "two_start"]
SCHEMA = {"pitcherlist_tiers": ["pitcher", "team", "tier", "note"], "cbs_week": ["player", "list", "tier_or_rank", "note"]}


def _dated(prefix: str) -> list[tuple[str, "pd.Series"]]:
    out = []
    for p in REF.glob(f"{prefix}*.csv"):
        m = re.search(r"(\d{4}-\d{2}-\d{2})", p.name)
        if m: out.append((m.group(1), p))
    return sorted(out)


def newest(prefix: str, cols: list[str] | None = None, stale_days: int = STALE_DAYS, asof: str | None = None) -> tuple[pd.DataFrame, str | None]:
    """The most recent list of this kind, or an empty frame if there is none or the newest is too old to trust."""
    cols = cols or SCHEMA.get(prefix, [])
    got = _dated(prefix)
    if not got: return pd.DataFrame(columns=cols), None
    stamp, path = got[-1]
    age = (date.fromisoformat(asof or date.today().isoformat()) - date.fromisoformat(stamp)).days
    if age > stale_days: return pd.DataFrame(columns=cols), None
    d = pd.read_csv(path)
    for c in cols:
        if c not in d.columns: d[c] = None
    return d[cols], stamp


def check(path: str) -> int:
    """Validate a freshly written list. Prints what is wrong and returns non-zero, so the writer can fix and retry."""
    p = C.ROOT / path if not str(path).startswith("/") else path
    name = str(p).rsplit("/", 1)[-1]
    prefix = "pitcherlist_tiers" if name.startswith("pitcherlist_tiers") else "cbs_week" if name.startswith("cbs_week") else None
    if prefix is None:
        print(f"FAIL name must start with pitcherlist_tiers_ or cbs_week<NN>_ : {name}"); return 1
    if not re.search(r"\d{4}-\d{2}-\d{2}\.csv$", name):
        print(f"FAIL name must end with _<YYYY-MM-DD>.csv : {name}"); return 1
    d = pd.read_csv(p); bad = []
    want = SCHEMA[prefix]
    if list(d.columns)[:len(want)] != want: bad.append(f"columns must be {want}, got {list(d.columns)}")
    if len(d) < 20: bad.append(f"only {len(d)} rows; a real weekly list has far more")
    if prefix == "pitcherlist_tiers":
        odd = sorted(set(d["tier"].dropna()) - set(TIERS))
        if odd: bad.append(f"unknown tiers {odd}; allowed: {TIERS}")
    else:
        odd = sorted(set(d["list"].dropna()) - set(LISTS))
        if odd: bad.append(f"unknown list values {odd}; allowed: {LISTS}")
        if not len(d[d["list"] == "two_start"]): bad.append("no two_start rows")
    dup = d.duplicated(subset=[want[0], want[1]]).sum()
    if dup: bad.append(f"{dup} duplicate rows")
    for b in bad: print("FAIL", b)
    if not bad: print(f"OK {name}: {len(d)} rows, {d[want[1]].nunique()} distinct {want[1]} values")
    return 1 if bad else 0


def status(asof: str | None = None) -> pd.DataFrame:
    today = date.fromisoformat(asof or date.today().isoformat()); rows = []
    for prefix in SCHEMA:
        got = _dated(prefix)
        stamp = got[-1][0] if got else None
        age = (today - date.fromisoformat(stamp)).days if stamp else None
        rows.append(dict(list=prefix, newest=stamp, age_days=age, files=len(got),
                         used=bool(stamp and age is not None and age <= STALE_DAYS)))
    return pd.DataFrame(rows)


def main(argv=None):
    argv = argv or sys.argv[1:]
    if argv and argv[0] == "--check":
        return max(check(p) for p in argv[1:])
    print(status().to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

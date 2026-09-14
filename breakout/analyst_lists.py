"""Fetch the two hand-written analyst lists automatically, inside the daily GitHub Action.

Everything else the lab uses comes from an API. These two are prose written by people — PitcherList's weekly starter
tiers and Scott White's CBS sleeper / two-start columns — so until now a person had to read them and drop a CSV in
data/reference/. This module does that with one Anthropic API call using the server-side web search tool, so the whole
site refreshes without anybody's browser being open.

    python -m breakout.analyst_lists                 # fetch whatever is stale, validate, write
    python -m breakout.analyst_lists --force         # fetch even if what is on file is still fresh
    python -m breakout.analyst_lists --dry-run       # fetch and validate, write nothing

Needs ANTHROPIC_API_KEY in the environment (a repo secret in Actions). With no key it prints one line and exits 0, so a
build never fails over a missing optional overlay — a stale or absent list just leaves those columns blank on the site.

Everything it writes goes through reference.check() first. A file that does not validate is thrown away, not committed:
a wrong list is worse than no list, because the site would show it as this week's advice.
"""
from __future__ import annotations
import argparse, csv, io, os, re, shutil, sys, tempfile
from pathlib import Path
from datetime import date, timedelta
import pandas as pd
from . import config as C
from . import reference as REF

MODEL = os.environ.get("ANALYST_MODEL", "claude-sonnet-4-5")
API = "https://api.anthropic.com/v1/messages"
MAX_SEARCHES = 8

SPEC = {
    "pitcherlist_tiers": dict(
        cols=REF.SCHEMA["pitcherlist_tiers"],
        what="PitcherList's weekly starting-pitcher streamer tiers",
        find=("Search pitcherlist.com for this week's starting pitcher streamer ranks or tiers article "
              "(they publish it most Mondays or Tuesdays, often titled around 'SP Streamer Ranks' or 'Starting Pitcher Rankings'). "
              "Read the article itself, not a summary of it."),
        rules=("One row per pitcher listed. `team` is the MLB abbreviation (ARI, ATL, ATH, BAL, BOS, CHC, CWS, CIN, CLE, COL, "
               "DET, HOU, KC, LAA, LAD, MIA, MIL, MIN, NYM, NYY, PHI, PIT, SD, SEA, SF, STL, TB, TEX, TOR, WSH). "
               f"`tier` must be exactly one of: {REF.TIERS}. `note` is the article's own short reason, or empty."),
    ),
    "cbs_week": dict(
        cols=REF.SCHEMA["cbs_week"],
        what="Scott White's CBS Sports weekly fantasy baseball sleeper hitters, sleeper pitchers and two-start pitchers",
        find=("Search cbssports.com for Scott White's fantasy baseball columns for the coming week: the sleeper hitters, "
              "the sleeper pitchers, and the two-start pitcher rankings. They publish Monday for the week ahead."),
        rules=("`list` must be exactly one of sleeper_hitter, sleeper_pitcher, two_start. For the sleeper lists "
               "`tier_or_rank` is the rank within that list as a number (1, 2, 3 ...). For two_start it is the verdict text "
               "CBS gives, such as 'Advisable in most cases', 'No thanks', 'Better left for points leagues'. "
               "`note` is a short reason from the article, or empty. There must be two_start rows."),
    ),
}

PROMPT = """You are filling in one CSV for a fantasy baseball site. Today is {today}.

Fetch: {what}.

{find}

Return ONLY a CSV — no prose before or after, no markdown fence. First line is exactly this header:
{header}

{rules}

Hard rules, these matter more than completeness:
- Every row must come from a page you actually read this session. Do not add players from memory, do not fill gaps with
  your own judgment, and do not carry over last week's list.
- If you cannot reach the article, or it is paywalled, return the single word NONE and nothing else. A short honest list
  is fine; an invented one is not, because this goes straight onto the site as this week's advice.
- Quote fields containing commas. No duplicate rows for the same {key}.
- On the last line, after the CSV, add one comment line: `# published <YYYY-MM-DD>` with the date the article itself
  carries. If you cannot tell, use the Monday of the current week.
"""


def _call(prompt: str, key: str) -> str:
    import requests
    body = dict(model=MODEL, max_tokens=8000,
                tools=[dict(type="web_search_20250305", name="web_search", max_uses=MAX_SEARCHES)],
                messages=[dict(role="user", content=prompt)])
    r = requests.post(API, timeout=300, json=body,
                      headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    r.raise_for_status()
    return "".join(b.get("text", "") for b in r.json().get("content", []) if b.get("type") == "text").strip()


def _parse(text: str, cols: list[str]) -> tuple[pd.DataFrame | None, str | None]:
    """Pull the CSV and the published date out of the reply. Returns (None, None) when the model said NONE."""
    if not text or text.strip().upper().startswith("NONE"):
        return None, None
    text = re.sub(r"^```[a-z]*\n|```$", "", text.strip(), flags=re.M)
    pub = None
    m = re.search(r"^#\s*published\s+(\d{4}-\d{2}-\d{2})", text, flags=re.M)
    if m: pub = m.group(1)
    body = "\n".join(l for l in text.splitlines() if l.strip() and not l.lstrip().startswith("#"))
    start = body.find(cols[0])
    if start < 0: return None, None
    try:
        d = pd.read_csv(io.StringIO(body[start:]), dtype=str, keep_default_na=False)
    except Exception as e:
        print(f"  could not parse the reply as CSV: {e}"); return None, None
    d.columns = [c.strip() for c in d.columns]
    for c in cols:
        if c not in d.columns: d[c] = ""
    return d[cols].map(lambda v: str(v).strip()), pub


def fetch(prefix: str, key: str, today: str) -> tuple[pd.DataFrame | None, str | None]:
    s = SPEC[prefix]
    p = PROMPT.format(today=today, what=s["what"], find=s["find"], rules=s["rules"],
                      header=",".join(s["cols"]), key=s["cols"][0])
    try:
        txt = _call(p, key)
    except Exception as e:
        print(f"  {prefix}: the API call failed ({e}); leaving what is on file alone"); return None, None
    return _parse(txt, s["cols"])


def _monday(iso: str) -> date:
    d = date.fromisoformat(iso); return d - timedelta(days=d.weekday())


def _name(prefix: str, pub: str) -> str:
    if prefix == "pitcherlist_tiers":
        return f"pitcherlist_tiers_{pub}.csv"
    # continue the week numbering already on disk rather than guessing the season's opening Monday, which was off by one
    got = REF._dated("cbs_week")
    if got:
        stamp, path = got[-1]
        m = re.search(r"cbs_week(\d+)_", path.name)
        if m:
            n = int(m.group(1)) + max(0, (_monday(pub) - _monday(stamp)).days // 7)
            return f"cbs_week{max(1, n):02d}_{pub}.csv"
    d = date.fromisoformat(pub)
    return f"cbs_week{max(1, ((d - date(d.year, 3, 23)).days // 7) + 1):02d}_{pub}.csv"


def run(force: bool = False, dry: bool = False, asof: str | None = None) -> int:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    today = asof or date.today().isoformat()
    if not key:
        print("analyst lists: no ANTHROPIC_API_KEY, skipping (the site carries on with whatever is on file)")
        return 0
    REF.REF.mkdir(parents=True, exist_ok=True)
    wrote = 0
    for prefix in SPEC:
        have, stamp = REF.newest(prefix, asof=today)
        age = (date.fromisoformat(today) - date.fromisoformat(stamp)).days if stamp else None
        if not force and stamp and age is not None and age <= 5:
            print(f"  {prefix}: on file from {stamp} ({age}d old), still current — skipping"); continue
        print(f"  {prefix}: newest on file {stamp or 'none'}{f' ({age}d)' if age is not None else ''} — fetching")
        d, pub = fetch(prefix, key, today)
        if d is None or not len(d):
            print(f"  {prefix}: nothing usable came back; leaving what is on file alone"); continue
        pub = pub or (date.fromisoformat(today) - timedelta(days=date.fromisoformat(today).weekday())).isoformat()
        out = REF.REF / _name(prefix, pub)
        with tempfile.TemporaryDirectory() as td:          # validate under the real filename; check() insists on it
            tmp = Path(td) / out.name
            d.to_csv(tmp, index=False, quoting=csv.QUOTE_MINIMAL)
            if REF.check(str(tmp)) != 0:
                print(f"  {prefix}: the fetched list did not validate — throwing it away rather than publishing it")
                continue
            if dry:
                print(f"  {prefix}: --dry-run, not writing {out.name}"); continue
            shutil.copyfile(tmp, out)
        wrote += 1
        print(f"  {prefix}: wrote {out.name} ({len(d)} rows)")
    print(REF.status(asof=today).to_string(index=False))
    return 0 if wrote or True else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="fetch even when what is on file is still current")
    ap.add_argument("--dry-run", action="store_true", help="fetch and validate, write nothing")
    ap.add_argument("--asof", default=None)
    a = ap.parse_args(argv)
    return run(force=a.force, dry=a.dry_run, asof=a.asof)


if __name__ == "__main__":
    sys.exit(main())

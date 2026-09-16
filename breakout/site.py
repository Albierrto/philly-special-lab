"""Build the standalone website into ./site: index.html + data/*.js + live.js, plus the repo scaffolding
(GitHub Actions daily refresh, README, requirements).   python -m breakout.site [--default-team BB|'']

The site is static: it runs anywhere that serves files (GitHub Pages, Netlify, Cloudflare Pages, a folder on your PC).
Live pieces (posted probables, lineups, scores, first-pitch weather) are fetched in the browser straight from MLB and
Open-Meteo, which both allow cross-origin requests; the models and ownership tables come from the daily build.
"""
from __future__ import annotations
import argparse, json, shutil, sys, time
import pandas as pd
from pathlib import Path
from . import config as C
from . import picks, ownership
from .assemble import HD_COLS, PS_COLS, G_COLS, columnar

ROOT = Path(__file__).resolve().parents[1]

WORKFLOW = r'''name: refresh
on:
  schedule:
    # three times a day, and deliberately NOT on the hour: GitHub sheds scheduled runs at :00 under load, and the
    # single 10:00 slot silently never fired at all on 2026-09-14, leaving the site a day behind on ownership.
    # 6:17 am / 12:17 pm / 6:17 pm Eastern — a pickup is live within about six hours instead of up to a day.
    - cron: "17 10 * * *"
    - cron: "17 16 * * *"
    - cron: "17 22 * * *"
    - cron: "17 8 * * 1"      # Mondays: the same refresh, but rebuild the season models first (see below)
  workflow_dispatch:
    inputs:
      full:
        description: "Rebuild the season models too (re-downloads MLB/Savant/MiLB caches, ~2-3 min)"
        default: "false"
permissions:
  contents: write
  pages: write
  id-token: write
concurrency: refresh
jobs:
  refresh:
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11", cache: "pip" }
      - run: pip install -r requirements.txt
      # Projections, Breakout Index, keeper values and prospect cards come from here, and they used to move only when
      # someone remembered to run this by hand — so keeper and trade calls were being made off frozen models. Now it
      # runs itself every Monday morning, and still on demand with full=true.
      - name: Full model rebuild (Mondays, or on demand)
        if: ${{ github.event.inputs.full == 'true' || github.event.schedule == '17 8 * * 1' }}
        timeout-minutes: 60     # it takes ~2 min from a cold runner; this is only a runaway guard
        run: python -m breakout.pipeline3
      # The two lists no API gives us. One Anthropic API call with web search reads this week's PitcherList tiers and
      # Scott White's CBS columns and writes a validated CSV; without the secret it prints one line and does nothing,
      # and a list that fails validation is thrown away rather than published, so this can never fail the build.
      - name: Analyst lists (PitcherList tiers, CBS sleepers and two-starts)
        continue-on-error: true
        env: { ANTHROPIC_API_KEY: "${{ secrets.ANTHROPIC_API_KEY }}" }
        run: python -m breakout.analyst_lists
      - name: Fantrax rosters, standings and draft order (read-only, official API)
        run: python -m breakout.fantrax_api
      - name: This week's streamers (schedule, probables, rosters, weather, splits)
        run: python -m breakout.pipeline_streamers
      # Keep score of the forecasts. Archives the day's projections while the games are still ahead (a later run
      # after first pitch leaves the file alone, so nothing is ever graded against a number written after the fact),
      # then grades every archived day whose box scores are in.
      - name: Forecast scorecard
        continue-on-error: true
        run: python -m breakout.scorecard
      - name: Assemble the site
        run: |
          python -m breakout.build_v4
          python -m breakout.assemble
          python -m breakout.site --default-team ""
      - name: Commit
        run: |
          git config user.name "lab-bot"
          git config user.email "lab-bot@users.noreply.github.com"
          git add -A site output data/fantrax data/statcast data/reference data/forecasts data/scorecard
          [ -d data/availability ] && git add -A data/availability
          git commit -m "refresh $(date -u +%F)" || echo "nothing to commit"
          # someone (or another run) can land a commit on main while this one is building, and a plain push then dies
          # with exit 128 and takes the whole refresh down with it. Rebase onto whatever arrived and try again.
          for i in 1 2 3; do
            git push && break
            echo "push rejected, rebasing onto origin/main (attempt $i)"
            git pull --rebase --autostash origin main || exit 1
            sleep 3
          done
      # publish the site folder to GitHub Pages (a commit made by the workflow does not trigger other workflows)
      - uses: actions/configure-pages@v5
      - uses: actions/upload-pages-artifact@v3
        with: { path: site }
      - id: deployment
        uses: actions/deploy-pages@v4
'''

PAGES_WORKFLOW = r'''name: pages
# Publishes ./site to GitHub Pages whenever the site changes on main (and on demand). Settings -> Pages -> Source must be "GitHub Actions".
on:
  push:
    branches: [main]
    paths: ["site/**"]
  workflow_dispatch:
permissions:
  contents: read
  pages: write
  id-token: write
concurrency:
  group: pages
  cancel-in-progress: true
jobs:
  deploy:
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/configure-pages@v5
      - uses: actions/upload-pages-artifact@v3
        with: { path: site }
      - id: deployment
        uses: actions/deploy-pages@v4
'''

README = r'''# Philly Special Lab

A fantasy-baseball site for a 12-team Fantrax H2H points keeper league. Everything is scored in the league's own points.

**What is on the site:** dashboard (your lineup by day, parks and weather, pickups, keeper board, breakout watch),
hitter and starting-pitcher labs with Statcast, Stuff+/Location+/Pitching+ proxies, career-relative Breakout Indexes,
a Streamers tab (matchups, park factors, first-pitch weather, two-start pitchers, the 10-start cap), keeper boards and
draft pools, the league's real drafts, TJStats-style prospect cards, and a Methods page.

**Live in the browser** (no server): posted probable pitchers, game states and scores, today's lineups once they are
posted, and first-pitch weather — refreshed every 10 minutes from the MLB Stats API and Open-Meteo. Click the LIVE dot
to refresh now.

**Daily build (GitHub Actions, 6 am ET):** re-runs the streamer pipeline (rotations, rosters, IL, splits, park factors,
forecasts) and republishes the site. "Run workflow" with `full = true` rebuilds the season models (projections,
Breakout Index, keeper values, prospect cards) — do that after the season or whenever you want the models refreshed.

## Put it online (GitHub Pages, free)

1. Create a GitHub account if you do not have one, then a new repository (public is fine; the data is public data).
2. Upload this whole folder (drag and drop on the repo page works: "Add file → Upload files"), or `git push` it.
3. Repository **Settings → Pages → Build and deployment → Source: GitHub Actions**.
4. **Settings → Actions → General → Workflow permissions → Read and write** so the daily refresh can commit.
5. Actions tab → **pages** → Run workflow. Your site is at `https://<your-username>.github.io/<repo-name>/` a minute later. Share that link.
   The refresh runs three times a day and republishes it; the Actions tab shows each run.
6. Optional, for the two analyst lists: **Settings → Secrets and variables → Actions → New repository secret**, name
   `ANTHROPIC_API_KEY`, value a key from console.anthropic.com. With it, the refresh reads this week's PitcherList
   tiers and Scott White's CBS columns itself (one call a week, web search on) and drops a validated CSV in
   `data/reference/`. Without it that step prints one line and does nothing, and the site simply leaves those columns
   blank — nothing else changes.

## What runs by itself

| | when (UTC) |
|---|---|
| Fantrax rosters, IL, standings, draft order; streamers, probables, splits, park, weather | 10:17, 16:17, 22:17 |
| Season models: projections, Breakout Index, keeper values, prospect cards | Mondays 08:17 |
| Analyst lists, if `ANTHROPIC_API_KEY` is set | with each refresh, fetched only when what is on file has aged |
| Posted lineups, live probables, scores, first-pitch weather | in the browser, every 10 minutes |

Not on the hour on purpose: GitHub sheds scheduled runs at `:00` under load, and a single 10:00 slot silently never
fired at all, which left the site a day behind on ownership without anything saying so.

Fantrax rosters (who owns whom, IR/minors status), standings and next year's draft order are pulled every morning through
Fantrax's official read-only API (`python -m breakout.fantrax_api`, no login needed because the league allows API reads).
Nothing in this repo can touch a roster, make a claim, propose a trade or post in the league.
Ownership everywhere on the site (Today, My team, keepers, trades, streamers) comes from that morning's sync
(`breakout/ownership.py` stamps it on top of the projection tables at build time), so a drop or a pickup shows up the
next morning, or right away if you run the **refresh** workflow by hand. The My team page shows the sync time.

## Run it locally

```
pip install -r requirements.txt
python -m breakout.pipeline_streamers      # this week's matchups (5–8 minutes; talks to MLB, Savant, Open-Meteo)
python -m breakout.build_v4 && python -m breakout.assemble && python -m breakout.site
open site/index.html
```

`python -m breakout.pipeline3` rebuilds the season models (25 minutes with a warm cache).
'''

REQS = "pandas>=2.2\nnumpy>=1.26\nrequests>=2.31\nscikit-learn>=1.4\npyarrow>=15\nscipy>=1.11\n"
GITIGNORE = """__pycache__/
*.pyc
.DS_Store
# raw caches the pipelines re-download when missing (keeps the repo small)
data/milb_statcast/
data/mlb/
data/savant/
# the arsenal leaderboards are hand-downloaded from Savant and have no fetch path in the code, so unlike every other
# cache here they cannot rebuild themselves on a fresh runner — they have to travel with the repo
!data/savant/arsenal/
!data/savant/arsenal/*.csv
data/milb/
data/injuries/transactions_*.json
# generated pages (the site/ folder is the one that is served)
output/*/philly_special*.html
"""


def build(default_team: str):
    site = ROOT / "site"; (site / "data").mkdir(parents=True, exist_ok=True)
    tpl = (C.OUT / "v4" / "explorer_template.html").read_text(encoding="utf-8")
    data = ownership.apply(picks.augment(json.loads((C.OUT / "v3" / "explorer_data.json").read_text())))
    cfg = json.loads((C.DATA / "fantrax" / "league_config.json").read_text()); data["meta"]["team_names"] = cfg.get("fantasy_team_abbrevs", {})
    data["meta"]["schedule"] = cfg.get("schedule", {})                                        # periods + head-to-head, for the matchup simulator
    data["meta"]["standings"] = cfg.get(f"standings_{data['meta']['season']}", [])            # season pace, to sanity-check it
    sc = C.DATA / "scorecard" / "summary.csv"
    data["scorecard"] = pd.read_csv(sc).to_dict(orient="records") if sc.exists() else []       # how the forecasts have actually done
    data["meta"]["default_team"] = default_team
    s = ownership.stamp_streamers(json.loads((C.OUT / "streamers" / "streamers.json").read_text()))
    stream = dict(meta=s["meta"], hitter_days=columnar(s["hitter_days"], HD_COLS), pitcher_starts=columnar(s["pitcher_starts"], PS_COLS), games=columnar(s["games"], G_COLS),
                  park_factors=s["park_factors"], venues=s.get("venues", []), logs=s.get("logs", {}))
    (site / "data" / "lab.js").write_text("window.__LAB__=" + json.dumps(data, separators=(",", ":")) + ";", encoding="utf-8")
    (site / "data" / "streamers.js").write_text("window.__STREAM__=" + json.dumps(stream, separators=(",", ":")) + ";", encoding="utf-8")
    # site index: data comes from the two script files; live layer after boot
    ver = time.strftime("%Y%m%dT%H%M")  # cache-buster: browsers fetch fresh data after every build
    html = tpl.replace('<script id="data" type="application/json">__DATA__</script>', f'<script src="data/lab.js?v={ver}"></script>\n<script src="data/streamers.js?v={ver}"></script>')
    html = html.replace("const D = JSON.parse(document.getElementById('data').textContent);", "const D = window.__LAB__; D.streamers = window.__STREAM__;")
    html = html.replace("<title>Philly Special Hitter Lab</title>", "<title>Philly Special Lab</title>")
    page = ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
            "<meta name=\"description\" content=\"Philly Special fantasy baseball lab: lineups, streamers, keepers, breakouts and prospect cards in the league's points.\">\n"
            "<link rel=\"icon\" href=\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Cpath d='M4 4h24v12L16 30 4 16z' fill='%23E5484D'/%3E%3Cpath d='M8 8h16v7l-8 10-8-10z' fill='%23fff' opacity='.92'/%3E%3C/svg%3E\">\n"
            "<style>body{margin:0;color-scheme:dark light}img{max-width:100%}[hidden]{display:none!important}</style>\n</head>\n<body>\n" + html + f"\n<script src=\"live.js?v={ver}\"></script>\n</body>\n</html>\n")
    (site / "index.html").write_text(page, encoding="utf-8")
    shutil.copy(Path(__file__).with_name("site_live.js"), site / "live.js")
    (site / ".nojekyll").write_text("")
    # repo scaffolding
    wf = ROOT / ".github" / "workflows"; wf.mkdir(parents=True, exist_ok=True); (wf / "refresh.yml").write_text(WORKFLOW); (wf / "pages.yml").write_text(PAGES_WORKFLOW)
    (ROOT / "README.md").write_text(README); (ROOT / "requirements.txt").write_text(REQS); (ROOT / ".gitignore").write_text(GITIGNORE)
    print("site ->", site, "index", round((site / "index.html").stat().st_size / 1e6, 2), "MB; data",
          round(sum(p.stat().st_size for p in (site / "data").iterdir()) / 1e6, 2), "MB")


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--default-team", default="")
    a = ap.parse_args(argv); build(a.default_team); return 0


if __name__ == "__main__":
    sys.exit(main())

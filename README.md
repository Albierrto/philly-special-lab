# Philly Special Lab

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
   The daily refresh republishes it every morning; the Actions tab shows each run.

Fantrax ownership (who owns whom) cannot be pulled by the daily job because it needs your Fantrax login; it is refreshed
from the CSVs in `data/fantrax/` whenever you update them (ask Claude to re-export them from your Chrome session, or
export the Players page yourself).

## Run it locally

```
pip install -r requirements.txt
python -m breakout.pipeline_streamers      # this week's matchups (5–8 minutes; talks to MLB, Savant, Open-Meteo)
python -m breakout.build_v4 && python -m breakout.assemble && python -m breakout.site
open site/index.html
```

`python -m breakout.pipeline3` rebuilds the season models (25 minutes with a warm cache).

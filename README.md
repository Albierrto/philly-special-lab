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

## Home Run Board (separate page)

`site/hr/` is a stand-alone page, deliberately not linked from the lab: who is most likely to homer today, hits and
total bases, starter strikeouts and game odds, each next to DraftKings (ESPN's public odds feed) and Kalshi (public
market data), plus a scorecard that grades every board once the games are played. Code in `hrboard/`
(`python -m hrboard.build` for the daily board, `python -m hrboard.train` to refit and re-test the model), data in
`data/hrboard/`. It rebuilds with every refresh and on its own `hrboard` workflow five more times a day; lineups,
DraftKings lines, weather and home runs update live in the browser.

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

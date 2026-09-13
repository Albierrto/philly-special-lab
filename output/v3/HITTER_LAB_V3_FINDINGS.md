# Philly Special Lab v3 — pitchers, prospect cards, real keepers

Third pass on the Philly Special breakout project. What changed since v2: your real 2026 keepers are wired in (Kurtz, Tatis, Julio, Witt, Wood as hitters; Woo and Ohtani as pitchers, with Ohtani treated as a pitcher-slot keeper whose hitting projection rides along), a full starting-pitcher module built around Pitching+/Stuff+/Location+ style proxies, and TJStats-style prospect cards with minor-league Statcast pulled from Savant's Triple-A tracking. Everything lives in the same explorer artifact (Pitchers tab, Keepers tab now has pitcher boards, Prospects tab opens cards) and in the `output/v3/` CSVs.

## Keepers, with the real 2026 list

Your hitter board for 2027 (KSV = projected 2027 points minus the (60+12)th-best projected hitter, i.e. what you would get with your first pick):

| # | Hitter | Age | 2026 pts | Proj 2027 | KSV | Tier |
|---|---|---|---|---|---|---|
| 1 | Bobby Witt Jr. (K'26) | 26.0 | 684 | 770 | +259 | lock |
| 2 | Fernando Tatis Jr. (K'26) | 27.5 | 750 | 695 | +184 | lock |
| 3 | James Wood (K'26) | 23.8 | 716 | 676 | +165 | lock |
| 4 | Julio Rodríguez (K'26) | 25.5 | 598 | 595 | +84 | clear keep |
| 5 | Sal Stewart | 22.6 | 716 | 566 | +55 | clear keep |
| 6 | Nick Kurtz (K'26) | 23.3 | 491 | 497 | −14 | let go |
| 7 | Brandon Lowe | 32.0 | 619 | 492 | −19 | let go |

Kurtz's 2026 (491 points in 434 PA, 82 IL days over three years) is the one real decision: the projection says his rate is fine (proj/600 is top-30) but the PA estimate is what drags him under replacement. If you believe he plays 150 games, he is a keeper; the model does not. Sal Stewart's rookie year (716 points, no IL history) makes him the fifth hitter as the model sees it.

Pitcher board (value for a pitcher-slot keeper = pitching projection + hitting projection, so Ohtani's two-way total counts):

| # | Pitcher | Age | 2026 GS | 2026 pts (pitching) | Proj pitching | + hitting | Total | KSV | Tier |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Shohei Ohtani (K'26) | 32.0 | 14 | 212 | 157 | 711 | 868 | +621 | lock |
| 2 | Dylan Cease | 30.5 | 27 | 423 | 379 | | 379 | +132 | lock |
| 3 | Logan Gilbert | 29.2 | 29 | 340 | 315 | | 315 | +68 | clear keep |
| 4 | Bryan Woo (K'26) | 26.4 | 28 | 319 | 315 | | 315 | +68 | clear keep |
| 5 | Payton Tolle | 23.7 | 24 | 296 | 271 | | 271 | +24 | marginal |

Ohtani is an obvious pitcher keeper in this league: keeping him as a pitcher costs a pitcher slot but returns a top-5 hitter plus 14–18 starts. The interesting call is Cease versus Woo/Gilbert for the second slot. Cease projects ~65 points higher on a 127 Stuff+ arsenal, but he is 30.5 with a 97 Location+; Woo and Gilbert are tied at 315, Woo four years younger with the best Location+ (122) on your staff. The board takes Cease on points. Whoever you let go lands at the top of the pitcher draft pool, so the second pick comes back to you either way if nobody else grabs him at pick 12 — the pool has Gilbert, Woo, Sale (37), McLean and Tolle as the top five available arms once every team keeps its two highest-KSV starters.

Ohtani is removed from the hitter draft pool (he was leaking into it in the earlier pass) and does not occupy one of your five hitter keeper slots.

## Pitching module

Scoring verified against Fantrax: IP + K − ER + 4×QS + 10×CG + 8×SHO − 3×BS (116 of 116 pitchers with 20+ IP match exactly; no-hitters and perfect games are not in public feeds). Because there is no RP slot and a 10-start weekly cap, only starters matter, and points per start is the rate that carries over.

The plus-stats are proxies. FanGraphs' Stuff+/Location+/Pitching+ and PitcherList's PLV are pitch-level models on velocity, movement, release and location; the per-pitch inputs are not free, so the lab builds season-level proxies from Savant's leaderboard and pitch-arsenal tables and scales them 100 ± 10 like the originals:

- Stuff+ = z(whiff%) + z(in-zone swing-and-miss%) + z(arsenal put-away%) + z(fastball velocity) + ½·z(movement uniqueness) + z(arsenal run value per 100 vs hitters), scaled.
- Location+ = z(edge%) + z(first-strike%) + ½·z(zone%) − z(meatball%) − z(BB%) + z(chase%), scaled.
- Pitching+ = ridge on [Stuff+, Location+, xERA, K−BB%] fit to next-year points per start. Fitted coefficients: K−BB% +0.60, Stuff+ +0.14, Location+ +0.02, xERA −0.02. Location+ is nearly redundant once BB% is inside K−BB%.

Leave-one-season-out check (starters with 10+ GS in consecutive seasons, r against next-year points per start): Pitching+ 0.50–0.67 in every fold versus 0.31–0.56 for carrying last year's rate forward and 0.40–0.53 for xERA alone. Pitching+ was the best single predictor in all five folds.

Pitcher Breakout Index: probability that next season's points per start beats the career best (10+ GS seasons) by 1.5 with 20+ starts and a top-40 finish. Gradient-boosted classifier on the plus proxies, xERA, K−BB%, whiff, velocity, year-over-year Stuff+/velocity change, age, GS, career track and IL days. Held-out AUC 0.61–0.78; the 20 highest-BI starters broke out at a 15–20% rate against a 4–12% base rate.

Projection: points per start from a ridge on the same inputs (two seasons) plus an empirical aging step (starters improve into their mid-20s, decline gently after 30), times a durability-aware start estimate (60/40 blend of last year and the three-year mean, minus 0.1% per IL day over three years, ×0.93 at 34+, clipped 12–32).

Top 2027 projections: Cristopher Sánchez 398, Cease 379, Skenes 362, Misiorowski 356 (139.5 Stuff+, the highest in the data), Schlittler 340, then Gilbert and Woo at 315. Pitcher BI leaders with 10+ starts: Jared Jones (36%, 18 GS back from surgery, 116 Stuff+), Braxton Ashcraft (29%), Brandon Sproat (25%, free agent), Chase Burns (21%), Walbert Ureña (20%, 22.4), Roki Sasaki (20%, free agent), Payton Tolle (16%, yours). Best free-agent arms by projection: Joey Cantillo 256, Matthew Liberatore 248, Brady Singer 239, Luis Castillo 223, Bubba Chandler 213.

## Prospect cards

The Prospects tab now opens a card per bat: PAS breakdown (pedigree 35 / age-vs-level 25 / minor-league bat 25 / MLB sample 15), every 2024–2026 minor-league line colored by percentile among hitters with 100+ PA at that level and season, and a minor-league Statcast block from Savant's Triple-A tracking (avg EV, EV90, max EV, hard-hit%, barrel%, sweet-spot%, GB%, an estimated wOBA from xwOBAcon plus walks and strikeouts, K%, BB%, whiff%, chase%) with percentiles inside the tracked prospect cohort for that season. Savant's minor-league feed carries bat-tracking columns but they were empty for 2025-2026, so bat speed only appears once a prospect has an MLB sample. The MLB sample, when there is one, is the regular hitter card underneath. Statcast coverage is Triple-A parks plus a handful of lower-level parks, so most Double-A bats show lines but no tracking.

PAS leaders: Franklin Arias (81, Pipeline #3, .884 OPS at AAA at 20), Josue De Paula (81), Max Clark (80, 150 MLB PA, on CS), Leo De Vries (79), Jesús Made (79), Ethan Salas (79), Sebastian Walcott (78), Walker Jenkins (78), Theo Gillen (77), George Lombard Jr. (77, on THALER). Of those, Arias, De Paula, De Vries, Made, Salas, Walcott and Jenkins are free agents in your league and eligible for your two minors slots.

Triple-A Statcast standouts from 2026 with 150+ tracked PA (percentiles are within the 130 tracked prospects): Emmanuel Rodriguez (109.8 EV90, .354 est. wOBA, 14% barrels, but 29% K), Kevin Alcántara (109.2 EV90, 18% barrels, .355), Lazaro Montes (109.2, 57.5% hard-hit, .351, Pipeline #56), Joshua Báez (108.5, 20% barrels, .348, owned by BAMBI), Abimelec Ortiz (.369 est. wOBA with a 19% K rate), and among the pedigree names Walker Jenkins (.364 est. wOBA, 15% K, 17% whiff, the cleanest contact profile of the top-20 PAS bats) and Ralphy Velazquez (.362, 14% barrels). Franklin Arias' Triple-A block is a contact profile (88th-percentile K%, 89th whiff%) on 30th-percentile EV90, which is what a 20-year-old shortstop hitting .331 at that level usually looks like.

## Files

- Explorer: `philly_special_hitter_lab.html` (same artifact URL as v2; new Pitchers tab, pitcher keeper board and pool, prospect cards, real-keeper markers, pitching weights and presets in the league bar).
- `output/v3/hitter_projections_2027.csv`, `hitter_keeper_values.csv`, `hitter_keeper_board.csv`, `hitter_draft_pool_2027.csv`
- `output/v3/pitcher_seasons_full.csv` (every SP season 2021–2026 with the proxies), `pitcher_projections_2027.csv`, `pitcher_keeper_board.csv`, `pitcher_draft_pool_2027.csv`, `cv_pitcher_projection.csv`, `cv_pitcher_breakout_index.csv`
- `output/v3/prospects.csv`, `prospect_milb_lines.csv`, `prospect_milb_statcast.csv`
- Code: `breakout/pitchers.py`, `breakout/milb_statcast.py`, `breakout/pipeline3.py` (run `python -m breakout.pipeline3`)

## Caveats

The plus-stats are proxies, not FanGraphs' or PitcherList's numbers; they are validated only on what they predict in this league's scoring. Pitcher BI rewards low baselines (a 10-start rookie with a weak rate can score high because his bar is low), so read it with the GS and Pts/GS columns beside it. Minor-league Statcast percentiles are within the prospect cohort, not all Triple-A hitters. Other teams' keepers remain inferred; only yours are real.

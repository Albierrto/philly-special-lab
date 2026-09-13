# Breakout & ADP-value model — methodology and 2026 findings

Session: 2026-09-13. Code + data delivered as `breakout_hitters.zip`; live dashboard artifact "Philly Special Breakout Board".

## What the program does
- Pulls 2021-2026 hitter data from Baseball Savant (custom leaderboard, 196 Statcast fields), the MLB Stats API (counting + fielding stats), and the FantasyPros ADP archive (composite preseason ADP by year incl. NFBC and Fantrax columns). Also reads read-only Fantrax exports of the league's rules and 2024-2026 draft results.
- Scores every hitter-season in the league's points. Verified exact against Fantrax's 2026 table (500/500) except grand slams/cycles (not in public feeds).
- ADP value: per season, fits an expected-points-by-ADP-slot curve (rolling median + isotonic). A *beat* = +75 pts and +25 hitter ranks over that price with a top-150 finish; *big beat* = +150/+60 with a top-75 finish. Rolled up into multi-year counts and streaks.
- Breakouts: (1) an interpretable skills score = z-scores of contact quality (barrel%, hard-hit%, EV50, xwOBAcon) + bat speed (avg bat speed, fast-swing rate, ideal attack angle rate) + approach (K%, BB%, chase, whiff, swing/take RV) + sprint speed + luck (xwOBA − wOBA) + year-over-year trend (Barrel Rate Test) + age; (2) a HistGradientBoosting model projecting next-year points/PA. `breakout_score = skills + 0.02 × (market rank − projected rank)`.
- Validation: model beats naive carry-over in every held-out season (r 0.39-0.59 vs 0.34-0.52). Backtest: top-40 candidate lists built from year t hit ADP-beaters in t+1 at 22-32% vs a 14-18% base rate (~1.7x), with a positive average return vs price each year.

## Rerun
`python -m breakout run` (cached data) / `--refresh` to re-download / `python -m breakout.verify` for the checks. League settings in `data/fantrax/league_config.json`.

## Findings (2026 data)

### Multi-year ADP beaters, active in 2026 (3+ beats in 2021-2026)
| name                |   age_2026 | elig     |   seasons_evaluated |   beats |   big_beats |   busts |   current_streak |   avg_pts_over_exp |   last_pts |   last_adp |   last_final_rank |
|:--------------------|-----------:|:---------|--------------------:|--------:|------------:|--------:|-----------------:|-------------------:|-----------:|-----------:|------------------:|
| Yandy Díaz          |         35 | UT       |                   6 |       5 |           2 |       0 |                2 |                172 |        670 |        126 |                25 |
| Luis Arraez         |         29 | 2B       |                   6 |       4 |           2 |       0 |                2 |                133 |        597 |        233 |                48 |
| Steven Kwan         |         29 | OF       |                   5 |       4 |           1 |       0 |                0 |                165 |        528 |        147 |                99 |
| Brandon Nimmo       |         33 | OF       |                   6 |       4 |           2 |       0 |                0 |                130 |        563 |        130 |                74 |
| Alec Burleson       |         28 | 1B       |                   4 |       3 |           2 |       0 |                3 |                179 |        674 |        173 |                24 |
| Josh Bell           |         34 | 1B       |                   6 |       3 |           1 |       0 |                3 |                114 |        567 |        376 |                69 |
| Ceddanne Rafaela    |         26 | OF       |                   3 |       3 |           0 |       0 |                3 |                138 |        619 |        153 |                41 |
| Randy Arozarena     |         31 | OF       |                   6 |       3 |           1 |       0 |                2 |                110 |        711 |         83 |                 9 |
| Bryan Reynolds      |         31 | OF       |                   6 |       3 |           2 |       0 |                1 |                155 |        677 |        179 |                22 |
| Brandon Marsh       |         28 | OF       |                   6 |       3 |           0 |       0 |                1 |                125 |        526 |        352 |               102 |
| Willi Castro        |         29 | 2B/3B/SS |                   6 |       3 |           1 |       1 |                1 |                118 |        477 |        287 |               124 |
| Joc Pederson        |         34 | UT       |                   6 |       3 |           0 |       1 |                1 |                 69 |        438 |        nan |               149 |
| Alec Bohm           |         30 | 1B/3B    |                   6 |       3 |           1 |       1 |                1 |                 43 |        508 |        230 |               107 |
| Nathaniel Lowe      |         31 | 1B       |                   6 |       3 |           2 |       0 |                0 |                186 |        382 |        nan |               184 |
| J.P. Crawford       |         32 | 3B/SS    |                   6 |       3 |           1 |       0 |                0 |                150 |        348 |        312 |               210 |
| Taylor Ward         |         32 | OF       |                   6 |       3 |           3 |       1 |                0 |                137 |        488 |        118 |               119 |
| Andrew McCutchen    |         40 | UT       |                   6 |       3 |           0 |       1 |                0 |                134 |         55 |        686 |               495 |
| Kyle Schwarber      |         33 | UT       |                   6 |       3 |           0 |       0 |                0 |                131 |        690 |         21 |                15 |
| Ian Happ            |         32 | OF       |                   6 |       3 |           3 |       0 |                0 |                130 |        576 |        147 |                61 |
| Willy Adames        |         31 | SS       |                   6 |       3 |           1 |       0 |                0 |                121 |        453 |        106 |               136 |
| Cal Raleigh         |         30 | C        |                   6 |       3 |           0 |       1 |                0 |                 96 |        432 |         15 |               155 |
| Nico Hoerner        |         29 | 2B/SS    |                   6 |       3 |           2 |       1 |                0 |                 96 |        596 |         90 |                49 |
| Maikel Garcia       |         26 | 3B       |                   4 |       3 |           2 |       1 |                0 |                142 |        285 |         64 |               251 |
| Mike Yastrzemski    |         36 | OF       |                   6 |       3 |           0 |       0 |                0 |                 87 |        326 |        362 |               217 |
| Jeremy Peña         |         29 | SS       |                   5 |       3 |           1 |       0 |                0 |                 93 |        449 |        106 |               141 |
| Eugenio Suárez      |         35 | 3B       |                   6 |       3 |           2 |       2 |                0 |                 74 |        418 |         85 |               165 |
| Jake Cronenworth    |         32 | 2B       |                   6 |       3 |           1 |       1 |                0 |                 67 |        280 |        340 |               259 |
| Lourdes Gurriel Jr. |         33 | OF       |                   6 |       3 |           1 |       0 |                0 |                 53 |        149 |        440 |               373 |
| Austin Hays         |         31 | OF       |                   6 |       3 |           0 |       2 |                0 |                 23 |         80 |        363 |               457 |
| Christian Yelich    |         35 | UT       |                   6 |       3 |           0 |       2 |                0 |                 12 |        455 |         90 |               133 |

Pattern: this league pays 2/single, 1/walk, 3/SB with no K or CS penalty, so high-contact, high-OBP bats (Arraez, Kwan, Yandy Díaz, Nimmo, Burleson) keep returning more than their price.

### Who broke out in 2026 (career-best by 100+, top-75 finish, beat ADP curve)
| name                |   age | elig        |   PA |   pts |   prev_best_pts |   jump |   adp_hitter_rank |   final_hitter_rank |   pts_over_exp |   xwoba |   barrel_batted_rate |   avg_swing_speed |   k_percent |
|:--------------------|------:|:------------|-----:|------:|----------------:|-------:|------------------:|--------------------:|---------------:|--------:|---------------------:|------------------:|------------:|
| Sal Stewart         |  22.6 | 1B/3B       |  641 |   716 |              65 |    651 |               113 |                   6 |            215 |       0 |                 12.9 |              72.1 |        21.1 |
| Kevin McGonigle     |  21.9 | 3B/SS       |  645 |   646 |               0 |    646 |               152 |                  32 |            255 |       0 |                  6.9 |              71.2 |        13.8 |
| Carson Benge        |  23.4 | OF          |  591 |   635 |               0 |    635 |               180 |                  35 |            287 |       0 |                  7   |              71   |        22   |
| TJ Rumfield         |  26.1 | 1B          |  583 |   580 |               0 |    580 |               nan |                  58 |            427 |       0 |                  5   |              68.5 |        13.8 |
| JJ Wetherholt       |  23.8 | 2B          |  586 |   564 |               0 |    564 |               131 |                  73 |            146 |       0 |                  7.6 |              71.9 |        16.2 |
| Carter Jensen       |  23   | C           |  556 |   579 |              89 |    490 |               139 |                  59 |            188 |       0 |                 12   |              73.8 |        27.9 |
| Jake Bauers         |  30.7 | 1B/OF       |  532 |   614 |             335 |    279 |               316 |                  43 |            461 |       0 |                 13.8 |              76.5 |        25.6 |
| Jordan Walker       |  24.1 | OF          |  616 |   743 |             480 |    263 |               229 |                   4 |            438 |       0 |                 12.9 |              79.2 |        25   |
| Daylen Lile         |  23.6 | OF          |  610 |   629 |             400 |    229 |               117 |                  38 |            128 |       0 |                  9.3 |              71.4 |        19.7 |
| Wilyer Abreu        |  27   | OF          |  632 |   662 |             479 |    183 |               125 |                  28 |            237 |       0 |                 12.6 |              73.8 |        20.1 |
| Miguel Vargas       |  26.6 | 1B/3B       |  644 |   701 |             529 |    172 |               174 |                  12 |            332 |       0 |                 13.6 |              74   |        17.8 |
| Otto Lopez          |  27.7 | SS          |  631 |   705 |             557 |    148 |               165 |                  11 |            314 |       0 |                  6.1 |              71.8 |        12.8 |
| Jonathan Aranda     |  28.1 | 1B          |  625 |   604 |             457 |    147 |               124 |                  45 |            179 |       0 |                  9.8 |              70.2 |        23.5 |
| Iván Herrera        |  26.1 | C           |  666 |   634 |             491 |    143 |               103 |                  36 |            117 |       0 |                  5.5 |              73.7 |        18.3 |
| Ezequiel Duran      |  27.1 | 2B/3B/SS/OF |  546 |   573 |             440 |    133 |               nan |                  64 |            420 |       0 |                  6.3 |              72.7 |        23.5 |
| Pete Crow-Armstrong |  24.3 | OF          |  668 |   895 |             767 |    128 |                25 |                   1 |            290 |       0 |                 13.2 |              75   |        26.1 |
| Ben Rice            |  27.4 | 1B          |  608 |   679 |             560 |    119 |                46 |                  20 |            121 |       0 |                 14.4 |              72.7 |        24.5 |
| Jake McCarthy       |  28.9 | OF          |  538 |   658 |             547 |    111 |               247 |                  29 |            404 |       0 |                  5.4 |              71.4 |        16.1 |
| Dillon Dingler      |  27.8 | C           |  567 |   594 |             494 |    100 |               160 |                  52 |            203 |       0 |                 10.1 |              71.8 |        22.9 |

### 2027 breakout candidates — full-time bats (400+ PA, market rank > 40), top 20
| name                |   age | elig   |   PA |   pts |   final_hitter_rank |   adp_hitter_rank |   proj_pts |   proj_rank |   skills_score |   breakout_score | tag                                                                                    |
|:--------------------|------:|:-------|-----:|------:|--------------------:|------------------:|-----------:|------------:|---------------:|-----------------:|:---------------------------------------------------------------------------------------|
| Cam Smith           |  23.4 | OF     |  521 |   493 |                 115 |               193 |        532 |          77 |            4.9 |              5.7 | unlucky (xwOBA >> wOBA); elite bat speed; skills trending up; plus speed; young        |
| Garrett Mitchell    |  27.8 | OF     |  485 |   537 |                  95 |               286 |        464 |         127 |            4   |              3.4 | elite bat speed; elite contact quality; plus speed                                     |
| Coby Mayo           |  24.6 | 3B     |  402 |   408 |                 172 |               232 |        387 |         181 |            3.5 |              3.3 | elite bat speed; elite contact quality; skills trending up; needs playing time; young  |
| Henry Bolte         |  22.9 | OF     |  415 |   436 |                 151 |               nan |        448 |         138 |            2.9 |              3.2 | elite bat speed; elite contact quality; plus speed; needs playing time; young          |
| Jac Caglianone      |  23.4 | 1B/OF  |  511 |   566 |                  71 |               105 |        493 |         103 |            3.7 |              3.1 | elite bat speed; elite contact quality; young                                          |
| JJ Wetherholt       |  23.8 | 2B     |  586 |   564 |                  73 |               131 |        580 |          49 |            2.6 |              3.1 | unlucky (xwOBA >> wOBA); young                                                         |
| Heriberto Hernández |  26.5 | OF     |  446 |   465 |                 130 |               nan |        418 |         160 |            3.6 |              3   | unlucky (xwOBA >> wOBA); elite bat speed; elite contact quality; needs playing time    |
| Riley Greene        |  25.8 | OF     |  536 |   568 |                  68 |                44 |        567 |          56 |            2.7 |              2.5 | elite contact quality                                                                  |
| Mike Trout          |  34.9 | OF     |  575 |   588 |                  55 |                98 |        565 |          59 |            2.6 |              2.5 | unlucky (xwOBA >> wOBA); elite contact quality; plus speed                             |
| A.J. Ewing          |  21.9 | OF     |  438 |   451 |                 138 |               nan |        474 |         120 |            2   |              2.3 | unlucky (xwOBA >> wOBA); plus speed; needs playing time; young                         |
| Munetaka Murakami   |  26.4 | 1B     |  492 |   501 |                 111 |               111 |        494 |         102 |            2.1 |              2.3 | elite contact quality                                                                  |
| Jo Adell            |  27.2 | OF     |  603 |   560 |                  76 |                79 |        610 |          34 |            1.3 |              2.1 | unlucky (xwOBA >> wOBA); elite bat speed                                               |
| Royce Lewis         |  27.1 | 1B/3B  |  460 |   432 |                 155 |               132 |        453 |         132 |            2.1 |              2.1 | elite bat speed                                                                        |
| Colt Keith          |  24.9 | 3B     |  410 |   379 |                 186 |               189 |        399 |         170 |            1.6 |              2   | unlucky (xwOBA >> wOBA); needs playing time; young                                     |
| JJ Bleday           |  28.6 | OF     |  498 |   479 |                 123 |               nan |        456 |         130 |            2   |              1.9 | unlucky (xwOBA >> wOBA); skills trending up                                            |
| Trent Grisham       |  29.7 | OF     |  484 |   472 |                 126 |               142 |        502 |          96 |            1.1 |              1.7 | unlucky (xwOBA >> wOBA); plus approach                                                 |
| Brandon Nimmo       |  33.3 | OF     |  606 |   563 |                  74 |                83 |        591 |          42 |            1.1 |              1.7 | unlucky (xwOBA >> wOBA); elite contact quality                                         |
| Casey Schmitt       |  27.3 | 3B/OF  |  409 |   434 |                 153 |               321 |        382 |         183 |            2.2 |              1.6 | unlucky (xwOBA >> wOBA); elite contact quality; skills trending up; needs playing time |
| Drake Baldwin       |  25.3 | C      |  559 |   595 |                  51 |                69 |        532 |          77 |            2   |              1.5 | elite contact quality                                                                  |
| Luis García Jr.     |  26.1 | 1B     |  503 |   593 |                  53 |               157 |        496 |         100 |            2.3 |              1.4 | elite bat speed; elite contact quality; skills trending up                             |

### 2027 breakout candidates — playing-time dependent, top 12
| name              |   age | elig   |   PA |   pts |   final_hitter_rank |   proj_pts_600pa |   proj_rank_600pa |   skills_score |   breakout_score | tag                                                                                                                 |
|:------------------|------:|:-------|-----:|------:|--------------------:|-----------------:|------------------:|---------------:|-----------------:|:--------------------------------------------------------------------------------------------------------------------|
| Brady House       |  23.1 | 3B     |  322 |   292 |                 245 |              598 |                95 |            3.2 |              3.5 | unlucky (xwOBA >> wOBA); elite contact quality; skills trending up; needs playing time; young                       |
| Spencer Jones     |  25.1 | OF     |  243 |   277 |                 262 |              700 |                10 |            2.9 |              3.2 | elite bat speed; elite contact quality; plus speed; needs playing time                                              |
| Lars Nootbaar     |  28.8 | OF     |  282 |   254 |                 278 |              575 |               176 |            2.3 |              2.7 | unlucky (xwOBA >> wOBA); plus approach; needs playing time                                                          |
| Henry Davis       |  26.8 | C      |  279 |   250 |                 280 |              621 |                51 |            2   |              2.6 | unlucky (xwOBA >> wOBA); elite bat speed; needs playing time                                                        |
| Luke Raley        |  31.8 | OF     |  287 |   267 |                 270 |              591 |               121 |            2.4 |              2.4 | unlucky (xwOBA >> wOBA); elite bat speed; elite contact quality; skills trending up; plus speed; needs playing time |
| Jose Fernandez    |  22.8 | 3B     |  207 |   194 |                 332 |              615 |                60 |            1.3 |              2   | plus speed; needs playing time; young                                                                               |
| Spencer Steer     |  28.6 | 1B/OF  |  384 |   368 |                 195 |              575 |               176 |            1.8 |              1.8 | unlucky (xwOBA >> wOBA); skills trending up; plus speed; needs playing time                                         |
| Edouard Julien    |  27.2 | 2B     |  276 |   225 |                 302 |              569 |               198 |            1.9 |              1.8 | unlucky (xwOBA >> wOBA); needs playing time                                                                         |
| Braden Montgomery |  23.2 | OF     |  325 |   289 |                 249 |              573 |               187 |            1.3 |              1.7 | unlucky (xwOBA >> wOBA); needs playing time; young                                                                  |
| Jose Siri         |  30.9 | OF     |  210 |   214 |                 309 |              643 |                33 |            1   |              1.6 | elite contact quality; plus speed; needs playing time                                                               |
| Ke'Bryan Hayes    |  29.4 | 3B     |  224 |   122 |                 402 |              550 |               270 |            1.4 |              1.6 | unlucky (xwOBA >> wOBA); needs playing time                                                                         |
| Oneil Cruz        |  27.7 | OF     |  375 |   501 |                 111 |              717 |                 4 |            2.6 |              1.6 | elite bat speed; elite contact quality; needs playing time                                                          |

### Bort's roster (Basketball) — keeper view
| player              | pos      | mlb   |   fpts |   age |   PA |   final_hitter_rank |   proj_pts |   proj_rank |   proj_pts_600pa |   proj_rank_600pa |   breakout_score | tag                                                                                             |
|:--------------------|:---------|:------|-------:|------:|-----:|--------------------:|-----------:|------------:|-----------------:|------------------:|-----------------:|:------------------------------------------------------------------------------------------------|
| Bobby Witt Jr.      | SS       | KC    |    684 |  26   |  576 |                  19 |        771 |           1 |              774 |                 1 |              4.2 | unlucky (xwOBA >> wOBA); elite contact quality; plus speed                                      |
| Fernando Tatis Jr.  | 2B,OF    | SD    |    750 |  27.5 |  645 |                   3 |        760 |           2 |              713 |                 6 |              5   | unlucky (xwOBA >> wOBA); elite bat speed; elite contact quality; skills trending up; plus speed |
| James Wood          | OF       | WSH   |    726 |  23.8 |  575 |                   6 |        746 |           4 |              748 |                 2 |              6.4 | unlucky (xwOBA >> wOBA); elite bat speed; elite contact quality; skills trending up; young      |
| Shohei Ohtani       | UT,SP    | LAD   |    689 |  32   |  596 |                  16 |        726 |           6 |              700 |                10 |              1.3 | unlucky (xwOBA >> wOBA); elite contact quality                                                  |
| Julio Rodriguez     | OF       | SEA   |    598 |  25.5 |  601 |                  47 |        622 |          29 |              599 |                89 |              2.4 | unlucky (xwOBA >> wOBA); elite bat speed; plus speed                                            |
| Sal Stewart         | 1B,3B    | CIN   |    715 |  22.6 |  641 |                   6 |        566 |          58 |              648 |                29 |              0.5 | young                                                                                           |
| Brandon Lowe        | 2B       | PIT   |    624 |  32   |  606 |                  41 |        548 |          66 |              553 |               262 |             -1.6 | nan                                                                                             |
| Nick Kurtz          | 1B       | ATH   |    496 |  23.3 |  434 |                 117 |        532 |          77 |              717 |                 4 |              3.5 | elite bat speed; elite contact quality; needs playing time; young                               |
| Carter Jensen       | C        | KC    |    579 |  23   |  556 |                  59 |        448 |         138 |              586 |               137 |              0.1 | young                                                                                           |
| Heriberto Hernandez | OF       | MIA   |    470 |  26.5 |  446 |                 130 |        418 |         160 |              603 |                75 |              3   | unlucky (xwOBA >> wOBA); elite bat speed; elite contact quality; needs playing time             |
| Jeff McNeil         | 1B,2B,OF | ATH   |    431 |  34.2 |  472 |                 157 |        408 |         164 |              521 |               358 |             -4.2 | nan                                                                                             |
| Ty France           | 1B       | SD    |    483 |  32   |  442 |                 125 |        386 |         182 |              513 |               373 |             -3   | needs playing time                                                                              |
| Corey Seager        | SS       | TEX   |    351 |  32.2 |  379 |                 208 |        349 |         207 |              534 |               328 |             -3.7 | unlucky (xwOBA >> wOBA); elite contact quality; needs playing time                              |
| Cole Carrigg        | SS,OF    | COL   |    337 |  24.1 |  313 |                 213 |        298 |         241 |              571 |               194 |             -0.2 | plus speed; needs playing time; young                                                           |
| Spencer Jones       | OF       | NYY   |    277 |  25.1 |  243 |                 262 |        292 |         248 |              700 |                10 |              3.2 | elite bat speed; elite contact quality; plus speed; needs playing time                          |

## Open questions for Bort
- Keep the beat thresholds (+75 pts / +25 ranks / top-150)? They're in `config.py`.
- Want a pitcher (SP-only, max 10 GS/week, QS=4, K=1, IP=1, ER=-1) version next?
- Want the in-season weekly refresh (savant + MLB API update daily) as a scheduled task?

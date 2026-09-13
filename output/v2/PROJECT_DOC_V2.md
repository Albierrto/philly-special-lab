# Hitter Lab v2 — formulas, keeper economics and 2027 findings

Session 2026-09-13 (second pass). Deliverables: `breakout_hitters_v2.zip` (code + data + output/v2 CSVs), the interactive artifact **Philly Special Hitter Lab**, and this doc. Supersedes the v1 breakout list where they disagree.

## What changed from v1 and why
- v1 ranked "skills above market", so an aging star with great Statcast (Trout) could top the list. v2 defines a breakout **relative to the player's own career**: 2027 points/PA at least 0.08 above his best 200-PA season, 350+ PA, top-90 finish. The Breakout Index (BI) is a gradient-boosted classifier trained on 2021→22 … 2025→26 pairs. Trout's BI for 2027: 1.9. Judge: 3.9.
- Empirical aging curve (delta method on 2021-2026, 300+ PA pairs) is now in the projection. In this league's scoring the curve peaks at 24-25 and falls ~0.036 pts/PA per year at 31-32 and ~0.05-0.06 per year from 33 on (singles and steals age badly).
- Injury history from the MLB transactions feed (IL stints, days, 60-day flags, injury text) feeds a durability multiplier on projected PA.
- Every skill is weighted by what the league pays for through xLP (expected league points): x1B/xXBH/xHR from xBA/xISO, walks/HBP/steals actual, R/RBI from a ridge model on the expected profile. xLP/PA predicts next year better than actual points/PA (r 0.52 vs 0.47).
- Keeper economics: 5 hitter keepers/team, keepers cost only the roster spot (assumption). KSV = projected points minus the (kept+12)th projected hitter. Past keepers inferred (top-ADP hitters missing from the draft) because Fantrax doesn't publish them.
- Prospect status: current MLB Pipeline top-100 (73 hitters) + AAA/AA/A+/A stats 2024-2026 + rookie eligibility → Prospect Arrival Score (PAS).
- League fit vs standard points (ESPN-style, K = -1) and vs 5x5 roto; the explorer can re-score everything under any weights or roto.

## Validation
Breakout Index, leave-one-season-out:
| held_out   |   n |   breakouts |   base_rate |   auc |   top40_hit_rate |   top40_avg_next_pts |
|:-----------|----:|------------:|------------:|------:|-----------------:|---------------------:|
| 2021->2022 | 379 |          19 |       0.05  | 0.64  |            0.1   |                  465 |
| 2022->2023 | 378 |          26 |       0.069 | 0.675 |            0.075 |                  459 |
| 2023->2024 | 370 |          12 |       0.032 | 0.802 |            0.125 |                  443 |
| 2024->2025 | 379 |          16 |       0.042 | 0.772 |            0.125 |                  516 |
| 2025->2026 | 362 |          13 |       0.036 | 0.83  |            0.15  |                  404 |

Points projection (rate), leave-one-season-out:
| train_on            |   n |   r_model |   r_naive_last_year |   r_xwoba_only |   mae_model |   mae_naive |
|:--------------------|----:|----------:|--------------------:|---------------:|------------:|------------:|
| 2021->2022 held out | 267 |     0.545 |               0.426 |          0.382 |       0.099 |       0.12  |
| 2022->2023 held out | 261 |     0.552 |               0.493 |          0.391 |       0.1   |       0.11  |
| 2023->2024 held out | 278 |     0.593 |               0.444 |          0.428 |       0.086 |       0.118 |
| 2024->2025 held out | 265 |     0.553 |               0.515 |          0.387 |       0.085 |       0.1   |
| 2025->2026 held out | 258 |     0.392 |               0.342 |          0.326 |       0.093 |       0.113 |

As of 2025 data only, the 25 highest-BI hitters produced a new career level in 2026 at 16% vs a 3.3% base rate (Caglianone, Otto Lopez, Crow-Armstrong, Meidroth inside the top 25).

## 2027 Breakout Index leaders (100+ PA in 2026)
| name              |   age | owner   |   PA |   pts |   final_hitter_rank |   career_best_pa |   luck_pa |   BI |   proj_pts |   proj_rank |   proj_pts_600 |   KSV | keep_tier   |
|:------------------|------:|:--------|-----:|------:|--------------------:|-----------------:|----------:|-----:|-----------:|------------:|---------------:|------:|:------------|
| Cam Smith         |  23.4 | FA      |  521 |   493 |                 115 |            0.946 |      -0.1 | 61.8 |        532 |          59 |            619 |    21 | marginal    |
| Henry Bolte       |  22.9 | CS      |  415 |   436 |                 151 |            1.051 |       0   | 46.6 |        448 |         116 |            648 |   -63 | let go      |
| JJ Wetherholt     |  23.8 | THALER  |  586 |   564 |                  73 |            0.962 |      -0.1 | 41.6 |        563 |          39 |            590 |    52 | clear keep  |
| Miguel Vargas     |  26.6 | GLIZZY  |  644 |   701 |                  12 |            1.089 |      -0.1 | 33.5 |        626 |          19 |            652 |   115 | clear keep  |
| Konnor Griffin    |  20.2 | THALER  |  280 |   312 |                 226 |            1.114 |       0   | 31.5 |        296 |         228 |            681 |  -215 | let go      |
| Kevin McGonigle   |  21.9 | GLIZZY  |  645 |   646 |                  32 |            1.002 |      -0   | 27.8 |        644 |          11 |            599 |   133 | lock        |
| Jackson Merrill   |  23.2 | CH      |  610 |   678 |                  21 |            1.175 |      -0.1 | 21.4 |        640 |          12 |            670 |   129 | lock        |
| Cole Young        |  22.9 | FA      |  593 |   535 |                  97 |            0.902 |      -0   | 20.9 |        511 |          72 |            583 |     0 | let go      |
| A.J. Ewing        |  21.9 | CH      |  438 |   451 |                 138 |            1.03  |      -0.1 | 20.4 |        476 |          91 |            652 |   -35 | let go      |
| José Fermín       |  27.3 | FA      |  344 |   300 |                 236 |            0.872 |       0   | 14.6 |        254 |         268 |            561 |  -257 | let go      |
| Casey Schmitt     |  27.3 | FA      |  409 |   434 |                 153 |            1.061 |      -0.1 | 14.6 |        318 |         206 |            572 |  -193 | let go      |
| Luisangel Acuña   |  24.3 | FA      |  209 |   192 |                 334 |            0.919 |      -0.1 | 14   |        220 |         311 |            659 |  -291 | let go      |
| Liam Hicks        |  27.1 | MMM     |  518 |   559 |                  77 |            1.079 |       0.2 | 14   |        484 |          84 |            593 |   -27 | let go      |
| Angel Genao       |  22.1 | FA      |  115 |    92 |                 438 |          nan     |      -0   | 13.7 |        185 |         378 |            556 |  -326 | let go      |
| Braden Montgomery |  23.2 | CJF     |  325 |   289 |                 249 |            0.889 |      -0.1 | 12.4 |        310 |         213 |            573 |  -201 | let go      |
| Jose Fernandez    |  22.8 | FA      |  207 |   194 |                 332 |            0.937 |       0   | 11.8 |        212 |         317 |            615 |  -299 | let go      |
| Colt Emerson      |  20.9 | FA      |  241 |   180 |                 344 |            0.747 |       0   | 11.7 |        208 |         323 |            535 |  -303 | let go      |
| Dylan Crews       |  24.3 | FA      |  396 |   368 |                 195 |            0.978 |      -0.1 | 10.5 |        331 |         194 |            607 |  -180 | let go      |
| Coby Mayo         |  24.6 | FA      |  402 |   408 |                 172 |            1.015 |       0   |  9.8 |        344 |         187 |            607 |  -167 | let go      |
| Roman Anthony     |  22.1 | CJF     |  192 |   168 |                 352 |          nan     |      -0.2 |  9.5 |        194 |         350 |            583 |  -317 | let go      |
| James Wood        |  23.8 | BB      |  575 |   716 |                   6 |            1.245 |      -0   |  9.1 |        676 |           6 |            744 |   165 | lock        |
| Samuel Basallo    |  21.9 | GK      |  376 |   365 |                 198 |            0.971 |      -0   |  9   |        310 |         213 |            586 |  -201 | let go      |
| Luke Keaschall    |  23.9 | Dpf     |  561 |   547 |                  87 |            1.203 |       0   |  8.7 |        423 |         132 |            566 |   -88 | let go      |
| Jac Caglianone    |  23.4 | CH      |  511 |   566 |                  71 |            1.108 |      -0   |  8.5 |        479 |          89 |            651 |   -32 | let go      |
| Tommy White       |  23.3 | FA      |  166 |   117 |                 413 |          nan     |      -0.2 |  8.2 |        193 |         354 |            580 |  -318 | let go      |

### Free agents in Philly Special with the highest BI
| name            |   age |   PA |   pts |   final_hitter_rank |   BI |   proj_pts |   proj_rank |   proj_pts_600 |
|:----------------|------:|-----:|------:|--------------------:|-----:|-----------:|------------:|---------------:|
| Cam Smith       |  23.4 |  521 |   493 |                 115 | 61.8 |        532 |          59 |            619 |
| Cole Young      |  22.9 |  593 |   535 |                  97 | 20.9 |        511 |          72 |            583 |
| José Fermín     |  27.3 |  344 |   300 |                 236 | 14.6 |        254 |         268 |            561 |
| Casey Schmitt   |  27.3 |  409 |   434 |                 153 | 14.6 |        318 |         206 |            572 |
| Luisangel Acuña |  24.3 |  209 |   192 |                 334 | 14   |        220 |         311 |            659 |
| Angel Genao     |  22.1 |  115 |    92 |                 438 | 13.7 |        185 |         378 |            556 |
| Jose Fernandez  |  22.8 |  207 |   194 |                 332 | 11.8 |        212 |         317 |            615 |
| Colt Emerson    |  20.9 |  241 |   180 |                 344 | 11.7 |        208 |         323 |            535 |
| Dylan Crews     |  24.3 |  396 |   368 |                 195 | 10.5 |        331 |         194 |            607 |
| Coby Mayo       |  24.6 |  402 |   408 |                 172 |  9.8 |        344 |         187 |            607 |
| Tommy White     |  23.3 |  166 |   117 |                 413 |  8.2 |        193 |         354 |            580 |
| Nolan Schanuel  |  24.4 |  446 |   394 |                 178 |  7.6 |        414 |         138 |            545 |
| Ryan Kreidler   |  28.6 |  238 |   213 |                 310 |  7.4 |        198 |         338 |            594 |
| Brice Matthews  |  24.3 |  226 |   185 |                 338 |  7.3 |        198 |         338 |            593 |
| Brady House     |  23.1 |  322 |   292 |                 245 |  7   |        311 |         211 |            598 |

## Bort's keeper board (Basketball)
| name               |   age |   pts |   proj_pts |   proj_rank |   KSV |   KSV_2nd | keep_tier   |   BI |
|:-------------------|------:|------:|-----------:|------------:|------:|----------:|:------------|-----:|
| Bobby Witt Jr.     |  26   |   684 |        770 |           1 |   259 |       286 | lock        |  2.2 |
| Shohei Ohtani      |  32   |   689 |        711 |           4 |   200 |       227 | lock        |  4.6 |
| Fernando Tatis Jr. |  27.5 |   750 |        695 |           5 |   184 |       211 | lock        |  3   |
| James Wood         |  23.8 |   716 |        676 |           6 |   165 |       192 | lock        |  9.1 |
| Julio Rodríguez    |  25.5 |   598 |        595 |          29 |    84 |       111 | clear keep  |  0.4 |
| Sal Stewart        |  22.6 |   716 |        566 |          36 |    55 |        82 | clear keep  |  7.5 |
| Nick Kurtz         |  23.3 |   491 |        497 |          78 |   -14 |        13 | let go      |  3.6 |

Decision point: five hitter slots, six clear keeps. The model has Julio (KSV 84) over Sal Stewart (55) on projection; Stewart is younger with the higher BI. Kurtz projects as a let-go by KSV but has the best per-600 rate on the roster (717).

## 2027 draft pool (likely keepers removed), top 20
|   pool_rank | name              |   age | owner   |   pts |   proj_pts |   proj_rank |   BI |
|------------:|:------------------|------:|:--------|------:|-----------:|------------:|-----:|
|           1 | Wilyer Abreu      |  27   | GLIZZY  |   662 |        594 |          30 |  3   |
|           2 | Jo Adell          |  27.2 | MMM     |   560 |        577 |          32 |  1.1 |
|           3 | Sal Stewart       |  22.6 | BB      |   716 |        566 |          36 |  7.5 |
|           4 | Seiya Suzuki      |  31.9 | BAMBI   |   647 |        550 |          49 |  0.1 |
|           5 | Luis Arraez       |  29.2 | GLIZZY  |   597 |        542 |          51 |  0.1 |
|           6 | Trea Turner       |  33   | CS      |   599 |        541 |          53 |  0.1 |
|           7 | Ben Rice          |  27.4 | CS      |   679 |        541 |          53 |  2   |
|           8 | Bryson Stott      |  28.7 | CS      |   562 |        540 |          56 |  0.3 |
|           9 | Freddie Freeman   |  36.8 | MMM     |   621 |        538 |          57 |  0.2 |
|          10 | Ketel Marte       |  32.7 | GK      |   557 |        536 |          58 |  0.4 |
|          11 | Cam Smith         |  23.4 | FA      |   493 |        532 |          59 | 61.8 |
|          12 | TJ Rumfield       |  26.1 | FA      |   580 |        526 |          62 |  1.7 |
|          13 | Jake McCarthy     |  28.9 | MMM     |   658 |        525 |          63 |  2   |
|          14 | Willy Adames      |  30.8 | FA      |   453 |        523 |          65 |  0.1 |
|          15 | Isaac Paredes     |  27.4 | Dan     |   575 |        523 |          65 |  0.2 |
|          16 | Steven Kwan       |  28.8 | Dan     |   528 |        521 |          67 |  0.1 |
|          17 | Drake Baldwin     |  25.3 | BAMBI   |   595 |        518 |          68 |  1.2 |
|          18 | Spencer Torkelson |  26.8 | Dan     |   495 |        517 |          70 |  0.8 |
|          19 | Taylor Ward       |  32.5 | FA      |   488 |        512 |          71 |  0.1 |
|          20 | José Ramírez      |  33.8 | MMM     |   508 |        511 |          72 |  0.3 |

## Top prospects by PAS
| name               |   age |   pipeline_rank | milb_level   |   milb_PA |   milb_OPS |   milb_ISO |   milb_K |   milb_BB |   milb_SB |   PAS | prospect_status   | owner   |
|:-------------------|------:|----------------:|:-------------|----------:|-----------:|-----------:|---------:|----------:|----------:|------:|:------------------|:--------|
| Franklin Arias     |  20   |               3 | AAA          |       161 |          1 |          0 |        0 |         0 |         3 |  81.4 | top-100 prospect  | nan     |
| Josue De Paula     |  21.1 |               6 | AA           |       588 |          1 |          0 |        0 |         0 |        31 |  80.9 | top-100 prospect  | nan     |
| Max Clark          |  21.5 |              13 | AAA          |       418 |          1 |          0 |        0 |         0 |        21 |  79.7 | top-100 prospect  | CS      |
| Leo De Vries       |  19   |               2 | AA           |       428 |          1 |          0 |        0 |         0 |        34 |  78.8 | top-100 prospect  | nan     |
| Jesús Made         |  19   |               1 | AA           |       554 |          1 |          0 |        0 |         0 |        36 |  78.8 | top-100 prospect  | nan     |
| Ethan Salas        |  20.1 |              23 | AAA          |       105 |          1 |          0 |        0 |         0 |         3 |  78.6 | top-100 prospect  | nan     |
| Sebastian Walcott  |  20   |               9 | AA           |       168 |          1 |          0 |        0 |         0 |        11 |  78.3 | top-100 prospect  | nan     |
| Walker Jenkins     |  21.4 |              15 | AAA          |       307 |          1 |          0 |        0 |         0 |        17 |  77.9 | top-100 prospect  | nan     |
| Theo Gillen        |  21   |              12 | AA           |       264 |          1 |          0 |        0 |         0 |        16 |  77.1 | top-100 prospect  | nan     |
| George Lombard Jr. |  21.1 |              14 | AAA          |       262 |          1 |          0 |        0 |         0 |        10 |  76.5 | top-100 prospect  | THALER  |
| Rainiel Rodriguez  |  19   |              18 | AA           |       402 |          1 |          0 |        0 |         0 |         7 |  75.1 | top-100 prospect  | nan     |
| Angel Genao        |  22.1 |              29 | AAA          |       309 |          1 |          0 |        0 |         0 |         9 |  74.7 | top-100 prospect  | FA      |
| Eli Willits        |  18   |               4 | A+           |       289 |          1 |          0 |        0 |         0 |        31 |  74.2 | top-100 prospect  | nan     |
| Alfredo Duno       |  20   |              22 | AA           |       138 |          1 |          0 |        0 |         0 |         3 |  72.4 | top-100 prospect  | nan     |
| Ralphy Velazquez   |  21   |              32 | AAA          |       352 |          1 |          0 |        0 |         0 |         1 |  71.5 | top-100 prospect  | nan     |
| Mike Sirota        |  23   |              16 | AA           |       333 |          1 |          0 |        0 |         0 |         5 |  70.7 | top-100 prospect  | nan     |
| Kaelen Culpepper   |  23.5 |              34 | AAA          |       343 |          1 |          0 |        0 |         0 |        17 |  70.6 | top-100 prospect  | FA      |
| Caleb Bonemer      |  20   |              30 | AA           |       313 |          1 |          0 |        0 |         0 |         7 |  70   | top-100 prospect  | nan     |
| Lazaro Montes      |  21.7 |              56 | AAA          |       203 |          1 |          0 |        0 |         0 |         3 |  68.9 | top-100 prospect  | nan     |
| Zyhir Hope         |  21   |              28 | AA           |       559 |          1 |          0 |        0 |         0 |        26 |  68.5 | top-100 prospect  | nan     |

## Aging curve (points/PA relative to age 27)
|   age |   rel_to_27 |   yoy_delta |   n_pairs |
|------:|------------:|------------:|----------:|
|    21 |           0 |           0 |         4 |
|    22 |           0 |           0 |        26 |
|    23 |           0 |           0 |        34 |
|    24 |           0 |          -0 |        71 |
|    25 |           0 |          -0 |        77 |
|    26 |           0 |          -0 |       106 |
|    27 |           0 |          -0 |       103 |
|    28 |          -0 |          -0 |       108 |
|    29 |          -0 |          -0 |        91 |
|    30 |          -0 |          -0 |        97 |
|    31 |          -0 |          -0 |        79 |
|    32 |          -0 |          -0 |        63 |
|    33 |          -0 |          -0 |        41 |
|    34 |          -0 |          -0 |        36 |
|    35 |          -0 |          -0 |        21 |
|    36 |          -0 |          -0 |        12 |
|    37 |          -0 |          -0 |         6 |
|    38 |          -0 |          -0 |         7 |
|    39 |          -1 |         nan |         0 |

## Formulas (short form)
- Points: 2×1B + 3×2B + 4×3B + 5×HR + R + RBI + BB + HBP + 3×SB − E + 2×CSA + 2×AOF (+5 GS, +10 cycle)
- xLP: xH = xBA×AB; xXBH = xISO×AB/(r2B + 2r3B + 3rHR); x1B = xH − xXBH; xR, xRBI = ridge(xOBP, xISO, xBA, SB/PA, xHR/PA, sprint); luck = LP − xLP
- LPAR: points − replacement (fill C/SS/2B/3B/1B/OF for every team, then UT; replacement = best unrostered +2 bench)
- Value over ADP: isotonic rolling-median curve of points by ADP hitter rank; beat = +75 pts, +25 ranks, top-150
- BI: P(next pts/PA ≥ career best + 0.08 ∧ PA ≥ 350 ∧ rank ≤ 90 | Statcast, xLP gap, career best, gap-to-best, seasons, age, PA, IL, yoy K%/barrel/bat-speed)
- Projection: model rate + ½ aging step; PA = (0.6×PA + 0.4×3-yr mean) × (1 − 0.0008×IL days 3yr) × (0.95 if ≥ 33), clipped 200-650
- KSV = proj − proj[rank = kept + 12]; KSV_2nd uses kept + 24
- PAS = 35 pedigree + 25 age-vs-level + 25 MiLB bat + 15 MLB sample
- League fit = percentile(this scoring) − percentile(ESPN points) / − percentile(5x5 roto z)

## Open items
- Real keeper lists for 2025/2026 would replace the inferred ones (drop `data/fantrax/keepers_<year>.csv`, columns player,team).
- Historical prospect ranks (post-hype detection) are not public in a scrapeable form; pedigree only covers the current top 100.
- Pitcher model (SP-only, 10 GS/week cap) not built yet.

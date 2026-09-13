"""
breakout — find breakout hitters and hitters who beat their ADP, tuned to a specific fantasy league.

Data sources (all free, no login):
  * Baseball Savant custom leaderboard (Statcast: xwOBA, barrels, EV, bat speed, sprint speed, discipline)
  * MLB Stats API (season counting stats for every hitter + fielding for E / catcher CS / OF assists)
  * FantasyPros historical ADP archive (composite of NFBC, Fantrax, CBS, Yahoo, RTS, ESPN) by year
  * NFBC live ADP (current draft season)
  * Your Fantrax league export (rules, draft results) — read-only files you drop in data/fantrax/

Pipeline:  sources -> build.player_seasons -> scoring -> adp_value -> breakout -> report
"""
__version__ = "0.1.0"

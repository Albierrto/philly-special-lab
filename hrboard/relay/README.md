# Live Kalshi prices (optional, free)

Kalshi's public price API refuses requests that come from a web page, so the Home Run Board cannot read Kalshi by itself.
Without this, Kalshi prices refresh about every 20 minutes (the `hrlive` workflow). With it, the page reads them live.

The relay is `worker.js` in this folder: a Cloudflare Worker that forwards one read-only request (market prices) and
nothing else. It holds no keys, cannot place orders, only answers the board's own site, and caches for 30 seconds.

1. Create a free account at dash.cloudflare.com.
2. **Workers & Pages → Create → Create Worker** (the "Hello World" starter). Name it `kalshi-relay`, then **Deploy**.
3. **Edit code**, replace everything with the contents of `worker.js`, **Deploy** again.
4. Copy the worker's address (looks like `https://kalshi-relay.<your-name>.workers.dev`).
   Check it: `https://kalshi-relay.<your-name>.workers.dev/markets?series_ticker=KXMLBGAME&status=open&limit=2` should show JSON.
5. In this GitHub repo: **Settings → Secrets and variables → Actions → Variables → New repository variable**,
   name `KALSHI_RELAY`, value the address from step 4 (no trailing slash). It is not a secret; it is just an address.

The next board build picks it up; the page then says "Kalshi prices live".

# More sportsbooks (optional, free tier)

1. Get a free key at the-odds-api.com (Starter plan, 500 credits a month).
2. **Settings → Secrets and variables → Actions → Secrets → New repository secret**, name `ODDS_API_KEY`.

The board then adds FanDuel, BetMGM, Caesars, Fanatics, BetRivers, theScore Bet, Hard Rock, Novig and ProphetX to the
game lines (moneyline, run line, total). Each call costs 3 credits; `hrboard/books.py` calls about four times a day and
spreads what is left over the rest of the month, so the free plan never runs out. Player bets from these books need a
paid plan and are not requested.

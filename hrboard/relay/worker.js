// Kalshi price relay for the Home Run Board (Cloudflare Worker, free plan).
//
// Kalshi's public market-data API refuses requests that come from a web page, so the board cannot read live prices
// by itself. This worker forwards ONE kind of request, a read of market prices, and adds the header that lets the
// board's page read the answer. It holds no keys and can place no orders: it only ever calls
// GET https://api.elections.kalshi.com/trade-api/v2/markets with a handful of whitelisted query parameters.
// Answers are cached for 30 seconds, so however many people have the page open, Kalshi sees at most a few calls a minute.

const ALLOWED_ORIGINS = ["https://albierrto.github.io"];
const PARAMS = ["tickers", "event_ticker", "series_ticker", "status", "limit", "cursor"];
const UPSTREAM = "https://api.elections.kalshi.com/trade-api/v2/markets";

function corsHeaders(origin) {
  return {
    "Access-Control-Allow-Origin": ALLOWED_ORIGINS.includes(origin) ? origin : ALLOWED_ORIGINS[0],
    "Access-Control-Allow-Methods": "GET, OPTIONS",
    "Access-Control-Max-Age": "86400",
    "Vary": "Origin",
  };
}

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const cors = corsHeaders(request.headers.get("Origin") || "");
    if (request.method === "OPTIONS") return new Response(null, { status: 204, headers: cors });
    if (request.method !== "GET" || url.pathname.replace(/\/+$/, "") !== "/markets") {
      return new Response("Not found", { status: 404, headers: cors });
    }
    const q = new URLSearchParams();
    for (const k of PARAMS) {
      const v = url.searchParams.get(k);
      if (v !== null && v.length <= 4000) q.set(k, v);
    }
    const target = `${UPSTREAM}?${q.toString()}`;
    const cache = caches.default;
    const cacheKey = new Request(target, { method: "GET" });
    let res = await cache.match(cacheKey);
    if (!res) {
      const upstream = await fetch(target, { headers: { Accept: "application/json" } });
      res = new Response(upstream.body, { status: upstream.status, headers: { "Content-Type": "application/json", "Cache-Control": "public, max-age=30" } });
      if (upstream.ok) ctx.waitUntil(cache.put(cacheKey, res.clone()));
    }
    const out = new Response(res.body, res);
    for (const [k, v] of Object.entries(cors)) out.headers.set(k, v);
    return out;
  },
};

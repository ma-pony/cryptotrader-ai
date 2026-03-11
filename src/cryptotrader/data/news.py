"""News sentiment collector — RSS feeds + keyword sentiment + social buzz.

Uses unified SQLite store for caching: check cache first, only call API if stale.
Sentiment scoring uses keyword matching — the LLM (NewsAgent) handles deeper analysis.
"""

from __future__ import annotations

import asyncio
import logging

import feedparser
import httpx

from cryptotrader.data.store import _record_fetch, _should_fetch, cache_result, get_cached_or_none
from cryptotrader.models import NewsSentiment

logger = logging.getLogger(__name__)

RSS_FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed",
]

POSITIVE = {
    "bullish",
    "surge",
    "rally",
    "soar",
    "gain",
    "rise",
    "jump",
    "breakout",
    "adoption",
    "approval",
    "partnership",
    "upgrade",
    "record",
    "high",
    "buy",
}
NEGATIVE = {
    "bearish",
    "crash",
    "plunge",
    "drop",
    "fall",
    "dump",
    "hack",
    "ban",
    "fraud",
    "lawsuit",
    "sell",
    "fear",
    "risk",
    "loss",
    "decline",
    "sec",
}

# CoinGecko coin ID mapping for social buzz lookup
_COINGECKO_IDS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "BNB": "binancecoin",
    "XRP": "ripple",
    "ADA": "cardano",
    "DOGE": "dogecoin",
    "AVAX": "avalanche-2",
    "DOT": "polkadot",
    "MATIC": "matic-network",
    "LINK": "chainlink",
    "UNI": "uniswap",
    "ATOM": "cosmos",
    "LTC": "litecoin",
}


def _score_text(text: str) -> float:
    """Keyword-based sentiment score (-1 to +1)."""
    words = set(text.lower().split())
    pos = len(words & POSITIVE)
    neg = len(words & NEGATIVE)
    total = pos + neg
    return (pos - neg) / total if total else 0.0


def _score_headlines(headlines: list[str]) -> float:
    """Score headlines using keyword sentiment."""
    if not headlines:
        return 0.0
    return _score_text(" ".join(headlines))


async def _fetch_social_buzz(symbol: str, date: str | None = None) -> float:
    """Fetch social buzz score from CoinGecko community data (free, no key).

    Returns a normalized 0-1 score based on community engagement metrics.
    """
    cache_key = f"live_social_buzz_{symbol.upper()}"
    cached = get_cached_or_none(cache_key, date)
    if cached is not None:
        return float(cached) if isinstance(cached, int | float) else 0.0

    # Backtest mode: no live API call
    if date is not None:
        return 0.0

    coin_id = _COINGECKO_IDS.get(symbol.upper(), symbol.lower())
    if not coin_id:
        return 0.0
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"https://api.coingecko.com/api/v3/coins/{coin_id}",
                params={"localization": "false", "tickers": "false", "market_data": "false", "sparkline": "false"},
            )
            resp.raise_for_status()
            data = resp.json()

        community = data.get("community_data", {})
        # Twitter followers + Reddit subscribers as activity proxy
        twitter = community.get("twitter_followers", 0) or 0
        reddit = community.get("reddit_subscribers", 0) or 0
        # Sentiment votes as real-time buzz indicator
        sentiment_up = data.get("sentiment_votes_up_percentage", 50) or 50
        # Normalize: high follower coins (BTC) get ~0.5 base, sentiment shifts it
        # Score range: 0.0 (dead) to 1.0 (viral)
        reach = min(1.0, (twitter + reddit) / 10_000_000)  # cap at 10M
        sentiment_shift = (sentiment_up - 50) / 50  # -1 to +1
        result = round(reach * 0.5 + (0.5 * abs(sentiment_shift)), 3)
        cache_result(cache_key, result)
        return result
    except Exception:
        logger.debug("Social buzz fetch failed for %s", symbol, exc_info=True)
        return 0.0


class NewsCollector:
    async def collect(self, pair: str, date: str | None = None) -> NewsSentiment:
        symbol = pair.split("/")[0]

        # Check RSS cache first — use symbol-specific key for stored data,
        # but "live_news_rss" as the rate-limit key (matches _RATE_LIMITS entry).
        rss_store_key = f"live_news_rss_{symbol.lower()}"
        rss_rate_key = "live_news_rss"
        cached_rss = get_cached_or_none(rss_store_key, date)

        if cached_rss is not None and isinstance(cached_rss, dict):
            headlines = cached_rss.get("headlines", [])
            score = cached_rss.get("score", 0.0)
            key_events = cached_rss.get("key_events", [])
        elif date is not None:
            # Backtest mode: no live API call
            headlines, score, key_events = [], 0.0, []
        elif not _should_fetch(rss_rate_key):
            # Rate-limited globally; no new fetch for any symbol right now
            headlines, score, key_events = [], 0.0, []
        else:
            # Fetch from RSS and CryptoCompare in parallel, merge results
            rss_task = self._collect_rss(symbol.lower())
            cc_task = self._collect_cryptocompare(symbol)
            (rss_headlines, rss_score, rss_events), cc_headlines = await asyncio.gather(rss_task, cc_task)

            # Merge: RSS headlines first, then CryptoCompare (deduplicated)
            seen = set(rss_headlines)
            merged = list(rss_headlines)
            for h in cc_headlines:
                if h not in seen:
                    merged.append(h)
                    seen.add(h)

            # Re-score merged headlines
            headlines = merged[:15]
            score = _score_headlines(headlines) if headlines else rss_score
            key_events = [
                h for h in headlines if any(w in h.lower() for w in ("sec", "etf", "hack", "ban", "approval", "record"))
            ][:5]
            if not key_events:
                key_events = rss_events

            if headlines:
                cache_result(rss_store_key, {"headlines": headlines, "score": score, "key_events": key_events})
            _record_fetch(rss_rate_key)

        # Fetch social buzz (with its own caching)
        social_buzz = await _fetch_social_buzz(symbol, date)

        return NewsSentiment(
            headlines=headlines,
            sentiment_score=round(score, 3),
            key_events=key_events,
            social_buzz=social_buzz,
        )

    async def _collect_rss(self, symbol: str) -> tuple[list[str], float, list[str]]:
        all_headlines: list[str] = []
        for url in RSS_FEEDS:
            try:
                feed = await asyncio.to_thread(feedparser.parse, url)
                all_headlines.extend(e.get("title", "") for e in feed.entries[:15])
            except Exception:
                logger.warning("RSS fetch failed: %s", url, exc_info=True)

        if not all_headlines:
            return [], 0.0, []

        relevant = [h for h in all_headlines if symbol in h.lower() or "crypto" in h.lower() or "bitcoin" in h.lower()]
        if not relevant:
            relevant = all_headlines[:10]

        score = _score_headlines(relevant)
        key_events = [
            h for h in relevant if any(w in h.lower() for w in ("sec", "etf", "hack", "ban", "approval", "record"))
        ]
        return relevant[:10], score, key_events[:5]

    async def _collect_cryptocompare(self, symbol: str) -> list[str]:
        """Fetch latest news from CoinDesk/CryptoCompare."""
        try:
            from cryptotrader.config import load_config

            cfg = load_config()
            api_key = cfg.providers.coindesk_api_key

            async with httpx.AsyncClient(timeout=10) as client:
                if api_key:
                    resp = await client.get(
                        "https://data-api.coindesk.com/news/v1/article/list",
                        params={"lang": "EN", "asset_ids": symbol.upper(), "limit": 50},
                        headers={"Authorization": f"Bearer {api_key}"},
                    )
                else:
                    resp = await client.get(
                        "https://min-api.cryptocompare.com/data/v2/news/",
                        params={"lang": "EN", "categories": symbol.upper()},
                    )
                resp.raise_for_status()
                body = resp.json()
                if body.get("Response") == "Error":
                    return []
                articles = body.get("Data", [])
                if not isinstance(articles, list):
                    return []
                title_key = "TITLE" if api_key else "title"
                return [a[title_key] for a in articles[:20] if a.get(title_key)]
        except Exception:
            logger.debug("News fetch failed", exc_info=True)
            return []

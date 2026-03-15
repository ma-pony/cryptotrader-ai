"""News collector — RSS feeds + social buzz.

Uses unified SQLite store for caching: check cache first, only call API if stale.
Raw headlines are passed to the LLM (NewsAgent) for sentiment analysis.
"""

from __future__ import annotations

import asyncio
import logging

import feedparser
import httpx
from pydantic import ValidationError

from cryptotrader.data.store import _record_fetch, _should_fetch, cache_result, get_cached_or_none
from cryptotrader.models import NewsArticle, NewsHeadlineResponse, NewsSentiment

logger = logging.getLogger(__name__)

RSS_FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed",
]

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


_MAX_SUMMARY_LEN = 500  # Truncate article body/summary to avoid prompt bloat


def _strip_html(text: str) -> str:
    """Remove HTML tags from text (simple regex, no dependency)."""
    import re

    return re.sub(r"<[^>]+>", "", text).strip()


def _truncate(text: str, max_len: int = _MAX_SUMMARY_LEN) -> str:
    """Truncate text to max_len, appending '...' if truncated."""
    if len(text) <= max_len:
        return text
    return text[:max_len].rsplit(" ", 1)[0] + "..."


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
            key_events = cached_rss.get("key_events", [])
            articles = [NewsArticle(**a) for a in cached_rss.get("articles", [])]
        elif date is not None:
            # Backtest mode: no live API call
            headlines, key_events, articles = [], [], []
        elif not _should_fetch(rss_rate_key):
            # Rate-limited globally; no new fetch for any symbol right now
            headlines, key_events, articles = [], [], []
        else:
            # Fetch from RSS and CryptoCompare in parallel, merge results
            rss_task = self._collect_rss(symbol.lower())
            cc_task = self._collect_cryptocompare(symbol)
            rss_result, cc_articles = await asyncio.gather(rss_task, cc_task)

            # Merge: RSS articles first, then CryptoCompare (deduplicated by title)
            seen: set[str] = set()
            articles: list[NewsArticle] = []
            for a in [*rss_result, *cc_articles]:
                if a.title not in seen:
                    articles.append(a)
                    seen.add(a.title)

            articles = articles[:15]
            headlines = [a.title for a in articles]
            key_events = [
                a.title
                for a in articles
                if any(w in a.title.lower() for w in ("sec", "etf", "hack", "ban", "approval", "record"))
            ][:5]

            if articles:
                cache_result(
                    rss_store_key,
                    {
                        "headlines": headlines,
                        "key_events": key_events,
                        "articles": [
                            {"title": a.title, "summary": a.summary, "source": a.source, "published": a.published}
                            for a in articles
                        ],
                    },
                )
            _record_fetch(rss_rate_key)

        # Fetch social buzz (with its own caching)
        social_buzz = await _fetch_social_buzz(symbol, date)

        return NewsSentiment(
            headlines=headlines,
            key_events=key_events,
            social_buzz=social_buzz,
            articles=articles,
        )

    async def _collect_rss(self, symbol: str) -> list[NewsArticle]:
        """Fetch articles from RSS feeds, extracting title + summary + metadata."""
        all_entries: list[NewsArticle] = []
        source_map = {
            "coindesk.com": "CoinDesk",
            "cointelegraph.com": "CoinTelegraph",
            "decrypt.co": "Decrypt",
        }
        for url in RSS_FEEDS:
            source = next((v for k, v in source_map.items() if k in url), "RSS")
            try:
                feed = await asyncio.to_thread(feedparser.parse, url)
                for e in feed.entries[:15]:
                    title = e.get("title", "")
                    if not title:
                        continue
                    summary = _strip_html(e.get("summary", "") or e.get("description", "") or "")
                    published = e.get("published", "")
                    all_entries.append(
                        NewsArticle(
                            title=title,
                            summary=_truncate(summary),
                            source=source,
                            published=published,
                        )
                    )
            except Exception:
                logger.warning("RSS fetch failed: %s", url, exc_info=True)

        if not all_entries:
            return []

        relevant = [
            a
            for a in all_entries
            if symbol in a.title.lower() or "crypto" in a.title.lower() or "bitcoin" in a.title.lower()
        ]
        if not relevant:
            relevant = all_entries[:10]

        return relevant[:10]

    async def _collect_cryptocompare(self, symbol: str) -> list[NewsArticle]:
        """Fetch latest news from CoinDesk/CryptoCompare with full article content."""
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
                resp_body = resp.json()
                if resp_body.get("Response") == "Error":
                    return []
                raw_articles = resp_body.get("Data", [])
                if not isinstance(raw_articles, list):
                    return []

                title_key = "TITLE" if api_key else "title"
                body_key = "BODY" if api_key else "body"
                source_key = "source" if not api_key else "SOURCE"
                result: list[NewsArticle] = []
                for a in raw_articles[:20]:
                    raw_title = a.get(title_key, "")
                    try:
                        validated = NewsHeadlineResponse(title=raw_title)
                    except ValidationError as exc:
                        logger.warning("News headline schema validation failed, skipping row: %s", exc)
                        continue
                    body = _strip_html(a.get(body_key, "") or "")
                    source_name = a.get(source_key, "") or ""
                    if isinstance(source_name, dict):
                        source_name = source_name.get("name", "")
                    result.append(
                        NewsArticle(
                            title=validated.title,
                            summary=_truncate(body),
                            source=str(source_name) or "CryptoCompare",
                            published="",
                        )
                    )
                return result
        except Exception:
            logger.debug("News fetch failed", exc_info=True)
            return []

"""News sentiment collector — RSS feeds + FinBERT/keyword sentiment + social buzz."""

from __future__ import annotations

import asyncio
import logging

import feedparser
import httpx

from cryptotrader.models import NewsSentiment

logger = logging.getLogger(__name__)

# ── FinBERT sentiment (lazy-loaded, CPU-friendly) ──

_finbert_pipeline = None
_finbert_available: bool | None = None


def _get_finbert():
    """Lazy-load FinBERT pipeline. Returns None if transformers/torch unavailable."""
    global _finbert_pipeline, _finbert_available
    if _finbert_available is False:
        return None
    if _finbert_pipeline is not None:
        return _finbert_pipeline
    try:
        from transformers import pipeline

        _finbert_pipeline = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            truncation=True,
            max_length=512,
        )
        _finbert_available = True
        logger.info("FinBERT loaded successfully")
        return _finbert_pipeline
    except Exception:
        _finbert_available = False
        logger.info("FinBERT unavailable, falling back to keyword sentiment")
        return None


def _score_texts_finbert(texts: list[str]) -> float:
    """Score texts using FinBERT. Returns -1 to +1."""
    pipe = _get_finbert()
    if not pipe or not texts:
        return 0.0
    try:
        results = pipe(texts[:20])  # limit to 20 headlines for speed
        score = 0.0
        for r in results:
            label = r["label"].lower()
            conf = r["score"]
            if label == "positive":
                score += conf
            elif label == "negative":
                score -= conf
        return max(-1.0, min(1.0, score / len(results)))
    except Exception:
        logger.debug("FinBERT scoring failed, falling back to keywords")
        return 0.0


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
    """Score headlines using FinBERT if available, else keyword fallback."""
    if not headlines:
        return 0.0
    finbert_score = _score_texts_finbert(headlines)
    if finbert_score != 0.0 or _finbert_available:
        return finbert_score
    return _score_text(" ".join(headlines))


async def _fetch_social_buzz(symbol: str) -> float:
    """Fetch social buzz score from CoinGecko community data (free, no key).

    Returns a normalized 0-1 score based on community engagement metrics.
    """
    coin_id = _COINGECKO_IDS.get(symbol.upper())
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
        return round(reach * 0.5 + (0.5 * abs(sentiment_shift)), 3)
    except Exception:
        logger.debug("Social buzz fetch failed for %s", symbol, exc_info=True)
        return 0.0


class NewsCollector:
    async def collect(self, pair: str) -> NewsSentiment:
        symbol = pair.split("/")[0]

        # Fetch RSS and social buzz in parallel
        rss_task = self._collect_rss(symbol.lower())
        buzz_task = _fetch_social_buzz(symbol)
        (headlines, score, key_events), social_buzz = await asyncio.gather(
            rss_task,
            buzz_task,
        )

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

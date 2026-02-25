"""News sentiment collector — RSS feeds + keyword sentiment."""

from __future__ import annotations

import asyncio
import logging

import feedparser

from cryptotrader.models import NewsSentiment

logger = logging.getLogger(__name__)

RSS_FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed",
]

POSITIVE = {"bullish", "surge", "rally", "soar", "gain", "rise", "jump", "breakout",
            "adoption", "approval", "partnership", "upgrade", "record", "high", "buy"}
NEGATIVE = {"bearish", "crash", "plunge", "drop", "fall", "dump", "hack", "ban",
            "fraud", "lawsuit", "sell", "fear", "risk", "loss", "decline", "sec"}


def _score_text(text: str) -> float:
    words = set(text.lower().split())
    pos = len(words & POSITIVE)
    neg = len(words & NEGATIVE)
    total = pos + neg
    return (pos - neg) / total if total else 0.0


class NewsCollector:

    async def collect(self, pair: str) -> NewsSentiment:
        all_headlines: list[str] = []
        for url in RSS_FEEDS:
            try:
                feed = await asyncio.to_thread(feedparser.parse, url)
                all_headlines.extend(e.get("title", "") for e in feed.entries[:15])
            except Exception:
                logger.warning("RSS fetch failed: %s", url, exc_info=True)

        if not all_headlines:
            return NewsSentiment()

        symbol = pair.split("/")[0].lower()
        relevant = [h for h in all_headlines if symbol in h.lower() or "crypto" in h.lower() or "bitcoin" in h.lower()]
        if not relevant:
            relevant = all_headlines[:10]

        score = _score_text(" ".join(relevant))
        key_events = [h for h in relevant if any(w in h.lower() for w in ("sec", "etf", "hack", "ban", "approval", "record"))]

        return NewsSentiment(
            headlines=relevant[:10],
            sentiment_score=round(score, 3),
            key_events=key_events[:5],
            social_buzz=0.0,
        )

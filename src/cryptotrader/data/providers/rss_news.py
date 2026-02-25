"""Crypto news collector — aggregates RSS feeds for real-time sentiment."""

from __future__ import annotations

import logging

import feedparser

logger = logging.getLogger(__name__)

RSS_SOURCES = {
    "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "cointelegraph": "https://cointelegraph.com/rss",
    "decrypt": "https://decrypt.co/feed",
}


def fetch_crypto_news(max_per_source: int = 10) -> list[dict]:
    """Fetch recent crypto headlines from RSS feeds. Returns [{title, source, published}]."""
    articles = []
    for name, url in RSS_SOURCES.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_per_source]:
                articles.append({
                    "title": entry.get("title", ""),
                    "source": name,
                    "published": entry.get("published", ""),
                })
        except Exception:
            logger.warning("RSS fetch failed for %s", name, exc_info=True)
    return articles


def headlines_to_text(articles: list[dict], limit: int = 15) -> str:
    """Format articles into a compact string for LLM consumption."""
    lines = []
    for a in articles[:limit]:
        lines.append(f"[{a['source']}] {a['title']}")
    return "\n".join(lines) if lines else "(no recent news available)"

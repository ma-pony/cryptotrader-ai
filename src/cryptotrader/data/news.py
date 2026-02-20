"""News sentiment collector placeholder."""

from __future__ import annotations

from cryptotrader.models import NewsSentiment


class NewsCollector:

    async def collect(self, pair: str) -> NewsSentiment:
        return NewsSentiment()

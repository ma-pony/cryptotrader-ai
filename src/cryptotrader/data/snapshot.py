"""Snapshot aggregator — collects all data sources in parallel."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime

from cryptotrader.data.macro import MacroCollector
from cryptotrader.data.market import MarketCollector
from cryptotrader.data.news import NewsCollector
from cryptotrader.data.onchain import OnchainCollector
from cryptotrader.models import DataSnapshot

logger = logging.getLogger(__name__)


class SnapshotAggregator:
    def __init__(self, providers_config=None) -> None:
        self.market = MarketCollector()
        self.onchain = OnchainCollector(providers_config)
        self.news = NewsCollector()
        self.macro = MacroCollector(providers_config)

    async def collect(
        self,
        pair: str,
        exchange_id: str = "binance",
        timeframe: str = "1h",
        limit: int = 100,
        date: str | None = None,
    ) -> DataSnapshot:
        logger.info("Collecting snapshot: pair=%s exchange=%s tf=%s limit=%d", pair, exchange_id, timeframe, limit)
        market_data, news_data, macro_data = await asyncio.gather(
            self.market.collect(pair, exchange_id, timeframe, limit),
            self.news.collect(pair, date=date),
            self.macro.collect(date=date),
        )
        logger.info(
            "Snapshot market: price=%.2f funding=%.6f volatility=%s",
            market_data.ticker.get("last", 0),
            market_data.funding_rate or 0,
            f"{market_data.volatility:.4f}" if market_data.volatility else "N/A",
        )

        onchain_data = await self.onchain.collect(pair, market_data.funding_rate)
        logger.info("Snapshot complete: market + %d news + onchain + macro", len(news_data.headlines))

        return DataSnapshot(
            timestamp=datetime.now(UTC),
            pair=pair,
            market=market_data,
            onchain=onchain_data,
            news=news_data,
            macro=macro_data,
        )

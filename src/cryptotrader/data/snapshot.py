"""Snapshot aggregator — collects all data sources in parallel."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from cryptotrader.models import DataSnapshot
from cryptotrader.data.market import MarketCollector
from cryptotrader.data.onchain import OnchainCollector
from cryptotrader.data.news import NewsCollector
from cryptotrader.data.macro import MacroCollector


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
    ) -> DataSnapshot:
        market_data, news_data, macro_data = await asyncio.gather(
            self.market.collect(pair, exchange_id, timeframe, limit),
            self.news.collect(pair),
            self.macro.collect(),
        )

        onchain_data = await self.onchain.collect(pair, market_data.funding_rate)

        return DataSnapshot(
            timestamp=datetime.now(UTC),
            pair=pair,
            market=market_data,
            onchain=onchain_data,
            news=news_data,
            macro=macro_data,
        )

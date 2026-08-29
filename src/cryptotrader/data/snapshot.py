"""Snapshot aggregator — collects all data sources in parallel."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import TYPE_CHECKING

from cryptotrader._compat import UTC

if TYPE_CHECKING:
    from cryptotrader.mcp.adapter import MCPAdapter

from cryptotrader.data.macro import MacroCollector
from cryptotrader.data.market import MarketCollector
from cryptotrader.data.news import NewsCollector
from cryptotrader.data.onchain import OnchainCollector
from cryptotrader.models import DataSnapshot

logger = logging.getLogger(__name__)


class SnapshotAggregator:
    def __init__(self, providers_config=None, *, coindesk_api_key: str = "") -> None:
        self.market = MarketCollector()
        self.onchain = OnchainCollector(providers_config)
        self.news = NewsCollector(coindesk_api_key=coindesk_api_key)
        self.macro = MacroCollector(providers_config)

    async def collect(
        self,
        pair: str,
        market_adapter_id: str = "",
        timeframe: str = "1h",
        limit: int = 100,
        date: str | None = None,
        *,
        adapter: MCPAdapter | None = None,
        backtest_mode: bool = False,
        kronos_aux: bool = False,
        kronos_aux_symbol: str = "BTCUSDT",
    ) -> DataSnapshot:
        logger.info("Collecting snapshot: pair=%s adapter=%s tf=%s limit=%d", pair, market_adapter_id, timeframe, limit)
        market_data, news_data, macro_data = await asyncio.gather(
            self.market.collect(pair, market_adapter_id, timeframe, limit),
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

        if kronos_aux:
            from cryptotrader.data.kronos_aux import fetch_kronos_aux

            auxiliary = await fetch_kronos_aux(kronos_aux_symbol)
            if auxiliary["lsr_top_count"] is not None:
                onchain_data.lsr_top_count = auxiliary["lsr_top_count"]
            if auxiliary["premium_close_5d"] is not None:
                market_data.premium_index_5d = auxiliary["premium_close_5d"]
            if auxiliary["spy_btc_corr"] is not None:
                macro_data.spy_btc_corr_30d = auxiliary["spy_btc_corr"]

        if adapter is not None:
            from cryptotrader.pair import Pair

            try:
                symbol = Pair.parse(pair).base
            except (ValueError, NotImplementedError):
                symbol = pair.replace("USDT", "")
            await self._enrich_via_mcp(adapter, symbol, market_data, onchain_data, backtest_mode)

        logger.info("Snapshot complete: market + %d news + onchain + macro", len(news_data.headlines))

        return DataSnapshot(
            timestamp=datetime.now(UTC),
            pair=pair,
            market=market_data,
            onchain=onchain_data,
            news=news_data,
            macro=macro_data,
        )

    async def _enrich_via_mcp(
        self,
        adapter: MCPAdapter,
        symbol: str,
        market_data,
        onchain_data,
        backtest_mode: bool,
    ) -> None:
        """Supplement snapshot with MCP-sourced derivatives data when adapter is available."""
        from cryptotrader.data.providers.binance import fetch_derivatives_binance

        try:
            derivatives = await adapter.call(
                "binance_derivatives",
                {"symbol": symbol},
                backtest_mode=backtest_mode,
                python_fallback=fetch_derivatives_binance,
                fallback_args={"symbol": symbol},
                zero_value={"open_interest": 0.0, "long_short_ratio": 0.0},
            )
            if derivatives.get("open_interest") and not onchain_data.open_interest:
                onchain_data.open_interest = derivatives["open_interest"]
            if derivatives.get("liquidations_24h") and not onchain_data.liquidations_24h:
                onchain_data.liquidations_24h = derivatives["liquidations_24h"]
        except Exception:
            logger.info("MCP enrichment failed for derivatives", exc_info=True)

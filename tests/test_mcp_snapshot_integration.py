"""Tests for MCPAdapter integration with SnapshotAggregator (T034)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def mock_providers():
    """Mock all data providers so SnapshotAggregator.collect() doesn't hit network."""
    market_data = MagicMock()
    market_data.ticker = {"last": 100_000.0}
    market_data.funding_rate = 0.0001
    market_data.volatility = 0.02
    market_data.orderbook_imbalance = 0.1
    market_data.ohlcv = None

    onchain_data = MagicMock()
    onchain_data.open_interest = 0.0
    onchain_data.liquidations_24h = {}
    onchain_data.derivatives_extra = None

    news_data = MagicMock()
    news_data.headlines = ["BTC rallies"]

    macro_data = MagicMock()

    with (
        patch("cryptotrader.data.snapshot.MarketCollector") as mc,
        patch("cryptotrader.data.snapshot.OnchainCollector") as oc,
        patch("cryptotrader.data.snapshot.NewsCollector") as nc,
        patch("cryptotrader.data.snapshot.MacroCollector") as mac,
    ):
        mc.return_value.collect = AsyncMock(return_value=market_data)
        oc.return_value.collect = AsyncMock(return_value=onchain_data)
        nc.return_value.collect = AsyncMock(return_value=news_data)
        mac.return_value.collect = AsyncMock(return_value=macro_data)
        yield {
            "market": market_data,
            "onchain": onchain_data,
            "news": news_data,
            "macro": macro_data,
        }


@pytest.mark.usefixtures("mock_providers")
async def test_collect_without_adapter_unchanged():
    """adapter=None should produce identical behavior to pre-MCP code."""
    from cryptotrader.data.snapshot import SnapshotAggregator

    agg = SnapshotAggregator()
    snapshot = await agg.collect("BTC/USDT", "binance")
    assert snapshot.pair == "BTC/USDT"
    assert snapshot.market.ticker["last"] == 100_000.0


async def test_collect_with_adapter_calls_mcp(mock_providers):
    """adapter provided should trigger MCP enrichment via adapter.call()."""
    from cryptotrader.data.snapshot import SnapshotAggregator

    adapter = MagicMock()
    adapter.call = AsyncMock(return_value={"open_interest": 5_000_000.0, "long_short_ratio": 1.2})

    agg = SnapshotAggregator()
    await agg.collect("BTC/USDT", "binance", adapter=adapter, backtest_mode=False)

    adapter.call.assert_called_once()
    call_args = adapter.call.call_args
    assert call_args[0][0] == "binance_derivatives"
    assert mock_providers["onchain"].open_interest == 5_000_000.0


async def test_collect_adapter_does_not_overwrite_existing_oi(mock_providers):
    """If onchain collector already has OI data, MCP should not overwrite it."""
    from cryptotrader.data.snapshot import SnapshotAggregator

    mock_providers["onchain"].open_interest = 3_000_000.0

    adapter = MagicMock()
    adapter.call = AsyncMock(return_value={"open_interest": 5_000_000.0})

    agg = SnapshotAggregator()
    await agg.collect("BTC/USDT", "binance", adapter=adapter, backtest_mode=False)

    assert mock_providers["onchain"].open_interest == 3_000_000.0


async def test_collect_adapter_failure_is_silent(mock_providers):
    """MCP enrichment failure should not crash the snapshot collection."""
    from cryptotrader.data.snapshot import SnapshotAggregator

    adapter = MagicMock()
    adapter.call = AsyncMock(side_effect=RuntimeError("MCP server down"))

    agg = SnapshotAggregator()
    snapshot = await agg.collect("BTC/USDT", "binance", adapter=adapter, backtest_mode=False)

    assert snapshot.pair == "BTC/USDT"
    assert mock_providers["onchain"].open_interest == 0.0

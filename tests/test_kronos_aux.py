"""Offline tests for Kronos auxiliary market data."""

from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from cryptotrader.data.kronos_aux import _closed_interval_rows, fetch_premium_close_5d
from cryptotrader.models import MacroData, MarketData, NewsSentiment, OnchainData


def test_closed_interval_rows_excludes_forming_period():
    interval_ms = 4 * 60 * 60 * 1000
    now_ms = 3 * interval_ms + 60_000
    rows = [[i * interval_ms, i] for i in range(4)]

    assert _closed_interval_rows(rows, interval_ms, now_ms) == rows[:3]


@pytest.mark.asyncio
async def test_premium_average_uses_closed_klines_only():
    interval_ms = 4 * 60 * 60 * 1000
    rows = [
        [0, "0", "0", "0", "1.0"],
        [interval_ms, "0", "0", "0", "2.0"],
        [2 * interval_ms, "0", "0", "0", "3.0"],
        [3 * interval_ms, "0", "0", "0", "99.0"],
    ]

    with (
        patch("cryptotrader.data.kronos_aux._get_json", new=AsyncMock(return_value=rows)),
        patch("time.time", return_value=(3 * interval_ms + 60_000) / 1000),
    ):
        result = await fetch_premium_close_5d()

    assert result == pytest.approx(2.0)


@pytest.mark.asyncio
async def test_snapshot_aggregator_materializes_requested_kronos_aux_data():
    from cryptotrader.data.snapshot import SnapshotAggregator

    market = MarketData(
        "BTC/USDT:USDT",
        pd.DataFrame({"close": [100.0]}),
        {"last": 100.0},
        0.0,
        0.0,
        0.0,
    )
    aggregator = SnapshotAggregator()
    aggregator.market.collect = AsyncMock(return_value=market)
    aggregator.news.collect = AsyncMock(return_value=NewsSentiment())
    aggregator.macro.collect = AsyncMock(return_value=MacroData())
    aggregator.onchain.collect = AsyncMock(return_value=OnchainData())
    auxiliary = {"lsr_top_count": 1.7, "premium_close_5d": 0.002, "spy_btc_corr": 0.4}

    with patch(
        "cryptotrader.data.kronos_aux.fetch_kronos_aux",
        new=AsyncMock(return_value=auxiliary),
    ) as fetch:
        snapshot = await aggregator.collect("BTC/USDT:USDT", kronos_aux=True)

    fetch.assert_awaited_once_with("BTCUSDT")
    assert snapshot.onchain.lsr_top_count == 1.7
    assert snapshot.market.premium_index_5d == 0.002
    assert snapshot.macro.spy_btc_corr_30d == 0.4

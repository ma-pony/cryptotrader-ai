"""point-in-time 多周期 SignalContext 物化契约。"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pandas as pd
import pytest

from cryptotrader.models import DataSnapshot, MacroData, MarketData, NewsSentiment, OnchainData
from cryptotrader.pair import Pair
from cryptotrader.signals.models import CandleRequirement, DataRequirements

PAIR = Pair.parse("BTC/USDT:USDT")
AS_OF = datetime(2026, 1, 2, tzinfo=UTC)


def _market(rows: int = 20, *, start="2026-01-01", price: float = 100.0) -> MarketData:
    index = pd.date_range(start, periods=rows, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": [price] * rows,
            "high": [price + 1.0] * rows,
            "low": [price - 1.0] * rows,
            "close": [price] * rows,
            "volume": [10.0] * rows,
        },
        index=index,
    )
    return MarketData("BTC/USDT:USDT", frame, {"last": price}, 0.0, 0.0, 0.0)


def _snapshot(timeframe: str, *, timestamp=None, market=None) -> DataSnapshot:
    return DataSnapshot(
        timestamp=timestamp or datetime(2026, 1, 2, tzinfo=UTC),
        pair="BTC/USDT:USDT",
        market=market or _market(start="2026-01-01"),
        onchain=OnchainData(),
        news=NewsSentiment(),
        macro=MacroData(),
    )


class FakeSnapshotAggregator:
    def __init__(self) -> None:
        self.calls = []

    async def collect(self, **kwargs):
        self.calls.append(kwargs)
        return _snapshot(kwargs["timeframe"])


class FakeMarketCollector:
    def __init__(self) -> None:
        self.calls = []

    async def collect(self, pair, exchange_id, timeframe, limit):
        self.calls.append((pair, timeframe, limit))
        return _market(rows=limit, start="2025-12-01", price=101.0)


def _requirements() -> DataRequirements:
    return DataRequirements(
        candles=(CandleRequirement("1h", 20), CandleRequirement("4h", 30)),
        onchain=True,
        news=True,
        macro=True,
        kronos_aux=True,
    )


@pytest.mark.asyncio
async def test_live_provider_materializes_each_required_timeframe_once():
    from cryptotrader.signals.context import LiveSignalContextProvider

    aggregator = FakeSnapshotAggregator()
    market = FakeMarketCollector()
    provider = LiveSignalContextProvider(aggregator, market, exchange_id="okx", default_timeframe="1h")

    context = await provider.collect(PAIR, AS_OF, _requirements())

    assert set(context.snapshots) == {"1h", "4h"}
    assert {snapshot.timestamp for snapshot in context.snapshots.values()} == {context.as_of}
    assert market.calls == [("BTC/USDT:USDT", "4h", 30)]
    assert len(aggregator.calls) == 1
    assert aggregator.calls[0]["kronos_aux"] is True
    assert context.current_price == 100.0
    assert context.atr == pytest.approx(2.0)


@pytest.mark.asyncio
async def test_historical_provider_never_includes_future_bars():
    from cryptotrader.signals.context import HistoricalSignalContextProvider

    as_of = datetime(2025, 1, 2, tzinfo=UTC)
    market = _market(rows=4, start="2025-01-01", price=80.0)
    history = {"4h": replace(_snapshot("4h", timestamp=as_of, market=market))}
    provider = HistoricalSignalContextProvider(history, default_timeframe="4h")

    context = await provider.collect(
        PAIR,
        as_of,
        DataRequirements(candles=(CandleRequirement("4h", 10),)),
    )

    assert context.as_of == as_of
    assert context.snapshots["4h"].market.ohlcv.index.max().to_pydatetime() <= as_of
    assert context.current_price == 80.0

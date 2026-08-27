"""point-in-time 多周期 SignalContext 物化契约。"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pandas as pd
import pytest

from cryptotrader.decision.models import CycleRequest
from cryptotrader.models import DataSnapshot, MacroData, MarketData, NewsSentiment, OnchainData
from cryptotrader.pair import Pair
from cryptotrader.signals.models import CandleRequirement, DataRequirements
from tests.factories.signal_fusion import position


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


class FakePortfolioReader:
    def __init__(self) -> None:
        self.total_value = 10_000.0
        self.current_position = position()

    async def read(self, request, current_price):
        current = self.current_position
        amount = current.amount if current.side != "short" else -current.amount
        return {
            "total_value": self.total_value,
            "positions": {
                request.pair.canonical(): {
                    "amount": amount,
                    "side": current.side,
                    "avg_price": current.avg_price,
                    "unrealized_pnl": current.unrealized_pnl,
                }
            }
            if current.side != "flat"
            else {},
        }


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
    provider = LiveSignalContextProvider(aggregator, market, FakePortfolioReader(), default_timeframe="1h")

    context = await provider.collect(
        CycleRequest(Pair.parse("BTC/USDT:USDT"), "paper", "okx"),
        _requirements(),
    )

    assert set(context.snapshots) == {"1h", "4h"}
    assert {snapshot.timestamp for snapshot in context.snapshots.values()} == {context.as_of}
    assert market.calls == [("BTC/USDT:USDT", "4h", 30)]
    assert len(aggregator.calls) == 1
    assert aggregator.calls[0]["kronos_aux"] is True
    assert context.current_price == 100.0
    assert context.atr == pytest.approx(2.0)


@pytest.mark.asyncio
async def test_live_provider_normalizes_current_position_to_maximum_position_ratio():
    from cryptotrader.signals.context import LiveSignalContextProvider

    portfolio = FakePortfolioReader()
    portfolio.current_position = position("long", amount=2.0, size_ratio=0.0, avg_price=90.0)
    provider = LiveSignalContextProvider(
        FakeSnapshotAggregator(),
        FakeMarketCollector(),
        portfolio,
        default_timeframe="1h",
        max_single_pct=0.1,
    )

    context = await provider.collect(CycleRequest(Pair.parse("BTC/USDT:USDT"), "paper"), _requirements())

    assert context.current_position.amount == 2.0
    assert context.current_position.size_ratio == pytest.approx(0.2)


@pytest.mark.asyncio
async def test_historical_provider_never_includes_future_bars():
    from cryptotrader.signals.context import HistoricalSignalContextProvider

    as_of = datetime(2025, 1, 2, tzinfo=UTC)
    market = _market(rows=4, start="2025-01-01", price=80.0)
    history = {"4h": replace(_snapshot("4h", timestamp=as_of, market=market))}
    provider = HistoricalSignalContextProvider(history, default_timeframe="4h", equity=5_000.0)

    context = await provider.collect(
        CycleRequest(Pair.parse("BTC/USDT:USDT"), "backtest", as_of=as_of),
        DataRequirements(candles=(CandleRequirement("4h", 10),)),
    )

    assert context.as_of == as_of
    assert context.snapshots["4h"].market.ohlcv.index.max().to_pydatetime() <= as_of
    assert context.current_price == 80.0


@pytest.mark.asyncio
async def test_refresh_execution_state_only_updates_live_portfolio_fields():
    from cryptotrader.signals.context import LiveSignalContextProvider

    portfolio = FakePortfolioReader()
    provider = LiveSignalContextProvider(
        FakeSnapshotAggregator(),
        FakeMarketCollector(),
        portfolio,
        default_timeframe="1h",
    )
    original = await provider.collect(CycleRequest(Pair.parse("BTC/USDT:USDT"), "paper"), _requirements())
    portfolio.current_position = position("long", 0.2, 0.2)
    portfolio.total_value = 12_000.0

    refreshed = await provider.refresh_execution_state(original)

    assert refreshed.snapshots is original.snapshots
    assert refreshed.as_of == original.as_of
    assert refreshed.atr == original.atr
    assert refreshed.equity == 12_000.0
    assert refreshed.current_position.side == "long"


@pytest.mark.asyncio
async def test_historical_refresh_preserves_original_context():
    from cryptotrader.signals.context import HistoricalSignalContextProvider

    as_of = datetime(2025, 1, 2, tzinfo=UTC)
    provider = HistoricalSignalContextProvider(
        {"4h": _snapshot("4h", timestamp=as_of, market=_market(start="2025-01-01"))},
        default_timeframe="4h",
    )
    original = await provider.collect(
        CycleRequest(Pair.parse("BTC/USDT:USDT"), "backtest", as_of=as_of),
        DataRequirements(candles=(CandleRequirement("4h", 10),)),
    )

    assert await provider.refresh_execution_state(original) is original

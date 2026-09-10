"""Configured exit candles determine protection prices in live analysis and replay."""

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from cryptotrader.backtest.engine import BacktestEngine
from cryptotrader.cycle_events import NullCycleEventSink
from cryptotrader.decision.analysis import SignalAnalysisService
from cryptotrader.decision.engine import DecisionEngine
from cryptotrader.decision.exit_policy import AtrExitPolicy
from cryptotrader.market_sources.default import DefaultMarketDataSource
from cryptotrader.market_sources.protocol import HistoricalCandle
from cryptotrader.market_sources.registry import MarketSourceRegistry
from cryptotrader.models import DataSnapshot, MacroData, MarketData, NewsSentiment, OnchainData
from cryptotrader.pair import Pair
from cryptotrader.runtime_config.models import RuntimeConfigSnapshot, SignalComponentConfig
from cryptotrader.signals.fusion import WeightedSignalFusion
from cryptotrader.signals.models import CandleRequirement, ComponentSignal, DataRequirements
from cryptotrader.signals.presentation import interval_delta
from cryptotrader.signals.registry import SignalComponentRegistry
from cryptotrader.signals.runner import ComponentRunner
from tests.factories.runtime_config import market_config, runtime_document, signal_config

AS_OF = datetime(2024, 1, 1, 4, tzinfo=UTC)
PAIR = Pair.parse("BTC/USDT:USDT")
# Constant closes make true range exactly high - low: ATR is 2, 20 or 6.
EXIT_CASES = [("1h", 2.0, 96.0, 108.0), ("4h", 20.0, 60.0, 180.0), ("15m", 6.0, 88.0, 124.0)]


class CandleFeed:
    async def read_candles(self, pair, adapter, timeframe, start, end, as_of):
        delta = interval_delta(timeframe)
        width = Decimal({"1h": "1", "4h": "10", "15m": "3"}[timeframe])
        rows = []
        opened = start
        while opened + delta <= min(end, as_of):
            rows.append(
                HistoricalCandle(
                    open_time=opened,
                    open=Decimal("100"),
                    high=Decimal("100") + width,
                    low=Decimal("100") - width,
                    close=Decimal("100"),
                    volume=Decimal("10"),
                )
            )
            opened += delta
        return tuple(rows)

    async def collect(self, pair, adapter, timeframe, limit):
        bars = await self.read_candles(
            pair, adapter, timeframe, AS_OF - limit * interval_delta(timeframe), AS_OF, AS_OF
        )
        frame = pd.DataFrame(
            {name: [float(getattr(bar, name)) for bar in bars] for name in ("open", "high", "low", "close", "volume")},
            index=pd.DatetimeIndex([bar.open_time for bar in bars]),
        )
        return MarketData(pair, frame, {"last": 100.0}, 0.0, 0.0, 0.0)


class SnapshotFeed:
    def __init__(self):
        self.market = CandleFeed()

    async def collect(self, *, pair, market_adapter_id, timeframe, limit, **kwargs):
        market = await self.market.collect(pair, market_adapter_id, timeframe, limit)
        return DataSnapshot(AS_OF, pair, market, OnchainData(), NewsSentiment(), MacroData())


class RecordingComponent:
    display_name = "Offline fixture"
    description = "Fixed market opinion with a component-specific input timeframe"

    def __init__(self, timeframe):
        self.id = f"fixture-{timeframe}"
        self.timeframe = timeframe
        self.contexts = []

    def requirements(self):
        return DataRequirements(candles=(CandleRequirement(self.timeframe, 20),))

    async def evaluate(self, context):
        self.contexts.append(context)
        return ComponentSignal(self.id, "long", 0.4, "fixed offline opinion")


def setup_case(exit_timeframe, reference_timeframe, component_order):
    components = tuple(RecordingComponent(timeframe) for timeframe in component_order)
    document = runtime_document(
        market_data=market_config(timeframe=reference_timeframe, parameters={"timeframe": exit_timeframe, "limit": 20}),
        signals=signal_config(
            components=tuple(
                SignalComponentConfig(component_id=item.id, enabled=True, weight=0.5) for item in components
            )
        ),
    )
    snapshot = RuntimeConfigSnapshot(7, document, AS_OF)
    source = DefaultMarketDataSource(document.market_data, aggregator=SnapshotFeed(), clock=lambda: AS_OF)
    return snapshot, source, SignalComponentRegistry(components), components


async def analyze(snapshot, source, registry):
    service = SignalAnalysisService(
        market_source=source,
        registry=registry,
        runner=ComponentRunner(NullCycleEventSink()),
        fusion=WeightedSignalFusion(),
        decisions=DecisionEngine(),
    )
    result = await service.analyze(PAIR, snapshot, AS_OF)
    assert result.failure is None
    return result, AtrExitPolicy().build_plan(
        result.context,
        result.target_position,
        result.component_signals,
        result.fused_signal,
        snapshot.document.signals.to_profile(snapshot.revision),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("component_order", [("4h", "1h"), ("1h", "4h")])
@pytest.mark.parametrize("reference_timeframe", ["1h", "15m"])
@pytest.mark.parametrize(("exit_timeframe", "expected_atr", "stop", "take_profit"), EXIT_CASES)
async def test_live_exit_prices_follow_configured_period_not_component_order_or_evaluation_window(
    component_order, reference_timeframe, exit_timeframe, expected_atr, stop, take_profit
):
    snapshot, source, registry, _ = setup_case(exit_timeframe, reference_timeframe, component_order)

    result, plan = await analyze(snapshot, source, registry)

    assert result.context.current_price == 100.0
    assert result.context.atr == pytest.approx(expected_atr)
    assert plan.stop_loss == pytest.approx(stop)
    assert plan.take_profit == pytest.approx(take_profit)
    assert result.context.evaluation_reference.interval == reference_timeframe


@pytest.mark.asyncio
@pytest.mark.parametrize("reference_timeframe", ["1h", "15m"])
@pytest.mark.parametrize(("exit_timeframe", "expected_atr", "stop", "take_profit"), EXIT_CASES)
async def test_replay_collects_exit_candles_and_matches_live_protection_prices(
    monkeypatch, reference_timeframe, exit_timeframe, expected_atr, stop, take_profit
):
    snapshot, source, registry, components = setup_case(exit_timeframe, reference_timeframe, ("4h", "1h"))
    monkeypatch.setattr("cryptotrader.backtest.historical_data.fetch_fear_greed", AsyncMock(return_value={}))
    replay = await BacktestEngine(
        PAIR.canonical(),
        "2024-01-01",
        "2024-01-01T04:00:00",
        interval="4h",
        lookback=20,
        snapshot=snapshot,
        signal_registry=registry,
        market_registry=MarketSourceRegistry((source,)),
    ).run()

    assert len(replay.cycle_records) == 1
    record = replay.cycle_records[0]
    assert record.run.failure is None
    assert components[0].contexts[0].atr == pytest.approx(expected_atr)
    proposal = record.book_results[0].proposal
    assert proposal is not None
    assert proposal.ready
    assert proposal.connection_plans[0].stop_loss == Decimal(str(stop))
    assert proposal.connection_plans[0].take_profit == Decimal(str(take_profit))

    _, live_plan = await analyze(snapshot, source, registry)
    assert record.target_position == live_plan.target
    assert proposal.connection_plans[0].stop_loss == Decimal(str(live_plan.stop_loss))
    assert proposal.connection_plans[0].take_profit == Decimal(str(live_plan.take_profit))

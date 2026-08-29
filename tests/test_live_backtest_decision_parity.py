"""Historical Paper and production cycles share one decision/proposal contract."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from cryptotrader.backtest.cache import _TF_MS
from cryptotrader.backtest.engine import BacktestEngine
from cryptotrader.cycle_events import NullCycleEventSink
from cryptotrader.decision.models import CycleRequest
from cryptotrader.execution.models import ConnectionAllocation, ExecutionBook
from cryptotrader.journal.store import MultiVenueCycleStore
from cryptotrader.market_sources.registry import MarketSourceRegistry
from cryptotrader.pair import Pair
from cryptotrader.runtime import build_runtime
from cryptotrader.runtime_config.models import (
    MarketDataConfig,
    RuntimeConfigSnapshot,
    SignalComponentConfig,
    SignalConfig,
    SystemConfig,
)
from cryptotrader.signals.models import CandleRequirement, ComponentSignal, DataRequirements, SignalContext
from cryptotrader.signals.registry import SignalComponentRegistry
from cryptotrader.venues.paper import PaperVenueAdapter
from cryptotrader.venues.registry import VenueAdapterRegistry
from tests.factories.runtime_config import connection, runtime_document

PAIR = Pair.parse("BTC/USDT:USDT")


class _DeterministicComponent:
    id = "fixture"
    display_name = "Fixture"
    description = "deterministic parity signal"

    @staticmethod
    def requirements() -> DataRequirements:
        return DataRequirements(candles=(CandleRequirement("1h", 20),))

    async def evaluate(self, context: SignalContext) -> ComponentSignal:
        return ComponentSignal(self.id, "long", 1.0, context.pair.canonical())


class _ProductionMarketSource:
    id = "default"

    def requirements(self) -> DataRequirements:
        return DataRequirements()

    async def collect(self, pair, as_of, requirements) -> SignalContext:
        return SignalContext(
            pair=pair,
            as_of=as_of,
            market_data_source_id=self.id,
            market_type=pair.market_type,
            current_price=101.0,
            atr=1.0,
            snapshots={},
        )


class _FrozenRepository:
    database_url = None

    def __init__(self, snapshot) -> None:
        self.snapshot = snapshot

    async def get_or_create(self):
        return self.snapshot


def _snapshot() -> RuntimeConfigSnapshot:
    paper = connection("production-paper", "paper", adapter_id="paper")
    book = ExecutionBook(
        "production",
        "Production Paper",
        "simulated",
        True,
        False,
        (ConnectionAllocation("production-paper", True, 1.0),),
    )
    return RuntimeConfigSnapshot(
        44,
        runtime_document(
            connections=(paper,),
            books=(book,),
            system=SystemConfig(active=True),
            market_data=MarketDataConfig(source_id="default", parameters={"timeframe": "1h", "limit": 20}),
            signals=SignalConfig(
                components=(SignalComponentConfig(component_id="fixture", enabled=True, weight=1.0),),
                neutral_threshold=0.2,
                max_target_ratio=1.0,
                atr_stop_multiplier=2.0,
                reward_ratio=2.0,
            ),
        ),
        datetime.now(UTC),
    )


@pytest.mark.asyncio
async def test_historical_paper_backtest_matches_production_cycle_target_and_one_hundred_percent_target(monkeypatch):
    snapshot = _snapshot()
    registry = SignalComponentRegistry((_DeterministicComponent(),))
    engine = BacktestEngine(
        "BTC/USDT:USDT",
        "2024-01-01",
        "2024-01-02",
        interval="1h",
        initial_capital=10_000.0,
        snapshot=snapshot,
        signal_registry=registry,
    )

    async def historical_bars(requirements):
        for requirement in requirements.candles:
            interval_ms = _TF_MS[requirement.timeframe]
            start_ms = engine.start_ms - requirement.limit * interval_ms
            engine._candles_by_timeframe[requirement.timeframe] = [
                [start_ms + index * interval_ms, 100.0 + index, 102.0 + index, 99.0 + index, 101.0 + index, 10.0]
                for index in range(requirement.limit + 3)
            ]
        engine._candles = engine._candles_by_timeframe[engine.interval]

    monkeypatch.setattr(engine, "_fetch_historical_data", historical_bars)
    backtest = await engine.run()
    historical_record = backtest.cycle_records[0]

    runtime = await build_runtime(
        repository=_FrozenRepository(snapshot),
        snapshot=snapshot,
        signal_registry=registry,
        venue_registry=VenueAdapterRegistry((PaperVenueAdapter(),)),
        market_registry=MarketSourceRegistry((_ProductionMarketSource(),)),
        event_sink=NullCycleEventSink(),
    )
    try:
        async with runtime.cycle_lease() as cycle:
            cycle.journal = MultiVenueCycleStore()
            await runtime.sessions["production-paper"].set_quote(PAIR, Decimal("101"))
            outcome = await cycle.run(CycleRequest(PAIR))
            production_record = await cycle.journal.get(outcome.cycle_id)
    finally:
        await runtime.close()

    assert production_record is not None
    assert historical_record.target_position == production_record.target_position
    proposal = historical_record.book_results[0].proposal
    assert proposal is not None
    assert proposal.risk.connection_weights == (Decimal("1.0"),)
    assert tuple(item.connection_id for item in proposal.risk.connection_targets) == ("backtest-paper",)
    assert proposal.risk.connection_targets[0].target_signed_notional == proposal.target_exposure * Decimal("10000")

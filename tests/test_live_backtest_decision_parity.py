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
    InfrastructureConfig,
    MarketDataConfig,
    RuntimeConfigSnapshot,
    SignalComponentConfig,
    SignalConfig,
)
from cryptotrader.signals.models import CandleRequirement, ComponentSignal, DataRequirements, SignalContext
from cryptotrader.signals.registry import SignalComponentRegistry
from cryptotrader.venues.paper import PaperVenueAdapter
from cryptotrader.venues.registry import VenueAdapterRegistry
from tests.factories.runtime_config import connection, runtime_document

PAIR = Pair.parse("BTC/USDT:USDT")


class _RecordingDeterministicComponent:
    id = "fixture"
    display_name = "Fixture"
    description = "deterministic parity signal"

    def __init__(self) -> None:
        self.contexts: list[SignalContext] = []

    @staticmethod
    def requirements() -> DataRequirements:
        return DataRequirements(candles=(CandleRequirement("1h", 20),))

    async def evaluate(self, context: SignalContext) -> ComponentSignal:
        self.contexts.append(context)
        confidence = 1.0 if context.current_price > context.atr else 0.5
        return ComponentSignal(self.id, "long", confidence, context.pair.canonical())


class _ProductionMarketSource:
    id = "default"

    def __init__(self, template: SignalContext) -> None:
        self.template = template

    def requirements(self) -> DataRequirements:
        return DataRequirements()

    async def read_candles(self, pair, timeframe, start, end, as_of):
        return ()

    async def collect(self, pair, as_of, requirements) -> SignalContext:
        return SignalContext(
            pair=self.template.pair,
            as_of=as_of,
            market_data_source_id=self.id,
            market_type=self.template.market_type,
            current_price=self.template.current_price,
            atr=self.template.atr,
            snapshots=self.template.snapshots,
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
            infrastructure=InfrastructureConfig(redis_url="redis://runtime-test:6379/0"),
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
async def test_historical_paper_backtest_matches_production_cycle_target_and_one_hundred_percent_target(
    monkeypatch, tmp_path
):
    snapshot = _snapshot()
    historical_component = _RecordingDeterministicComponent()
    production_component = _RecordingDeterministicComponent()
    engine = BacktestEngine(
        "BTC/USDT:USDT",
        "2024-01-01",
        "2024-01-02",
        interval="1h",
        initial_capital=10_000.0,
        snapshot=snapshot,
        signal_registry=SignalComponentRegistry((historical_component,)),
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

    def reject_redis(*_):
        raise AssertionError("backtest cannot open Redis")

    monkeypatch.setattr("cryptotrader.cycle_lock.RedisStateManager", reject_redis)
    backtest = await engine.run()
    historical_record = backtest.cycle_records[0]
    historical_context = historical_component.contexts[0]

    from cryptotrader.accounts.store import AccountStore
    from tests.test_book_execution_ownership import StrictRedis

    redis = StrictRedis()
    monkeypatch.setattr("cryptotrader.cycle_lock.RedisStateManager", lambda _: redis)
    repository = _FrozenRepository(snapshot)
    repository.database_url = f"sqlite+aiosqlite:///{tmp_path / 'parity.db'}"
    from cryptotrader.migrations.workbench import migrate_workbench_schema

    await migrate_workbench_schema(repository.database_url)
    repository.account_store = AccountStore(repository.database_url)
    runtime = await build_runtime(
        repository=repository,
        snapshot=snapshot,
        signal_registry=SignalComponentRegistry((production_component,)),
        venue_registry=VenueAdapterRegistry((PaperVenueAdapter(),)),
        market_registry=MarketSourceRegistry((_ProductionMarketSource(historical_context),)),
        event_sink=NullCycleEventSink(),
    )
    try:
        async with runtime.cycle_lease() as cycle:
            cycle.journal = MultiVenueCycleStore()
            await runtime.sessions["production-paper"].set_quote(PAIR, Decimal(str(historical_context.current_price)))
            outcome = await cycle.run(CycleRequest(PAIR))
            production_record = await cycle.journal.get(outcome.cycle_id)
    finally:
        await runtime.close()

    assert production_record is not None
    production_context = production_component.contexts[0]
    assert historical_context.pair == production_context.pair
    assert historical_context.market_type == production_context.market_type
    assert historical_context.current_price == production_context.current_price
    assert historical_context.atr == production_context.atr
    assert historical_context.snapshots == production_context.snapshots
    assert len(historical_record.component_signals) == len(production_record.component_signals) == 1
    for historical_signal, production_signal in zip(
        historical_record.component_signals, production_record.component_signals, strict=True
    ):
        assert historical_signal.component_id == production_signal.component_id == "fixture"
        assert historical_signal.status == production_signal.status == "completed"
        assert historical_signal.direction == production_signal.direction
        assert historical_signal.confidence == production_signal.confidence
        assert historical_signal.reasoning == production_signal.reasoning
        assert historical_signal.details == production_signal.details
        assert historical_signal.blocks == production_signal.blocks
        historical_reference = historical_signal.evaluation_reference
        production_reference = production_signal.evaluation_reference
        assert historical_reference is not None
        assert production_reference is not None
        assert historical_reference.reference_time == production_reference.reference_time
        assert historical_reference.reference_price == production_reference.reference_price
        assert historical_reference.due_at == production_reference.due_at
        assert historical_reference.interval == production_reference.interval == "1h"
        assert historical_reference.market_source_id == "default"
        assert production_reference.market_source_id == "default"
        for signal in (historical_signal, production_signal):
            assert type(signal.duration_ms) is int
            assert signal.duration_ms >= 0
    assert historical_record.config_revision == production_record.config_revision == snapshot.revision
    assert (
        historical_record.book_results[0].portfolio_before.total_equity
        == production_record.book_results[0].portfolio_before.total_equity
    )
    assert historical_record.target_position == production_record.target_position
    proposal = historical_record.book_results[0].proposal
    assert proposal is not None
    assert proposal.risk.connection_weights == (Decimal("1.0"),)
    assert tuple(item.connection_id for item in proposal.risk.connection_targets) == ("backtest-paper",)
    assert proposal.risk.connection_targets[0].target_signed_notional == proposal.target_exposure * Decimal("10000")
    production_proposal = production_record.book_results[0].proposal
    assert production_proposal is not None
    assert production_proposal.risk.connection_weights == proposal.risk.connection_weights
    assert tuple(item.connection_id for item in production_proposal.risk.connection_targets) == ("production-paper",)
    assert (
        production_proposal.risk.connection_targets[0].target_signed_notional
        == proposal.risk.connection_targets[0].target_signed_notional
        == production_proposal.target_exposure * Decimal("10000")
    )

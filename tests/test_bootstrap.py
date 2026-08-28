"""生产依赖只能装配为一个 TradingCycle。"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from cryptotrader.config import (
    AppConfig,
    SignalPluginsConfig,
    SignalProfileDefaultsConfig,
)
from cryptotrader.profiles.models import ComponentWeight


def _snapshot(price: float = 100.0):
    from cryptotrader.models import DataSnapshot, MacroData, MarketData, NewsSentiment, OnchainData

    index = pd.date_range("2026-01-01", periods=100, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": [price] * len(index),
            "high": [price + 1.0] * len(index),
            "low": [price - 1.0] * len(index),
            "close": [price] * len(index),
            "volume": [10.0] * len(index),
        },
        index=index,
    )
    return DataSnapshot(
        timestamp=datetime(2026, 1, 5, tzinfo=UTC),
        pair="BTC/USDT:USDT",
        market=MarketData("BTC/USDT:USDT", frame, {"last": price}, 0.0, 0.0, 0.0),
        onchain=OnchainData(),
        news=NewsSentiment(),
        macro=MacroData(),
    )


class _StaticRunner:
    def __init__(self) -> None:
        from tests.factories.signal_fusion import signal

        self.signals = (
            signal("kronos", "long", 0.8),
            signal("llm_committee", "long", 0.6),
        )
        self.calls = 0

    async def run(self, _components, _context):
        self.calls += 1
        return self.signals


class _PassRisk:
    def __init__(self) -> None:
        self.calls = []

    async def check(self, risk_request, portfolio):
        from cryptotrader.risk.models import RiskDecision

        self.calls.append((risk_request, portfolio))
        return RiskDecision(True, risk_request.plan)


async def _build_paper_hitl_cycle(latest_price: float | Exception):
    from cryptotrader.bootstrap import SeededProfileRepository, build_trading_cycle
    from cryptotrader.execution.planner import ExecutionPlanner
    from cryptotrader.hitl.store import ApprovalStore
    from cryptotrader.journal.store import CycleJournalStore
    from tests.factories.signal_fusion import profile

    class _RecordingPlanner(ExecutionPlanner):
        def __init__(self) -> None:
            super().__init__(max_single_pct=0.2)
            self.context = None

        def plan(self, signal_context, trade_plan):
            self.context = signal_context
            return super().plan(signal_context, trade_plan)

    cycle = build_trading_cycle(
        AppConfig(),
        mode="paper",
        profile_repository=SeededProfileRepository(None, profile(hitl=True)),
        approval_store=ApprovalStore(),
        journal_store=CycleJournalStore(),
    )
    snapshot = _snapshot()
    cycle.contexts.aggregator.collect = AsyncMock(return_value=snapshot)
    cycle.contexts.market.collect = AsyncMock(return_value=snapshot.market)
    cycle.contexts.market.latest_price = (
        AsyncMock(side_effect=latest_price)
        if isinstance(latest_price, Exception)
        else AsyncMock(return_value=latest_price)
    )
    cycle.runner = _StaticRunner()
    cycle.risk = _PassRisk()
    cycle.execution_planner = _RecordingPlanner()
    cycle.executor.execute = AsyncMock(wraps=cycle.executor.execute)
    return cycle


def test_bootstrap_registers_builtins_and_custom_factories():
    from cryptotrader.bootstrap import build_trading_cycle

    config = AppConfig(
        signal_plugins=SignalPluginsConfig(
            factories=["tests.factories.custom_signal_component:create"],
        )
    )

    cycle = build_trading_cycle(config, mode="paper")

    assert set(cycle.registry.ids()) == {"kronos", "llm_committee", "fake"}
    assert cycle.runner.events is cycle.events
    assert cycle.executor.exchange is cycle.contexts.portfolio.exchange


def test_bootstrap_rejects_default_profile_component_missing_from_registry():
    from cryptotrader.bootstrap import build_trading_cycle

    config = AppConfig(
        signal_profile_defaults=SignalProfileDefaultsConfig(
            components=[ComponentWeight("missing_component", True, 1.0)],
        ),
    )

    with pytest.raises(ValueError, match="missing_component"):
        build_trading_cycle(config, mode="paper")


@pytest.mark.asyncio
async def test_shared_startup_rejects_persisted_profile_component_missing_from_registry():
    from types import SimpleNamespace

    from cryptotrader.bootstrap import initialize_trading_cycle
    from cryptotrader.signals.registry import SignalComponentRegistry
    from tests.factories.custom_signal_component import FakeSignalComponent
    from tests.factories.signal_fusion import profile

    class Profiles:
        async def get(self):
            return profile(ComponentWeight("missing_component", True, 1.0))

    cycle = SimpleNamespace(
        profiles=Profiles(),
        registry=SignalComponentRegistry((FakeSignalComponent(),)),
    )

    with pytest.raises(ValueError, match="missing_component"):
        await initialize_trading_cycle(cycle)


def test_paper_bootstrap_uses_market_collector_as_read_only_ticker_source():
    from cryptotrader.bootstrap import build_trading_cycle

    cycle = build_trading_cycle(AppConfig(), mode="paper")

    assert cycle.contexts.portfolio.ticker_source is cycle.contexts.market
    assert cycle.contexts.portfolio.ticker_source is not cycle.executor.exchange


def test_bootstrap_backtest_is_owned_by_backtest_engine():
    from cryptotrader.bootstrap import build_trading_cycle

    with pytest.raises(ValueError, match="BacktestEngine"):
        build_trading_cycle(AppConfig(), mode="backtest")


async def test_paper_executor_and_context_reader_share_position_state():
    from cryptotrader.bootstrap import build_trading_cycle
    from cryptotrader.decision.models import ExecutionPlan, OrderIntent
    from tests.factories.signal_fusion import context, request

    cycle = build_trading_cycle(AppConfig(), mode="paper")
    execution = ExecutionPlan(
        intents=(OrderIntent("BTC/USDT:USDT", "buy", 1.0, False),),
        stop_loss=90.0,
        take_profit=120.0,
    )

    result = await cycle.executor.execute(execution, context())
    portfolio = await cycle.contexts.portfolio.read(request(), 100.0)

    assert result.succeeded is True
    assert portfolio["positions"]["BTC/USDT:USDT"]["amount"] == 1.0
    assert len(await cycle.executor.exchange.list_pending_algos("BTC/USDT:USDT")) == 1


async def test_paper_hitl_refreshes_real_ticker_and_preserves_frozen_analysis():
    from tests.factories.signal_fusion import request

    cycle = await _build_paper_hitl_cycle(105.0)
    pending = await cycle.run(request())
    frozen = await cycle.approvals.get(pending.approval_id)
    assert frozen is not None

    approved = await cycle.resume_approved(pending.approval_id)

    assert approved.status == "completed"
    assert approved.trade_plan == frozen.plan
    assert approved.component_signals == frozen.plan.component_signals
    assert approved.fused_signal == frozen.plan.fused_signal
    assert cycle.execution_planner.context.current_price == 105.0
    assert cycle.execution_planner.context.snapshots == frozen.signal_context.snapshots
    assert cycle.execution_planner.context.atr == frozen.signal_context.atr
    assert cycle.runner.calls == 1
    cycle.contexts.market.latest_price.assert_awaited_once_with("BTC/USDT:USDT", "okx")


@pytest.mark.parametrize(
    "ticker_result",
    [0.0, RuntimeError("market offline")],
    ids=["invalid", "unavailable"],
)
async def test_paper_hitl_unavailable_ticker_fails_closed_with_terminal_audit(ticker_result):
    from tests.factories.signal_fusion import request

    cycle = await _build_paper_hitl_cycle(ticker_result)
    pending = await cycle.run(request())
    frozen = await cycle.approvals.get(pending.approval_id)
    assert frozen is not None

    outcome = await cycle.resume_approved(pending.approval_id)

    assert outcome.status == "risk_rejected"
    assert outcome.trade_plan == frozen.plan
    assert outcome.risk_result.rejected_by == "execution_state_refresh"
    assert "ticker" in outcome.error.lower()
    assert cycle.executor.execute.await_count == 0
    record = await cycle.journal.get(outcome.cycle_id)
    assert record.status == "risk_rejected"
    assert record.hitl_result["refresh_status"] == "failed"
    assert record.component_signals
    assert record.fused_signal is not None
    from cryptotrader.cycle_serialization import signal_profile_payload, trade_plan_payload

    assert record.trade_plan == trade_plan_payload(frozen.plan)
    assert record.profile_snapshot == signal_profile_payload(frozen.profile)


async def test_injected_runtime_state_is_shared_across_cycles_without_database():
    from cryptotrader.bootstrap import SeededProfileRepository, build_trading_cycle
    from cryptotrader.hitl.store import ApprovalStore
    from cryptotrader.journal.store import CycleJournalStore
    from tests.factories.custom_signal_component import FakeSignalComponent
    from tests.factories.signal_fusion import profile

    config = AppConfig()
    profiles = SeededProfileRepository(None, profile())
    approvals = ApprovalStore()
    journal = CycleJournalStore()
    custom = FakeSignalComponent()
    first = build_trading_cycle(
        config,
        "paper",
        profile_repository=profiles,
        approval_store=approvals,
        journal_store=journal,
        custom_components=(custom,),
    )
    second = build_trading_cycle(
        config,
        "paper",
        profile_repository=profiles,
        approval_store=approvals,
        journal_store=journal,
        custom_components=(custom,),
    )

    updated = await first.profiles.replace(profile(kronos=0.75, llm=0.25))

    assert second.profiles is first.profiles
    assert second.approvals is first.approvals
    assert second.journal is first.journal is journal
    assert second.registry.get("fake") is first.registry.get("fake") is custom
    assert (await second.profiles.get()).revision == updated.revision
    assert (await second.profiles.get()).components[0].weight == 0.75

"""生产依赖只能装配为一个 TradingCycle。"""

import pytest

from cryptotrader.config import AppConfig, SignalPluginsConfig


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


async def test_injected_runtime_state_is_shared_across_cycles_without_database():
    from cryptotrader.bootstrap import SeededProfileRepository, build_trading_cycle
    from cryptotrader.hitl.store import ApprovalStore
    from tests.factories.signal_fusion import profile

    config = AppConfig()
    profiles = SeededProfileRepository(None, profile())
    approvals = ApprovalStore()
    first = build_trading_cycle(
        config,
        "paper",
        profile_repository=profiles,
        approval_store=approvals,
    )
    second = build_trading_cycle(
        config,
        "paper",
        profile_repository=profiles,
        approval_store=approvals,
    )

    updated = await first.profiles.replace(profile(kronos=0.75, llm=0.25))

    assert second.profiles is first.profiles
    assert second.approvals is first.approvals
    assert (await second.profiles.get()).revision == updated.revision
    assert (await second.profiles.get()).components[0].weight == 0.75

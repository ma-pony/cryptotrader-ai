from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from cryptotrader.backtest.engine import BacktestEngine, BacktestExecutor
from cryptotrader.backtest.result import BacktestResult
from cryptotrader.decision.models import ExecutionPlan, OrderIntent
from tests.factories.signal_fusion import context


def _candles(count: int = 8) -> list[list]:
    start = 1_704_067_200_000
    return [
        [start + index * 3_600_000, 100.0 + index, 103.0 + index, 99.0 + index, 102.0 + index, 10.0]
        for index in range(count)
    ]


def test_backtest_result_summary_uses_cycle_records():
    result = BacktestResult(total_return=0.15, sharpe_ratio=1.2, max_drawdown=-0.05, win_rate=0.6)

    assert result.summary()["total_return"] == "15.00%"
    assert result.cycle_records == []
    assert result.profile_revisions == []


@pytest.mark.asyncio
async def test_executor_schedules_target_delta_for_next_bar_open():
    executor = BacktestExecutor(initial_capital=10_000.0, slippage_bps=10.0, fee_bps=5.0)
    execution_plan = ExecutionPlan(
        intents=(OrderIntent("BTC/USDT:USDT", "buy", 10.0, False),),
        stop_loss=95.0,
        take_profit=115.0,
    )

    scheduled = await executor.execute(execution_plan, context(price=100.0))

    assert scheduled.succeeded is True
    assert executor.position.side == "flat"

    executor.execute_pending_at([1, 101.0, 105.0, 100.0, 104.0, 10.0])

    assert executor.position.side == "long"
    assert executor.position.amount == pytest.approx(10.0)
    assert executor.position.avg_price == pytest.approx(101.101)
    assert executor.protection == (95.0, 115.0)


@pytest.mark.asyncio
async def test_executor_applies_protection_from_existing_trade_plan():
    executor = BacktestExecutor(initial_capital=10_000.0, slippage_bps=0.0, fee_bps=0.0)
    plan = ExecutionPlan(
        intents=(OrderIntent("BTC/USDT:USDT", "buy", 10.0, False),),
        stop_loss=95.0,
        take_profit=115.0,
    )
    await executor.execute(plan, context(price=100.0))
    executor.execute_pending_at([1, 100.0, 104.0, 99.0, 103.0, 10.0])

    executor.process_protection([2, 103.0, 104.0, 94.0, 96.0, 10.0])

    assert executor.position.side == "flat"
    assert executor.trades[-1]["reason"] == "stop_loss"
    assert executor.trades[-1]["pnl"] == pytest.approx(-50.0)


@pytest.mark.asyncio
async def test_engine_uses_one_frozen_profile_revision_for_every_cycle(monkeypatch):
    from cryptotrader.decision.models import CycleOutcome
    from tests.factories.signal_fusion import profile

    class MutableProfiles:
        def __init__(self):
            self.value = profile(kronos=1.0, llm=0.0, revision=1)

        async def get(self):
            return self.value

        async def replace(self, value):
            self.value = value

    release_second_bar = asyncio.Event()

    class FakeCycle:
        def __init__(self, frozen_profiles):
            self.profiles = frozen_profiles
            self.calls = 0

        async def run(self, _request):
            self.calls += 1
            selected = await self.profiles.get()
            if self.calls == 2:
                await release_second_bar.wait()
            return CycleOutcome(f"cycle-{self.calls}", "no_change", selected.revision)

    profiles = MutableProfiles()
    engine = BacktestEngine(
        "BTC/USDT:USDT",
        "2024-01-01",
        "2024-01-02",
        interval="1h",
        lookback=2,
        profile_repository=profiles,
        cycle_factory=lambda frozen, _contexts, _executor, _journal: FakeCycle(frozen),
    )

    async def fake_fetch(_requirements):
        engine._candles_by_timeframe = {"1h": _candles()}
        engine._candles = engine._candles_by_timeframe["1h"]

    monkeypatch.setattr(engine, "_fetch_historical_data", AsyncMock(side_effect=fake_fetch))
    task = asyncio.create_task(engine.run())
    await engine.first_bar_processed.wait()
    await profiles.replace(profile(kronos=1.0, llm=0.0, revision=2))
    release_second_bar.set()
    result = await task

    assert set(result.profile_revisions) == {1}
    assert len(result.cycle_ids) > 1


def test_result_computes_metrics_from_executor_state():
    engine = BacktestEngine(
        "BTC/USDT:USDT",
        "2024-01-01",
        "2024-01-02",
        initial_capital=10_000.0,
    )
    result = engine._compute_result(
        equity=10_100.0,
        curve=[10_000.0, 10_050.0, 10_100.0],
        trades=[{"pnl": 100.0}],
    )

    assert result.total_return == pytest.approx(0.01)
    assert result.win_rate == 1.0
    assert result.max_drawdown == 0.0

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from cryptotrader.backtest.engine import BacktestEngine, BacktestExecutor
from cryptotrader.backtest.result import BacktestResult
from cryptotrader.decision.models import CycleOutcome, ExecutionPlan, OrderIntent
from cryptotrader.signals.models import CandleRequirement, DataRequirements
from tests.factories.signal_fusion import context, cycle_record


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


def test_snapshot_excludes_candle_at_open_and_includes_it_at_close():
    engine = BacktestEngine("BTC/USDT:USDT", "2024-01-01", "2024-01-02", interval="1h")
    candle = _candles(1)[0]
    engine._candles_by_timeframe = {"1h": [candle]}
    opened_at = datetime.fromtimestamp(candle[0] / 1000, UTC)

    with pytest.raises(ValueError, match="no 1h candles available"):
        engine._snapshot_at("1h", opened_at)

    snapshot = engine._snapshot_at("1h", opened_at + timedelta(hours=1))

    assert snapshot.market.ohlcv["close"].tolist() == [candle[4]]


def test_one_hour_decision_excludes_still_open_four_hour_candle():
    engine = BacktestEngine("BTC/USDT:USDT", "2024-01-01", "2024-01-02", interval="1h")
    opened_at = datetime(2024, 1, 1, tzinfo=UTC)
    candle = [int(opened_at.timestamp() * 1000), 100.0, 104.0, 99.0, 103.0, 10.0]
    engine._candles_by_timeframe = {"4h": [candle]}
    one_hour_decision = opened_at + timedelta(hours=1)

    with pytest.raises(ValueError, match="no 4h candles available"):
        engine._snapshot_at("4h", one_hour_decision)

    closed_snapshot = engine._snapshot_at("4h", opened_at + timedelta(hours=4))
    assert closed_snapshot.market.ohlcv["close"].tolist() == [103.0]


@pytest.mark.asyncio
async def test_cycle_decides_at_signal_close_and_fills_at_next_bar_open():
    bars = _candles(3)
    engine = BacktestEngine("BTC/USDT:USDT", "2024-01-01", "2024-01-02", interval="1h")
    engine._candles_by_timeframe = {"1h": bars}
    engine._candles = bars
    executor = BacktestExecutor(initial_capital=10_000.0, slippage_bps=0.0, fee_bps=0.0)
    observed: list[tuple[datetime, float]] = []
    from cryptotrader.journal.store import CycleJournalStore

    journal = CycleJournalStore()
    plan = ExecutionPlan(
        intents=(OrderIntent("BTC/USDT:USDT", "buy", 1.0, False),),
        stop_loss=1.0,
        take_profit=1_000.0,
    )

    class Cycle:
        async def run(self, request):
            snapshot = engine._snapshot_at("1h", request.as_of)
            observed.append((request.as_of, float(snapshot.market.ohlcv["close"].iloc[-1])))
            if len(observed) == 1:
                await executor.execute(plan, None)
            cycle_id = f"cycle-{len(observed)}"
            await journal.append(cycle_record(cycle_id=cycle_id))
            return CycleOutcome(cycle_id, "no_change", 1)

    class Contexts:
        def set_execution_state(self, **_state):
            return None

    await engine._run_bars(Cycle(), Contexts(), executor, journal)

    first_close = datetime.fromtimestamp((bars[0][0] + 3_600_000) / 1000, UTC)
    assert observed[0] == (first_close, bars[0][4])
    assert executor.trades[0]["price"] == bars[1][1]
    assert executor.trades[0]["ts"] == bars[1][0]


async def _run_journal_scoped_backtest(prefix: str, journal):
    bars = _candles(4)
    engine = BacktestEngine("BTC/USDT:USDT", "2024-01-01", "2024-01-02", interval="1h")
    engine._candles_by_timeframe = {"1h": bars}
    engine._candles = bars
    executor = BacktestExecutor(initial_capital=10_000.0, slippage_bps=0.0, fee_bps=0.0)

    class Cycle:
        def __init__(self):
            self.calls = 0

        async def run(self, _request):
            self.calls += 1
            cycle_id = f"{prefix}-{self.calls}"
            await journal.append(cycle_record(cycle_id=cycle_id))
            await asyncio.sleep(0)
            return CycleOutcome(cycle_id, "no_change", 1)

    class Contexts:
        def set_execution_state(self, **_state):
            return None

    return await engine._run_bars(Cycle(), Contexts(), executor, journal)


@pytest.mark.asyncio
@pytest.mark.parametrize("use_sqlite", [False, True])
async def test_backtest_result_resolves_only_ordered_outcome_records(tmp_path, use_sqlite):
    from cryptotrader.journal.store import CycleJournalStore

    database_url = f"sqlite+aiosqlite:///{tmp_path / 'shared-cycles.db'}" if use_sqlite else None
    journal = CycleJournalStore(database_url)
    await journal.append(cycle_record(cycle_id="prior-live"))

    result = await _run_journal_scoped_backtest("run", journal)

    assert [record.cycle_id for record in result.cycle_records] == result.cycle_ids
    assert result.cycle_ids == ["run-1", "run-2", "run-3"]


@pytest.mark.asyncio
async def test_concurrent_backtests_sharing_journal_do_not_cross_contaminate():
    from cryptotrader.journal.store import CycleJournalStore

    journal = CycleJournalStore()
    await journal.append(cycle_record(cycle_id="prior-live"))

    run_a, run_b = await asyncio.gather(
        _run_journal_scoped_backtest("run-a", journal),
        _run_journal_scoped_backtest("run-b", journal),
    )

    assert [record.cycle_id for record in run_a.cycle_records] == run_a.cycle_ids
    assert [record.cycle_id for record in run_b.cycle_records] == run_b.cycle_ids
    assert run_a.cycle_ids == ["run-a-1", "run-a-2", "run-a-3"]
    assert run_b.cycle_ids == ["run-b-1", "run-b-2", "run-b-3"]


def test_snapshot_uses_previous_completed_day_for_daily_inputs():
    engine = BacktestEngine("BTC/USDT:USDT", "2024-01-01", "2024-01-03", interval="1h")
    opened_at = datetime(2024, 1, 2, tzinfo=UTC)
    engine._candles_by_timeframe = {"1h": [[int(opened_at.timestamp() * 1000), 100.0, 102.0, 99.0, 101.0, 10.0]]}
    previous = "2024-01-01"
    current = "2024-01-02"
    engine._fng = {previous: 11, current: 99}
    engine._btc_dom = {previous: 51.0, current: 59.0}
    engine._fed_rate = {previous: 5.1, current: 5.9}
    engine._dxy = {previous: 101.0, current: 109.0}
    engine._etf_flows = {
        previous: {"totalNetInflow": 1.0, "totalNetAssets": 2.0, "cumNetInflow": 3.0},
        current: {"totalNetInflow": 91.0, "totalNetAssets": 92.0, "cumNetInflow": 93.0},
    }
    engine._vix = {previous: 12.0, current: 19.0}
    engine._sp500 = {previous: 4_700.0, current: 4_900.0}
    engine._stablecoin_supply = {previous: 100.0, current: 900.0}
    engine._btc_hashrate = {previous: 500.0, current: 900.0}
    engine._defi_tvl = {previous: 50.0, current: 90.0}

    snapshot = engine._snapshot_at("1h", opened_at + timedelta(hours=1))

    assert snapshot.macro.fear_greed_index == 11
    assert snapshot.macro.btc_dominance == 51.0
    assert snapshot.macro.fed_rate == 5.1
    assert snapshot.macro.dxy == 101.0
    assert snapshot.macro.etf_daily_net_inflow == 1.0
    assert snapshot.macro.etf_total_net_assets == 2.0
    assert snapshot.macro.etf_cum_net_inflow == 3.0
    assert snapshot.macro.vix == 12.0
    assert snapshot.macro.sp500 == 4_700.0
    assert snapshot.macro.stablecoin_total_supply == 100.0
    assert snapshot.macro.btc_hashrate == 500.0
    assert snapshot.onchain.defi_tvl == 50.0


@pytest.mark.asyncio
async def test_historical_daily_sources_include_previous_day():
    engine = BacktestEngine("BTC/USDT:USDT", "2024-01-01", "2024-01-03", interval="1h")
    loaded_starts: list[str] = []
    engine._load_extended_data = lambda *args: loaded_starts.extend(args)
    empty = AsyncMock(return_value={})

    with (
        patch("cryptotrader.backtest.engine.fetch_historical", new=AsyncMock(return_value=_candles(3))),
        patch("cryptotrader.backtest.historical_data.fetch_fear_greed", new=empty) as fear_greed,
        patch("cryptotrader.backtest.historical_data.fetch_funding_rate", new=AsyncMock(return_value={})),
        patch("cryptotrader.backtest.historical_data.fetch_btc_dominance", new=AsyncMock(return_value={})),
        patch("cryptotrader.backtest.historical_data.fetch_fred_series", new=AsyncMock(return_value={})),
        patch("cryptotrader.backtest.historical_data.fetch_futures_volume", new=AsyncMock(return_value={})),
    ):
        await engine._fetch_historical_data(
            DataRequirements(candles=(CandleRequirement("1h", 2),)),
        )

    fear_greed.assert_awaited_once_with("2023-12-31", "2024-01-03")
    assert loaded_starts == ["2023-12-31"]


def test_inexact_historical_kronos_aux_sources_remain_absent():
    start = datetime(2024, 1, 1, tzinfo=UTC)
    bars = []
    sp500 = {}
    for index in range(32):
        opened_at = start + timedelta(days=index)
        close = 100.0 + index * index
        bars.append([int(opened_at.timestamp() * 1000), close, close, close, close, 10.0])
        sp500[opened_at.strftime("%Y-%m-%d")] = close * 20.0
    as_of = start + timedelta(days=31)
    previous = (as_of - timedelta(days=1)).strftime("%Y-%m-%d")
    current = as_of.strftime("%Y-%m-%d")
    previous_sp500 = sp500[previous]
    sp500[current] = 1.0

    engine = BacktestEngine("BTC/USDT:USDT", "2024-01-01", "2024-02-02", interval="1d")
    engine._candles_by_timeframe = {"1d": bars}
    engine._sp500 = sp500
    engine._top_trader_ratio = {
        previous: {"topTraderRatio": 1.7},
        current: {"topTraderRatio": 9.9},
    }
    engine._ls_ratio = {previous: {"longShortRatio": 1.2}}

    snapshot = engine._snapshot_at("1d", as_of)

    assert not hasattr(snapshot.onchain, "lsr_top_count")
    assert not hasattr(snapshot.macro, "spy_btc_corr_30d")
    assert snapshot.onchain.liquidations_24h["long_short_ratio"] == pytest.approx(1.2)
    assert snapshot.macro.sp500 == previous_sp500
    assert not hasattr(snapshot.market, "premium_index_5d")


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
        def __init__(self, frozen_profiles, journal):
            self.profiles = frozen_profiles
            self.journal = journal
            self.calls = 0

        async def run(self, _request):
            self.calls += 1
            selected = await self.profiles.get()
            if self.calls == 2:
                await release_second_bar.wait()
            cycle_id = f"cycle-{self.calls}"
            await self.journal.append(cycle_record(cycle_id=cycle_id, profile_revision=selected.revision))
            return CycleOutcome(cycle_id, "no_change", selected.revision)

    profiles = MutableProfiles()
    engine = BacktestEngine(
        "BTC/USDT:USDT",
        "2024-01-01",
        "2024-01-02",
        interval="1h",
        lookback=2,
        profile_repository=profiles,
        cycle_factory=lambda frozen, _contexts, _executor, journal: FakeCycle(frozen, journal),
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

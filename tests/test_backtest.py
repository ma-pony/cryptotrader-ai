from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from cryptotrader.backtest.engine import BacktestEngine
from cryptotrader.backtest.result import BacktestResult
from cryptotrader.signals.models import CandleRequirement, DataRequirements


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
    assert result.config_revisions == []


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
async def test_backtest_replaces_configured_connections_with_one_hundred_percent_paper_book(monkeypatch):
    from cryptotrader.execution.models import ConnectionAllocation, ExecutionBook
    from cryptotrader.runtime import Runtime
    from cryptotrader.runtime_config.models import (
        MarketDataConfig,
        RuntimeConfigSnapshot,
        SignalComponentConfig,
        SignalConfig,
        SystemConfig,
    )
    from cryptotrader.signals.models import ComponentSignal
    from cryptotrader.signals.registry import SignalComponentRegistry
    from tests.factories.runtime_config import connection, runtime_document

    class Component:
        id = "fixture"
        display_name = "Fixture"
        description = "deterministic backtest signal"

        def requirements(self):
            return DataRequirements(candles=(CandleRequirement("1h", 20),))

        async def evaluate(self, context):
            return ComponentSignal(self.id, "long", 1.0, context.pair.canonical())

    live_connection = connection(
        "configured-live",
        "live",
        adapter_id="okx",
        credential_ref="must-not-be-read",
    )
    demo_connection = connection("configured-demo", "demo", adapter_id="okx")
    testnet_connection = connection("configured-testnet", "testnet", adapter_id="bybit")
    live_book = ExecutionBook(
        "configured-live",
        "Configured Live",
        "real",
        True,
        False,
        (ConnectionAllocation("configured-live", True, 1.0),),
    )
    document = runtime_document(
        connections=(live_connection, demo_connection, testnet_connection),
        books=(live_book,),
        system=SystemConfig(active=True),
        market_data=MarketDataConfig(source_id="default", parameters={"timeframe": "1h", "limit": 20}),
        signals=SignalConfig(
            components=(SignalComponentConfig(component_id="fixture", enabled=True, weight=1.0),),
            neutral_threshold=0.2,
            max_target_ratio=1.0,
            atr_stop_multiplier=2.0,
            reward_ratio=2.0,
        ),
    )
    snapshot = RuntimeConfigSnapshot(7, document, datetime.now(UTC))

    class Repository:
        async def get_or_create(self):
            return snapshot

        async def reveal_credentials(self, _credential_ref):
            raise AssertionError("backtest must not reveal configured venue credentials")

    from cryptotrader.venues.paper import PaperVenueAdapter
    from cryptotrader.venues.registry import VenueAdapterRegistry

    class RecordingPaperAdapter(PaperVenueAdapter):
        def __init__(self):
            super().__init__()
            self.connect_calls: list[tuple[str, str]] = []

        async def connect(self, configured_connection, credentials):
            assert credentials is None
            self.connect_calls.append((configured_connection.adapter_id, configured_connection.id))
            return await super().connect(configured_connection, credentials)

    paper = RecordingPaperAdapter()
    engine = BacktestEngine(
        "BTC/USDT:USDT",
        "2024-01-01",
        "2024-01-02",
        interval="1h",
        lookback=20,
        repository=Repository(),
        signal_registry=SignalComponentRegistry((Component(),)),
        venue_registry=VenueAdapterRegistry((paper,)),
    )

    async def fake_fetch(_requirements):
        engine._candles_by_timeframe = {"1h": _candles(24)}
        engine._candles = engine._candles_by_timeframe["1h"]

    monkeypatch.setattr(engine, "_fetch_historical_data", fake_fetch)

    @asynccontextmanager
    async def reject_production_execution_lease(_runtime, _pair):
        raise AssertionError("backtest must never acquire a production execution lease")
        yield

    monkeypatch.setattr(Runtime, "execution_lease", reject_production_execution_lease)

    result = await engine.run()

    assert result.config_revisions == [7] * len(result.cycle_records)
    assert {book.book_id for record in result.cycle_records for book in record.book_results} == {"backtest"}
    assert paper.connect_calls == [("paper", "backtest-paper")]


@pytest.mark.asyncio
async def test_explicit_backtest_snapshot_never_reads_ambient_bootstrap_or_repository(monkeypatch):
    from cryptotrader.bootstrap import BootstrapSettings
    from cryptotrader.runtime_config.models import RuntimeConfigSnapshot
    from tests.factories.runtime_config import runtime_document

    snapshot = RuntimeConfigSnapshot(44, runtime_document(), datetime.now(UTC))
    engine = BacktestEngine(
        "BTC/USDT:USDT",
        "2024-01-01",
        "2024-01-02",
        snapshot=snapshot,
        signal_registry=type("EmptyRegistry", (), {"enabled": lambda _self, _profile: ()})(),
    )

    def fail_ambient_bootstrap():
        raise AssertionError("explicit snapshot must bypass ambient bootstrap")

    async def no_candles(_requirements):
        engine._candles = []

    monkeypatch.setattr(BootstrapSettings, "from_environment", fail_ambient_bootstrap)
    monkeypatch.setattr(engine, "_fetch_historical_data", no_candles)

    assert await engine.run() == BacktestResult()


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


@pytest.mark.asyncio
async def test_fred_without_an_explicit_key_uses_only_cached_observations(monkeypatch, tmp_path):
    from cryptotrader.backtest import historical_data

    monkeypatch.setattr(historical_data, "CACHE_DB", tmp_path / "historical.sqlite")
    monkeypatch.setenv("FRED_API_KEY", "must-not-be-read")

    class NoNetworkClient:
        async def __aenter__(self):
            raise AssertionError("missing explicit FRED key must not start a remote request")

        async def __aexit__(self, *args):
            return None

    monkeypatch.setattr(historical_data.httpx, "AsyncClient", NoNetworkClient)

    assert await historical_data.fetch_fred_series("DFF", "2024-01-01", "2024-01-03") == {}


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


def test_result_computes_metrics_from_paper_cycle_equity():
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

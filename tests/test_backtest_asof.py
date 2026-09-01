"""Historical clocks reach account facts and source coverage."""

from datetime import timedelta

import pytest

from tests.test_backtest_accounting import START, replay


@pytest.mark.asyncio
async def test_explicit_historical_source_preserves_actual_ohlcv():
    from cryptotrader.backtest.engine import BacktestEngine
    from cryptotrader.signals.context import HistoricalSignalContextProvider

    engine = BacktestEngine("ETH/USDT:USDT", "2024-01-01", "2024-01-02", interval="1h")
    engine._candles_by_timeframe = {"1h": [[int(START.timestamp() * 1000), 101, 115, 92, 109, 7]]}
    source = HistoricalSignalContextProvider(engine._snapshot_at, default_timeframe="1h")
    bars = await source.read_candles(engine.pair, "1h", START, START + timedelta(hours=1), START + timedelta(hours=1))
    assert len(bars) == 1
    assert getattr(bars[0], "open", None) == 101
    assert (bars[0].high, bars[0].low, bars[0].close, bars[0].volume) == (115, 92, 109, 7)


@pytest.mark.asyncio
async def test_fill_account_and_coverage_use_historical_close_not_wall_clock():
    _, session = await replay([(100, "buy"), (110, "sell")])
    account = await session.fetch_account()
    assert account.observed_at == START + timedelta(hours=2)
    page = await session.fetch_fills(None)
    assert page.coverage_start == START
    assert page.coverage_end == START + timedelta(hours=2)
    assert [fill.occurred_at for fill in page.items] == [START + timedelta(hours=1), START + timedelta(hours=2)]
    assert "2024-01-01T02:00:00+00:00" in account.valuation_notes[0]


@pytest.mark.asyncio
async def test_replay_uses_selected_source_and_preserves_frozen_source_identity(monkeypatch):
    from unittest.mock import AsyncMock

    from cryptotrader.backtest.engine import BacktestEngine
    from tests.factories.backtest import registries, replay_config

    markets, signals, source, component = registries()
    engine = BacktestEngine(
        "ETH/USDT:USDT",
        "2024-01-01",
        "2024-01-01T02:00:00",
        interval="1h",
        lookback=20,
        snapshot=replay_config(),
        signal_registry=signals,
        market_registry=markets,
    )
    monkeypatch.setattr("cryptotrader.backtest.historical_data.fetch_fear_greed", AsyncMock(return_value={}))
    result = await engine.run()
    assert len(result.cycle_records) == 2
    assert source.requests[0][0] == engine.pair
    assert len(component.contexts) == 2
    assert [context.as_of for context in component.contexts] == [START + timedelta(hours=1), START + timedelta(hours=2)]
    for record in result.cycle_records:
        assert record.run.config_snapshot["market_data"]["source_id"] == "default"
        assert record.run.config_snapshot["market_data"]["parameters"]["market_adapter_id"] == "bybit"
    assert result.data_coverage["market_source_id"] == "default"
    assert result.data_coverage["market_type"] == "swap"


@pytest.mark.asyncio
async def test_repeated_runs_reset_paper_balance_and_risk_peak_even_with_reused_adapter(monkeypatch):
    from unittest.mock import AsyncMock

    from cryptotrader.backtest.engine import BacktestEngine
    from cryptotrader.venues.paper import PaperVenueAdapter
    from cryptotrader.venues.registry import VenueAdapterRegistry
    from tests.factories.backtest import registries, replay_config

    markets, signals, _, _ = registries()
    paper = PaperVenueAdapter()
    engine = BacktestEngine(
        "BTC/USDT:USDT",
        "2024-01-01",
        "2024-01-01T03:00:00",
        interval="1h",
        lookback=20,
        snapshot=replay_config(),
        signal_registry=signals,
        market_registry=markets,
        venue_registry=VenueAdapterRegistry((paper,)),
    )
    monkeypatch.setattr("cryptotrader.backtest.historical_data.fetch_fear_greed", AsyncMock(return_value={}))
    first = await engine.run()
    second = await engine.run()
    assert first.fill_count > 0
    assert first.equity_curve == second.equity_curve
    assert first.fees == second.fees
    assert first.fill_count == second.fill_count
    assert set(first.decision_ids).isdisjoint(second.decision_ids)
    assert len(set(first.decision_ids)) == len(first.cycle_records) == 3


@pytest.mark.asyncio
async def test_public_decimal_quote_reaches_actual_paper_fill_without_float_roundtrip(monkeypatch):
    from decimal import Decimal
    from unittest.mock import AsyncMock

    from cryptotrader.backtest.engine import BacktestEngine
    from tests.factories.backtest import registries, replay_config

    markets, signals, source, _ = registries()
    read = source.read_candles
    price = Decimal("100.123456789123456789")

    async def precise(*args):
        return tuple(
            bar.model_copy(update={"open": price, "high": price + 3, "low": price - 3, "close": price})
            for bar in await read(*args)
        )

    source.read_candles = precise
    monkeypatch.setattr("cryptotrader.backtest.historical_data.fetch_fear_greed", AsyncMock(return_value={}))
    engine = BacktestEngine(
        "BTC/USDT:USDT",
        "2024-01-01",
        "2024-01-01T01:00:00",
        interval="1h",
        lookback=20,
        snapshot=replay_config(),
        signal_registry=signals,
        market_registry=markets,
    )
    result = await engine.run()
    assert result.fills[0].price == price


@pytest.mark.asyncio
async def test_saved_forecast_keeps_original_open_points_and_closed_reference_in_result(monkeypatch, tmp_path):
    import json
    from dataclasses import replace
    from decimal import Decimal
    from unittest.mock import AsyncMock

    from api.routes.backtest import _result_to_dict
    from cryptotrader.backtest.engine import BacktestEngine
    from cryptotrader.signals.presentation import Series, SeriesBlock, SeriesPoint
    from tests.factories.backtest import registries, replay_config

    markets, signals, _, component = registries()
    evaluate = component.evaluate

    async def forecast(context):
        signal = await evaluate(context)
        block = SeriesBlock(
            title="预测",
            forecast_start=context.as_of,
            evaluation_target="candle_close",
            series=(
                Series(name="close", unit="USDT", points=(SeriesPoint(time=context.as_of, value=Decimal("108")),)),
            ),
        )
        return replace(signal, blocks=(block,))

    component.evaluate = forecast
    monkeypatch.setattr("cryptotrader.backtest.historical_data.fetch_fear_greed", AsyncMock(return_value={}))
    result = await BacktestEngine(
        "BTC/USDT:USDT",
        "2024-01-01",
        "2024-01-01T01:00:00",
        interval="1h",
        lookback=20,
        snapshot=replay_config(),
        signal_registry=signals,
        market_registry=markets,
    ).run()
    signal = result.cycle_records[0].component_signals[0]
    assert signal.blocks[0].series[0].points[0].time == START + timedelta(hours=1)
    assert signal.evaluation_reference.reference_time == START + timedelta(hours=1)
    assert signal.evaluation_reference.due_at == START + timedelta(hours=2)
    assert signal.evaluation_reference.market_source_id == "default"
    data = _result_to_dict(result)
    saved = data["decisions"][0]["components"][0]
    assert isinstance(saved["blocks"][0], dict)
    assert saved["blocks"][0]["evaluation_target"] == "candle_close"
    path = tmp_path / "result.json"
    result.to_json(str(path))
    exported = json.loads(path.read_text())
    assert exported["decisions"][0]["components"][0]["blocks"][0]["series"][0]["points"][0]["value"] == "108"


@pytest.mark.asyncio
async def test_unavailable_source_reports_missing_bars_and_costs_without_a_live_fallback(monkeypatch):
    from unittest.mock import AsyncMock

    from cryptotrader.backtest.engine import BacktestEngine
    from tests.factories.backtest import registries, replay_config

    markets, signals, source, component = registries()
    source.read_candles = AsyncMock(return_value=())
    monkeypatch.setattr("cryptotrader.backtest.historical_data.fetch_fear_greed", AsyncMock(return_value={}))
    result = await BacktestEngine(
        "BTC/USDT:USDT",
        "2024-01-01",
        "2024-01-01T02:00:00",
        interval="1h",
        lookback=20,
        snapshot=replay_config(),
        signal_registry=signals,
        market_registry=markets,
    ).run()
    assert result.fill_count == result.closed_trade_count == 0
    assert result.win_rate is None
    assert result.equity_curve[0].time == START
    assert component.contexts == []
    assert result.data_coverage["candles"]["1h"]["missing"] == 22
    assert result.data_coverage["funding"]["status"] == "unavailable"
    assert any("funding" in reason for reason in result.unmodeled_costs)

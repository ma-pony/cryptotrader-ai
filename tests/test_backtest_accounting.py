"""Replay accounting uses actual Paper fills, never inferred decision counts."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace

import pytest

from cryptotrader.backtest.engine import BacktestEngine
from cryptotrader.venues.models import BacktestCostModel, OrderIntent, ProtectionSpec
from cryptotrader.venues.paper import PaperVenueAdapter
from tests.factories.runtime_config import connection

START = datetime(2024, 1, 1, tzinfo=UTC)


async def replay(actions, *, funding=(), protect=False):
    engine = BacktestEngine(
        "BTC/USDT:USDT",
        "2024-01-01",
        (START + timedelta(hours=len(actions))).isoformat(),
        interval="1h",
        initial_capital=1000,
    )
    engine._candles = [
        [int((START + timedelta(hours=i)).timestamp() * 1000), price, price + 1, price - 1, price, 10]
        for i, (price, _) in enumerate(actions)
    ]
    adapter = PaperVenueAdapter(
        clock=engine._clock, cost_model=BacktestCostModel(Decimal("0.001"), Decimal("0"), False)
    )
    engine._as_of = START
    session = await adapter.connect(connection(parameters={"initial_equity": "1000"}), None)
    engine.cost_model = adapter.cost_model
    if funding:
        engine.cost_model = adapter.cost_model = session.cost_model = BacktestCostModel(
            Decimal("0.001"), Decimal("0"), True
        )
    engine.funding_settlements = funding

    class Cycle:
        journal = SimpleNamespace(get=None)

        async def run(self, request):
            _, side = actions.pop(0)
            if side:
                await session.place_order(OrderIntent(request.pair, side, Decimal("1"), "market", None, False))
                if protect:
                    await session.replace_protection(
                        ProtectionSpec(request.pair, "long", Decimal("1"), Decimal("99.5"), Decimal("100.5"))
                    )
            return SimpleNamespace(cycle_id="fixture", status="completed", config_revision=1)

    async def record(_):
        from cryptotrader.journal.models import DecisionRun, MultiVenueCycleRecord

        finished_at = engine._clock()
        return MultiVenueCycleRecord(
            cycle_id="fixture",
            config_revision=1,
            market_data_source_id="fixture-market",
            component_signals=(),
            fused_signal=None,
            target_position=None,
            book_results=(),
            cycle_status="completed",
            execution_status="not_started",
            requires_attention=False,
            created_at=finished_at,
            run=DecisionRun("BTC/USDT:USDT", "analysis", "backtest", {}, finished_at, None),
        )

    cycle = Cycle()
    cycle.journal.get = record
    result = await engine._run_bars(cycle, session)
    return result, session


@pytest.mark.asyncio
async def test_round_trip_charges_actual_fills_and_returns_timestamped_equity():
    result, session = await replay([(100, "buy"), (110, "sell")])
    assert (await session.fetch_account()).equity.amount == Decimal("1009.79")
    assert result.fill_count == 2
    assert result.closed_trade_count == 1
    assert result.fees == Decimal("0.21")
    assert result.closed_trades[0].gross_pnl == Decimal("10")
    assert result.equity_curve[-1].equity == Decimal("1009.79")
    assert result.equity_curve[0].time == START
    assert result.win_rate == 1.0


@pytest.mark.asyncio
@pytest.mark.parametrize(("actions", "count"), [([(100, None), (110, None)], 0), ([(100, "buy"), (110, None)], 1)])
async def test_no_completed_round_trip_has_unknown_win_rate(actions, count):
    result, _ = await replay(actions)
    assert result.win_rate is None
    assert result.fill_count == count
    assert result.closed_trade_count == 0


@pytest.mark.asyncio
async def test_funding_charged_only_at_provided_settlement_time():
    from cryptotrader.backtest.historical_data import HistoricalFundingSettlement

    event = HistoricalFundingSettlement(
        id="funding-1",
        pair="BTC/USDT:USDT",
        source_id="fixture-public",
        occurred_at=START + timedelta(hours=1, minutes=30),
        rate=Decimal("0.001"),
        mark_price=Decimal("100"),
    )
    result, session = await replay([(100, "buy"), (110, "sell")], funding=(event,))
    assert (await session.fetch_account()).equity.amount == Decimal("1009.69")
    assert result.funding == Decimal("-0.1")
    assert result.funding_entries[0].occurred_at == START + timedelta(hours=1, minutes=30)
    assert result.closed_trades[0].funding == Decimal("-0.1")
    assert result.closed_trades[0].net_pnl == Decimal("9.69")


def test_api_preserves_unknown_win_rate_without_invented_trade_count():
    from api.routes.backtest import _result_to_dict
    from cryptotrader.backtest.result import BacktestResult

    payload = _result_to_dict(BacktestResult())
    assert payload["metrics"]["win_rate"] is None
    assert payload["metrics"]["fill_count"] == 0
    assert payload["metrics"]["closed_trade_count"] == 0


@pytest.mark.asyncio
async def test_new_protection_begins_next_bar_and_its_fill_is_a_closed_round():
    result, _ = await replay([(100, "buy"), (100, None)], protect=True)
    assert result.equity_curve[1].equity == Decimal("999.9")
    assert result.equity_curve[-1].equity == Decimal("999.3005")
    assert result.fills[-1].occurred_at == START + timedelta(hours=2)
    assert result.fills[-1].price == Decimal("99.5")
    assert result.closed_trade_count == 1
    assert result.win_rate == 0


@pytest.mark.asyncio
async def test_flat_price_round_counts_as_loss_after_fee_and_pages_do_not_drop_later_fills():
    result, _ = await replay([(100, "buy"), *[(100, None)] * 190, (100, "sell")])
    assert result.fill_count == 2
    assert result.closed_trade_count == 1
    assert result.closed_trades[0].gross_pnl == 0
    assert result.closed_trades[0].net_pnl == Decimal("-0.2")
    assert result.win_rate == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("pair", ["BTC/USDT:USDT", "BTC/USDT"])
async def test_directional_slippage_and_fee_use_execution_price_for_both_markets(pair):
    from cryptotrader.pair import Pair

    pair = Pair.parse(pair)
    session = await PaperVenueAdapter(cost_model=BacktestCostModel(Decimal("0.001"), Decimal("100"), False)).connect(
        connection(parameters={"initial_equity": "1000"}), None
    )
    await session.set_quote(pair, Decimal("100"))
    await session.place_order(OrderIntent(pair, "buy", Decimal("1"), "market", None, False))
    await session.set_quote(pair, Decimal("110"))
    await session.place_order(OrderIntent(pair, "sell", Decimal("1"), "market", None, False))
    fills = (await session.fetch_fills(None)).items
    assert [fill.price for fill in fills] == [Decimal("101"), Decimal("108.9")]
    assert [fill.fee.amount for fill in fills] == [Decimal("0.101"), Decimal("0.1089")]
    assert (await session.fetch_account()).equity.amount == Decimal("1007.6901")


@pytest.mark.asyncio
async def test_reversal_and_partial_close_allocate_fee_to_completed_round_only():
    from cryptotrader.backtest.result import closed_round_trips
    from cryptotrader.pair import Pair

    pair = Pair.parse("BTC/USDT:USDT")
    session = await PaperVenueAdapter(cost_model=BacktestCostModel()).connect(
        connection(parameters={"initial_equity": "1000"}), None
    )
    for price, side, amount in [(100, "buy", "2"), (110, "sell", "1"), (120, "sell", "3"), (110, "buy", "2")]:
        await session.set_quote(pair, Decimal(price))
        await session.place_order(OrderIntent(pair, side, Decimal(amount), "market", None, False))
    closed = closed_round_trips(list((await session.fetch_fills(None)).items))
    assert [(trade.gross_pnl, trade.fees, trade.net_pnl) for trade in closed] == [
        (Decimal("30"), Decimal("0.43"), Decimal("29.57")),
        (Decimal("20"), Decimal("0.46"), Decimal("19.54")),
    ]
    assert (await session.fetch_account()).equity.amount == Decimal("1049.11")


@pytest.mark.asyncio
async def test_api_maps_real_historical_curve_and_exact_costs():
    from api.routes.backtest import _result_to_dict

    result, _ = await replay([(100, "buy"), (110, "sell")])
    data = _result_to_dict(result)
    assert data["equity_curve"] == [
        {"ts": "2024-01-01T00:00:00+00:00", "equity": 1000.0},
        {"ts": "2024-01-01T01:00:00+00:00", "equity": 999.9},
        {"ts": "2024-01-01T02:00:00+00:00", "equity": 1009.79},
    ]
    assert Decimal(data["fees"]) == Decimal("0.21")
    assert data["fills"][0]["instrument"]["pair"] == "BTC/USDT:USDT"

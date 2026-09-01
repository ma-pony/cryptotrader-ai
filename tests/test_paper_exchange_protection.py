"""Paper native-protection semantics on deterministic Decimal quotes."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace

import pytest

from cryptotrader.market_sources.protocol import HistoricalCandle
from cryptotrader.pair import Pair
from cryptotrader.venues.models import OrderIntent, ProtectionSpec
from tests.factories.runtime_config import connection

PAIR = Pair.parse("BTC/USDT:USDT")


async def _positioned_session(side: str = "long"):
    from cryptotrader.venues.paper import PaperVenueAdapter

    session = await PaperVenueAdapter().connect(
        connection(parameters={"initial_equity": "10000"}),
        None,
    )
    await session.set_quote(PAIR, Decimal("100"))
    await session.place_order(
        OrderIntent(PAIR, "buy" if side == "long" else "sell", Decimal("2"), "market", None, False)
    )
    return session


@pytest.mark.parametrize(
    ("side", "quote", "stop_loss", "take_profit", "expected_balance"),
    [
        ("long", "89", "90", "120", "9980"),
        ("long", "121", "90", "120", "10040"),
        ("short", "111", "110", "80", "9980"),
        ("short", "79", "110", "80", "10040"),
    ],
)
async def test_paper_protection_triggers_long_and_short_at_exact_decimal_trigger_price(
    side,
    quote,
    stop_loss,
    take_profit,
    expected_balance,
):
    session = await _positioned_session(side)
    protection = await session.replace_protection(
        ProtectionSpec(PAIR, side, Decimal("2"), Decimal(stop_loss), Decimal(take_profit))
    )
    await session.set_quote(PAIR, Decimal(quote))

    portfolio = await session.fetch_portfolio(PAIR)
    state = await session.list_open_state(PAIR)

    assert state.triggered_protections == (
        type(protection)(
            protection.protection_ids,
            PAIR,
            side,
            Decimal("2"),
            Decimal(stop_loss),
            Decimal(take_profit),
            False,
            True,
            protection.actual_order_ids,
        ),
    )
    assert state.position.signed_amount == Decimal("0")
    assert portfolio.position.signed_amount == Decimal("0")
    assert portfolio.balances["USDT"] == Decimal(expected_balance)


async def test_paper_protection_triggers_once_under_concurrent_reads():
    session = await _positioned_session("long")
    await session.replace_protection(ProtectionSpec(PAIR, "long", Decimal("2"), Decimal("90"), Decimal("120")))
    await session.set_quote(PAIR, Decimal("89"))

    states = await asyncio.gather(*(session.list_open_state(PAIR) for _ in range(8)))

    assert sum(bool(state.triggered_protections) for state in states) == 1
    assert all(state.position.signed_amount == Decimal("0") for state in states)
    assert (await session.list_open_state(PAIR)).protections == ()


async def test_paper_protection_replacement_retires_old_state_and_cancel_is_idempotent():
    session = await _positioned_session("long")
    old = await session.replace_protection(ProtectionSpec(PAIR, "long", Decimal("2"), Decimal("90"), Decimal("120")))
    replacement = await session.replace_protection(
        ProtectionSpec(PAIR, "long", Decimal("2"), Decimal("92"), Decimal("125"))
    )

    assert replacement.protection_ids != old.protection_ids
    assert (await session.list_open_state(PAIR)).protections == (replacement,)

    await session.cancel_protection(replacement.protection_ids)
    await session.cancel_protection(replacement.protection_ids)
    assert (await session.list_open_state(PAIR)).protections == ()


@pytest.mark.parametrize(
    "spec",
    [
        ProtectionSpec(PAIR, "short", Decimal("2"), Decimal("110"), Decimal("80")),
        ProtectionSpec(PAIR, "long", Decimal("3"), Decimal("90"), Decimal("120")),
        ProtectionSpec(PAIR, "long", Decimal("2"), Decimal("101"), Decimal("120")),
        ProtectionSpec(PAIR, "long", Decimal("2"), Decimal("90"), Decimal("99")),
    ],
)
async def test_paper_protection_rejects_wrong_side_amount_or_price_geometry(spec):
    from cryptotrader.venues.ccxt_base import VenueOperationError

    session = await _positioned_session("long")

    with pytest.raises(VenueOperationError, match="protection"):
        await session.replace_protection(spec)


@pytest.mark.parametrize(
    ("side", "opening", "high", "low", "expected"),
    [
        ("long", "100", "125", "85", "90"),
        ("long", "80", "125", "75", "80"),
        ("short", "100", "115", "75", "110"),
        ("short", "120", "125", "75", "120"),
    ],
)
async def test_historical_bar_stops_first_and_gap_uses_worse_open(side, opening, high, low, expected):
    session = await _positioned_session(side)
    await session.replace_protection(
        ProtectionSpec(
            PAIR,
            side,
            Decimal("2"),
            Decimal("90" if side == "long" else "110"),
            Decimal("120" if side == "long" else "80"),
        )
    )
    bar = HistoricalCandle(
        open_time=datetime(2024, 1, 1, tzinfo=UTC),
        open=Decimal(opening),
        high=Decimal(high),
        low=Decimal(low),
        close=Decimal("100"),
        volume=Decimal("10"),
    )
    await session.advance_bar(PAIR, bar)
    assert (await session.fetch_account()).positions == ()
    fills = await session.fetch_fills(None)
    assert fills.items[-1].price == Decimal(expected)
    assert len(fills.items) == 2


async def test_bar_protection_fill_retains_original_decision_binding_after_ledger_sync(tmp_path):
    from cryptotrader.accounts.store import AccountStore
    from cryptotrader.accounts.sync import AccountSyncService
    from cryptotrader.execution.service import VenueExecutionService
    from cryptotrader.migrations.workbench import migrate_workbench_schema
    from cryptotrader.venues.models import BacktestCostModel
    from cryptotrader.venues.paper import PaperVenueAdapter

    now = datetime(2024, 1, 1, tzinfo=UTC)
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'protection.sqlite'}"
    await migrate_workbench_schema(database_url)
    store = AccountStore(database_url, clock=lambda: now)
    adapter = PaperVenueAdapter(clock=lambda: now, cost_model=BacktestCostModel())
    adapter.account_store = store
    configured = connection(parameters={"initial_equity": "1000"})
    session = await adapter.connect(configured, None)
    await session.set_quote(PAIR, Decimal("100"))
    await session.place_order(OrderIntent(PAIR, "buy", Decimal("1"), "market", None, False))
    service = VenueExecutionService(session, connection=configured, account_store=store)
    plan = SimpleNamespace(connection_id=configured.id, book_id="original-book", decision_id="original-decision")
    protection = await service._replace_protection(
        plan,
        ProtectionSpec(PAIR, "long", Decimal("1"), Decimal("90"), Decimal("120")),
        [],
    )
    now += timedelta(hours=1)
    await session.advance_bar(
        PAIR,
        HistoricalCandle(
            open_time=now - timedelta(hours=1),
            open=Decimal("80"),
            high=Decimal("100"),
            low=Decimal("75"),
            close=Decimal("95"),
            volume=Decimal("1"),
        ),
    )
    await AccountSyncService(store, None).sync_session(session, configured.id)
    fills = await store.history(configured.id)
    exit_fill = fills[-1]
    attribution = await store.attribution(
        configured.id, exit_fill.venue_order_id, exit_fill.client_order_id, exit_fill.occurred_at
    )
    assert attribution["decision_id"] == "original-decision"
    assert attribution["book_id"] == "original-book"
    assert exit_fill.venue_order_id == protection.actual_order_ids[0]
    assert (await session.find_order(PAIR, order_id=exit_fill.venue_order_id)).status == "filled"
    assert exit_fill.price == Decimal("80")
    assert exit_fill.fee.amount == Decimal("0.08")
    assert (await store.latest(configured.id)).equity.amount == Decimal("979.82")

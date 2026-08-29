"""Paper native-protection semantics on deterministic Decimal quotes."""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

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

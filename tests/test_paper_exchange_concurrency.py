"""Connection-local Paper session concurrency guarantees."""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

from cryptotrader.pair import Pair
from cryptotrader.portfolio.models import ConnectionPortfolioSnapshot
from cryptotrader.venues.models import OpenVenueState, OrderIntent, ProtectionSpec
from tests.factories.runtime_config import connection

PAIR = Pair.parse("BTC/USDT:USDT")


async def _session(*, initial_equity: str = "1000", leverage: int = 1):
    from cryptotrader.venues.paper import PaperVenueAdapter

    session = await PaperVenueAdapter().connect(
        connection(parameters={"initial_equity": initial_equity}, leverage=leverage),
        None,
    )
    await session.set_quote(PAIR, Decimal("100"))
    return session


def _intent(side: str, *, reduce_only: bool = False) -> OrderIntent:
    return OrderIntent(PAIR, side, Decimal("1"), "market", None, reduce_only)


def _amount_intent(side: str, amount: str, *, reduce_only: bool = False) -> OrderIntent:
    return OrderIntent(PAIR, side, Decimal(amount), "market", None, reduce_only)


async def test_concurrent_entries_cannot_exceed_connection_margin():
    session = await _session()

    orders = await asyncio.gather(*(session.place_order(_intent("buy")) for _ in range(20)))
    portfolio = await session.fetch_portfolio(PAIR)

    assert sum(order.status == "filled" for order in orders) == 10
    assert sum(order.status == "rejected" for order in orders) == 10
    assert len({order.id for order in orders}) == 20
    assert portfolio.position.signed_amount == Decimal("10")
    assert portfolio.position.signed_notional == Decimal("1000")
    assert portfolio.equity == Decimal("1000")


async def test_concurrent_reduce_only_orders_close_once_without_flipping():
    session = await _session()
    await asyncio.gather(*(session.place_order(_intent("buy")) for _ in range(10)))

    reductions = await asyncio.gather(*(session.place_order(_intent("sell", reduce_only=True)) for _ in range(20)))
    state = await session.list_open_state(PAIR)

    assert sum(order.status == "filled" for order in reductions) == 10
    assert sum(order.status == "rejected" for order in reductions) == 10
    assert state.position.signed_amount == Decimal("0")
    assert state.position.signed_notional == Decimal("0")
    assert state.position.entry_price is None


async def test_different_pairs_use_independent_pair_locks_and_keep_state_separate():
    session = await _session(initial_equity="5000")
    eth = Pair.parse("ETH/USDT:USDT")
    await session.set_quote(eth, Decimal("25"))

    btc_order, eth_order = await asyncio.gather(
        session.place_order(_intent("buy")),
        session.place_order(OrderIntent(eth, "sell", Decimal("4"), "market", None, False)),
    )

    assert btc_order.status == eth_order.status == "filled"
    assert (await session.fetch_portfolio(PAIR)).position.signed_amount == Decimal("1")
    assert (await session.fetch_portfolio(eth)).position.signed_amount == Decimal("-4")


async def test_sign_changing_order_rechecks_margin_for_the_new_opposite_leg():
    session = await _session(initial_equity="1000", leverage=10)
    assert (await session.place_order(_amount_intent("buy", "100"))).status == "filled"
    await session.set_quote(PAIR, Decimal("95"))

    rejected = await session.place_order(_amount_intent("sell", "199"))

    assert rejected.status == "rejected"
    assert (await session.fetch_portfolio(PAIR)).position.signed_amount == Decimal("100")


async def test_affordable_small_flip_and_pure_reduce_only_order_remain_allowed():
    flip_session = await _session(initial_equity="1000", leverage=10)
    await flip_session.place_order(_amount_intent("buy", "100"))
    await flip_session.set_quote(PAIR, Decimal("95"))

    affordable_flip = await flip_session.place_order(_amount_intent("sell", "105"))

    assert affordable_flip.status == "filled"
    assert (await flip_session.fetch_portfolio(PAIR)).position.signed_amount == Decimal("-5")

    reduction_session = await _session(initial_equity="1000", leverage=10)
    await reduction_session.place_order(_amount_intent("buy", "100"))
    await reduction_session.set_quote(PAIR, Decimal("95"))

    reduction = await reduction_session.place_order(_amount_intent("sell", "50", reduce_only=True))

    assert reduction.status == "filled"
    assert (await reduction_session.fetch_portfolio(PAIR)).position.signed_amount == Decimal("50")


@pytest.mark.parametrize("bankrupt_price", ["90", "89"])
async def test_bankruptcy_atomically_liquidates_and_keeps_concurrent_reads_stable(bankrupt_price):
    session = await _session(initial_equity="1000", leverage=10)
    await session.place_order(_amount_intent("buy", "100"))
    await session.replace_protection(ProtectionSpec(PAIR, "long", Decimal("100"), Decimal("80"), Decimal("120")))
    await session.set_quote(PAIR, Decimal(bankrupt_price))

    reads = await asyncio.gather(
        *(session.fetch_portfolio(PAIR) for _ in range(8)),
        *(session.list_open_state(PAIR) for _ in range(8)),
    )

    portfolios = [result for result in reads if isinstance(result, ConnectionPortfolioSnapshot)]
    states = [result for result in reads if isinstance(result, OpenVenueState)]
    assert len(portfolios) == len(states) == 8
    assert all(snapshot.equity == Decimal("0") for snapshot in portfolios)
    assert all(snapshot.position.signed_amount == Decimal("0") for snapshot in portfolios)
    assert all(state.position.signed_amount == Decimal("0") for state in states)
    assert all(state.open_orders == () and state.protections == () for state in states)

    rejected = await session.place_order(_amount_intent("buy", "1"))
    assert rejected.status == "rejected"
    assert (await session.fetch_portfolio(PAIR)).equity == Decimal("0")


async def test_bankruptcy_is_committed_at_the_price_move_even_if_price_recovers_before_a_read():
    session = await _session(initial_equity="1000", leverage=10)
    await session.place_order(_amount_intent("buy", "100"))

    await session.set_quote(PAIR, Decimal("90"))
    await session.set_quote(PAIR, Decimal("100"))

    snapshot = await session.fetch_portfolio(PAIR)
    assert snapshot.equity == Decimal("0")
    assert snapshot.position.signed_amount == Decimal("0")

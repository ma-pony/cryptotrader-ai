"""Connection-local Paper session concurrency guarantees."""

from __future__ import annotations

import asyncio
from decimal import Decimal

from cryptotrader.pair import Pair
from cryptotrader.venues.models import OrderIntent
from tests.factories.runtime_config import connection

PAIR = Pair.parse("BTC/USDT:USDT")


async def _session(*, initial_equity: str = "1000"):
    from cryptotrader.venues.paper import PaperVenueAdapter

    session = await PaperVenueAdapter().connect(
        connection(parameters={"initial_equity": initial_equity}),
        None,
    )
    await session.set_quote(PAIR, Decimal("100"))
    return session


def _intent(side: str, *, reduce_only: bool = False) -> OrderIntent:
    return OrderIntent(PAIR, side, Decimal("1"), "market", None, reduce_only)


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

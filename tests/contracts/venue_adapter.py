"""Reusable structural assertions for venue adapter implementations."""

from __future__ import annotations

from decimal import Decimal
from inspect import iscoroutinefunction
from typing import TYPE_CHECKING

from cryptotrader.pair import Pair
from cryptotrader.portfolio.models import ConnectionPortfolioSnapshot
from cryptotrader.runtime_config.secrets import CredentialPayload
from cryptotrader.venues.models import ConnectionEnvironment, VenueCapabilities
from cryptotrader.venues.protocol import VenueAdapter
from tests.factories.runtime_config import connection

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


def assert_venue_contract(
    adapter_factory: Callable[[], VenueAdapter],
    environments: Iterable[ConnectionEnvironment],
) -> None:
    """Check the adapter surface shared by Paper and CCXT implementations."""
    adapter = adapter_factory()
    assert isinstance(adapter, VenueAdapter)
    assert adapter.adapter_id.strip()
    assert iscoroutinefunction(adapter.connect), "connect must be async"
    for environment in environments:
        capabilities = adapter.capabilities(environment)
        assert isinstance(capabilities, VenueCapabilities)


async def assert_ccxt_session_contract(adapter_factory, environment: ConnectionEnvironment, fake_factory) -> None:
    """Exercise platform-neutral CCXT behavior against a venue-shaped fake."""
    from cryptotrader.venues.models import OrderIntent, ProtectionSpec

    pair = Pair.parse("BTC/USDT:USDT")
    spot = Pair.parse("BTC/USDT")
    adapter = adapter_factory()
    session = await adapter.connect(
        connection(
            f"{adapter.adapter_id}-{environment}",
            environment,
            adapter_id=adapter.adapter_id,
            credential_ref="credentials",
        ),
        CredentialPayload(api_key="key", secret="secret", passphrase="passphrase"),  # pragma: allowlist secret
    )

    assert session.capabilities == adapter.capabilities(environment)
    assert await session.normalize_amount(spot, Decimal("0.123456")) == Decimal("0.123")
    assert await session.normalize_amount(pair, Decimal("0.123456")) == (
        Decimal("0.12") if adapter.adapter_id == "okx" else Decimal("0.123")
    )

    portfolio = await session.fetch_portfolio(pair)
    assert isinstance(portfolio, ConnectionPortfolioSnapshot)
    assert portfolio.connection_id == f"{adapter.adapter_id}-{environment}"
    expected_equity = Decimal("12345.67") if adapter.adapter_id == "okx" else Decimal("23456.78")
    assert portfolio.equity == expected_equity
    assert portfolio.balances["USDT"] == Decimal("10000.50")
    assert portfolio.position.signed_amount == Decimal("0.02")
    assert portfolio.position.signed_notional == Decimal("1000")
    assert isinstance(portfolio.position.signed_amount, Decimal)
    assert isinstance(portfolio.position.signed_notional, Decimal)

    spot_order = await session.place_order(OrderIntent(spot, "buy", Decimal("0.1"), "market", None, False))
    entry = await session.place_order(OrderIntent(pair, "buy", Decimal("0.1"), "market", None, False))
    reduction = await session.place_order(OrderIntent(pair, "sell", Decimal("0.04"), "market", None, True))
    close = await session.place_order(OrderIntent(pair, "sell", Decimal("0.02"), "market", None, True))

    assert spot_order.amount == Decimal("0.1")
    assert entry.amount == Decimal("0.1")
    assert reduction.filled_amount == Decimal("0.04")
    assert close.reduce_only is True

    protection = await session.replace_protection(
        ProtectionSpec(pair, "long", Decimal("0.02"), Decimal("48000"), Decimal("55000"))
    )
    assert protection.stop_loss == Decimal("48000")
    assert protection.take_profit == Decimal("55000")
    assert protection.active is True
    state = await session.list_open_state(pair)
    assert state.position.signed_amount == Decimal("0.02")
    assert state.protections == (protection,)
    assert state.open_orders[0].id == "open-1"
    assert state.open_orders[0].amount == Decimal("0.03")
    assert state.open_orders[0].filled_amount == Decimal("0.01")

    replacement = await session.replace_protection(
        ProtectionSpec(pair, "long", Decimal("0.02"), Decimal("47000"), Decimal("56000"))
    )
    assert replacement.stop_loss == Decimal("47000")
    assert replacement.take_profit == Decimal("56000")
    assert (await session.list_open_state(pair)).protections == (replacement,)

    await session.cancel_protection(replacement.protection_ids)
    assert (await session.list_open_state(pair)).protections == ()
    await session.close()
    await session.close()
    fake_client = fake_factory.clients[-1]
    assert fake_client.load_markets_calls == 1
    assert fake_client.close_calls == 1


async def assert_paper_session_contract(adapter_factory) -> None:
    """Exercise the normalized DTO surface without credentials or a client fake."""
    from cryptotrader.venues.models import OpenVenueState, OrderIntent

    pair = Pair.parse("BTC/USDT:USDT")
    session = await adapter_factory().connect(
        connection("paper-contract", "paper", parameters={"initial_equity": "20000"}),
        None,
    )
    await session.set_quote(pair, Decimal("50000"))

    assert session.capabilities == adapter_factory().capabilities("paper")
    assert await session.normalize_amount(pair, Decimal("0.123456")) == Decimal("0.123456")

    portfolio = await session.fetch_portfolio(pair)
    order = await session.place_order(OrderIntent(pair, "buy", Decimal("0.2"), "market", None, False))
    state = await session.list_open_state(pair)

    assert isinstance(portfolio, ConnectionPortfolioSnapshot)
    assert portfolio.equity == Decimal("20000")
    assert order.amount == order.filled_amount == Decimal("0.2")
    assert order.average_price == Decimal("50000")
    assert isinstance(state, OpenVenueState)
    assert state.position.signed_notional == Decimal("10000.0")
    await session.close()

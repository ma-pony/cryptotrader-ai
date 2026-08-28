"""Paper venue adapter contract and database-backed account parameters."""

from __future__ import annotations

from decimal import Decimal
from importlib import metadata

import pytest

from cryptotrader.pair import Pair
from cryptotrader.runtime_config.secrets import CredentialPayload
from cryptotrader.venues.models import OrderIntent
from tests.factories.runtime_config import connection

PAIR = Pair.parse("BTC/USDT:USDT")


class OptionPair(Pair):
    @property
    def market_type(self):
        return "option"


def paper_connection(connection_id: str = "paper-local", *, initial_equity="10000", **overrides):
    parameters = {} if initial_equity is None else {"initial_equity": initial_equity}
    return connection(connection_id, "paper", parameters=parameters, **overrides)


def order_intent(*, side: str = "buy", amount: str = "0.1", reduce_only: bool = False):
    return OrderIntent(PAIR, side, Decimal(amount), "market", None, reduce_only)


def test_paper_adapter_declares_platform_neutral_capabilities():
    from cryptotrader.venues.paper import PaperVenueAdapter
    from tests.contracts.venue_adapter import assert_venue_contract

    assert_venue_contract(PaperVenueAdapter, ("paper",))
    capabilities = PaperVenueAdapter().capabilities("paper")

    assert capabilities.market_types == frozenset({"spot", "swap"})
    assert capabilities.native_protection is True
    assert capabilities.reduce_only is True


@pytest.mark.parametrize("environment", ["demo", "testnet", "live"])
def test_paper_adapter_rejects_non_paper_environments(environment):
    from cryptotrader.venues.paper import PaperVenueAdapter

    with pytest.raises(ValueError, match="unsupported Paper environment"):
        PaperVenueAdapter().capabilities(environment)


async def test_paper_connection_rejects_credentials():
    from cryptotrader.venues.paper import PaperVenueAdapter

    credentials = CredentialPayload(api_key="paper-key", secret="paper-secret")  # pragma: allowlist secret
    with pytest.raises(ValueError, match="does not accept credentials"):
        await PaperVenueAdapter().connect(paper_connection(), credentials)


async def test_paper_connection_rejects_credential_reference():
    from cryptotrader.venues.paper import PaperVenueAdapter

    with pytest.raises(ValueError, match="does not accept credentials"):
        await PaperVenueAdapter().connect(paper_connection(credential_ref="paper-credentials"), None)


@pytest.mark.parametrize(
    "initial_equity",
    [None, 0, -1, "0", "-0.01", "NaN", "Infinity", "not-a-number", True],
)
async def test_paper_connection_requires_finite_positive_decimal_compatible_initial_equity(initial_equity):
    from cryptotrader.venues.paper import PaperVenueAdapter

    with pytest.raises(ValueError, match="initial_equity"):
        await PaperVenueAdapter().connect(paper_connection(initial_equity=initial_equity), None)


async def test_paper_connection_rejects_wrong_adapter_id():
    from cryptotrader.venues.paper import PaperVenueAdapter

    with pytest.raises(ValueError, match="adapter_id=paper"):
        await PaperVenueAdapter().connect(paper_connection(adapter_id="okx"), None)


async def test_paper_session_uses_same_portfolio_and_order_contract():
    from cryptotrader.portfolio.models import ConnectionPortfolioSnapshot
    from cryptotrader.venues.models import NormalizedOrder, OpenVenueState
    from cryptotrader.venues.paper import PaperVenueAdapter

    session = await PaperVenueAdapter().connect(paper_connection(initial_equity="10000.00"), None)
    await session.set_quote(PAIR, Decimal("50000"))

    order = await session.place_order(order_intent(amount="0.1"))
    snapshot = await session.fetch_portfolio(PAIR)
    state = await session.list_open_state(PAIR)

    assert isinstance(order, NormalizedOrder)
    assert order.status == "filled"
    assert order.average_price == Decimal("50000")
    assert isinstance(snapshot, ConnectionPortfolioSnapshot)
    assert snapshot.connection_id == "paper-local"
    assert snapshot.equity == Decimal("10000.00")
    assert snapshot.balances == {"USDT": Decimal("10000.00")}
    assert snapshot.position.signed_amount == Decimal("0.1")
    assert snapshot.position.signed_notional == Decimal("5000.0")
    assert isinstance(state, OpenVenueState)
    assert state.position == snapshot.position
    assert state.open_orders == ()


async def test_paper_session_exposes_capabilities_and_deterministic_amount_normalization():
    from cryptotrader.venues.paper import PaperVenueAdapter

    adapter = PaperVenueAdapter()
    session = await adapter.connect(paper_connection(), None)

    assert session.capabilities == adapter.capabilities("paper")
    assert await session.normalize_amount(PAIR, Decimal("0.123456")) == Decimal("0.123456")
    with pytest.raises(ValueError, match="positive finite Decimal"):
        await session.normalize_amount(PAIR, Decimal("0"))
    with pytest.raises(ValueError, match="positive finite Decimal"):
        await session.normalize_amount(PAIR, 0.1)


async def test_paper_accounts_are_independent_per_connection_id():
    from cryptotrader.venues.paper import PaperVenueAdapter

    adapter = PaperVenueAdapter()
    first = await adapter.connect(paper_connection("paper-a", initial_equity="10000"), None)
    second = await adapter.connect(paper_connection("paper-b", initial_equity="25000"), None)
    await first.set_quote(PAIR, Decimal("50000"))
    await second.set_quote(PAIR, Decimal("50000"))

    await first.place_order(order_intent(amount="0.1"))

    assert (await first.fetch_portfolio(PAIR)).position.signed_amount == Decimal("0.1")
    assert (await second.fetch_portfolio(PAIR)).position.signed_amount == Decimal("0")
    assert (await second.fetch_portfolio(PAIR)).equity == Decimal("25000")


async def test_same_connection_id_reconnect_retains_state_but_fresh_adapter_resets_runtime_state():
    from cryptotrader.venues.paper import PaperVenueAdapter

    account = paper_connection("paper-durable", initial_equity="10000")
    adapter = PaperVenueAdapter()
    first = await adapter.connect(account, None)
    await first.set_quote(PAIR, Decimal("50000"))
    await first.place_order(order_intent(amount="0.1"))
    await first.close()

    reconnected = await adapter.connect(account, None)
    assert (await reconnected.fetch_portfolio(PAIR)).position.signed_amount == Decimal("0.1")

    fresh_runtime = await PaperVenueAdapter().connect(account, None)
    await fresh_runtime.set_quote(PAIR, Decimal("50000"))
    fresh_snapshot = await fresh_runtime.fetch_portfolio(PAIR)
    assert fresh_snapshot.equity == Decimal("10000")
    assert fresh_snapshot.position.signed_amount == Decimal("0")


async def test_paper_session_requires_a_quote_before_orders_or_portfolio_reads():
    from cryptotrader.venues.ccxt_base import VenueOperationError
    from cryptotrader.venues.paper import PaperVenueAdapter

    session = await PaperVenueAdapter().connect(paper_connection(), None)

    with pytest.raises(VenueOperationError, match="quote is not set"):
        await session.place_order(order_intent())
    with pytest.raises(VenueOperationError, match="quote is not set"):
        await session.fetch_portfolio(PAIR)


@pytest.mark.parametrize(
    "unsupported_pair",
    [Pair.parse("BTC/USDT:USDT-261225"), OptionPair("BTC", "USDT", "BTC/USDT:USDT")],
)
async def test_paper_session_rejects_market_types_not_declared_by_capabilities(unsupported_pair):
    from cryptotrader.venues.ccxt_base import VenueOperationError
    from cryptotrader.venues.paper import PaperVenueAdapter

    session = await PaperVenueAdapter().connect(paper_connection(), None)

    with pytest.raises(VenueOperationError, match="unsupported Paper market type"):
        await session.set_quote(unsupported_pair, Decimal("100"))
    with pytest.raises(VenueOperationError, match="unsupported Paper market type"):
        await session.fetch_quote(unsupported_pair)
    with pytest.raises(VenueOperationError, match="unsupported Paper market type"):
        await session.place_order(OrderIntent(unsupported_pair, "buy", Decimal("1"), "market", None, False))


async def test_paper_spot_fill_uses_decimal_balances_and_cost_basis():
    from cryptotrader.venues.paper import PaperVenueAdapter

    spot = Pair.parse("BTC/USDT")
    session = await PaperVenueAdapter().connect(paper_connection(initial_equity="10000"), None)
    await session.set_quote(spot, Decimal("50000"))

    bought = await session.place_order(OrderIntent(spot, "buy", Decimal("0.1"), "market", None, False))
    await session.set_quote(spot, Decimal("51000"))
    sold = await session.place_order(OrderIntent(spot, "sell", Decimal("0.04"), "market", None, True))
    portfolio = await session.fetch_portfolio(spot)

    assert bought.status == sold.status == "filled"
    assert portfolio.balances == {"USDT": Decimal("7040.00"), "BTC": Decimal("0.06")}
    assert portfolio.equity == Decimal("10100.00")
    assert portfolio.position.signed_amount == Decimal("0.06")
    assert portfolio.position.entry_price == Decimal("50000")


async def test_paper_close_is_idempotent_and_closed_session_fails_closed():
    from cryptotrader.venues.ccxt_base import VenueOperationError
    from cryptotrader.venues.paper import PaperVenueAdapter

    session = await PaperVenueAdapter().connect(paper_connection(), None)
    await session.close()
    await session.close()

    with pytest.raises(VenueOperationError, match="closed"):
        await session.set_quote(PAIR, Decimal("50000"))


async def test_paper_shared_session_contract():
    from cryptotrader.venues.paper import PaperVenueAdapter
    from tests.contracts.venue_adapter import assert_paper_session_contract

    await assert_paper_session_contract(PaperVenueAdapter)


def test_installed_metadata_loads_all_three_production_venue_factories():
    expected = {"okx", "bybit", "paper"}
    entry_points = tuple(metadata.entry_points(group="cryptotrader.venue_adapters"))
    by_name = {entry_point.name: entry_point for entry_point in entry_points}

    assert expected <= set(by_name)
    adapters = {name: by_name[name].load()() for name in expected}
    assert {name: adapter.adapter_id for name, adapter in adapters.items()} == {
        "okx": "okx",
        "bybit": "bybit",
        "paper": "paper",
    }

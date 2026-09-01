"""Paper venue adapter contract and database-backed account parameters."""

from __future__ import annotations

from decimal import Decimal

import pytest

from cryptotrader.pair import Pair
from cryptotrader.runtime_config.secrets import CredentialPayload
from cryptotrader.venues.models import OrderIntent
from tests.factories.runtime_config import connection

PAIR = Pair.parse("BTC/USDT:USDT")


@pytest.mark.asyncio
async def test_paper_cross_quote_close_reports_unknown_cost_instead_of_zero_profit():
    from cryptotrader.venues.paper import PaperVenueAdapter

    session = await PaperVenueAdapter().connect(connection("paper-cost", "paper"), None)
    acquired, sold = Pair.parse("BTC/USDT"), Pair.parse("BTC/USDC")
    await session.set_quote(acquired, Decimal("100"))
    await session.set_quote(sold, Decimal("110"))
    buy = await session.place_order(OrderIntent(acquired, "buy", Decimal("1"), "market", None, False))
    sell = await session.place_order(OrderIntent(sold, "sell", Decimal("1"), "market", None, False))
    assert buy.status == sell.status == "filled"
    fills = (await session.fetch_fills(None)).items
    assert fills[0].realized_pnl.amount == Decimal("0")
    assert fills[1].realized_pnl.amount is None
    assert fills[1].realized_pnl.currency == "USDC"
    assert fills[1].realized_pnl.unavailable_reason


@pytest.mark.asyncio
async def test_full_account_reads_do_not_apply_bankruptcy_or_recreate_account_start():
    from cryptotrader.venues.models import ProtectionSpec, VenueQuote
    from cryptotrader.venues.paper import PaperVenueAdapter

    adapter = PaperVenueAdapter()
    config = connection("readonly-bankruptcy", "paper", leverage=10, parameters={"initial_equity": "100"})
    session = await adapter.connect(config, None)
    await session.set_quote(PAIR, Decimal("100"))
    await session.place_order(OrderIntent(PAIR, "buy", Decimal("5"), "market", None, False))
    await session.replace_protection(ProtectionSpec(PAIR, "long", Decimal("5"), Decimal("90"), None))
    # Install a market observation without invoking the execution engine's advancement.
    session._account.quotes[PAIR] = VenueQuote(PAIR, Decimal("1"), Decimal("1"), Decimal("1"))
    snapshot = await session.fetch_account()
    page = await session.fetch_fills(None)
    assert snapshot.equity.amount == Decimal("-395")
    assert len(snapshot.positions) == 1
    assert len(snapshot.orders) == 1
    assert len(page.items) == 1
    assert session._account.balances["USDT"] == Decimal("100")
    await session.close()
    reopened = await adapter.connect(config, None)
    assert (await reopened.fetch_fills(None)).coverage_start == page.coverage_start
    assert (await reopened.fetch_fills(page.next_cursor)).items == ()


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

    credentials = CredentialPayload(
        values={
            "api_key": "paper-key",  # pragma: allowlist secret
            "secret": "paper-secret",  # pragma: allowlist secret
        }
    )
    with pytest.raises(ValueError, match="does not accept credentials"):
        await PaperVenueAdapter().connect(paper_connection(), credentials)


async def test_paper_connection_rejects_credential_reference():
    from cryptotrader.venues.paper import PaperVenueAdapter

    with pytest.raises(ValueError, match="does not accept credentials"):
        await PaperVenueAdapter().connect(paper_connection(credential_ref="paper-credentials"), None)


@pytest.mark.parametrize(
    "initial_equity",
    [0, -1, "0", "-0.01", "NaN", "Infinity", "not-a-number", True],
)
async def test_paper_connection_requires_finite_positive_decimal_compatible_initial_equity(initial_equity):
    from cryptotrader.venues.paper import PaperVenueAdapter

    with pytest.raises(ValueError, match="initial_equity"):
        await PaperVenueAdapter().connect(paper_connection(initial_equity=initial_equity), None)


async def test_paper_connection_uses_declared_equity_default_when_parameter_is_absent():
    from cryptotrader.venues.paper import PaperVenueAdapter

    session = await PaperVenueAdapter().connect(paper_connection(initial_equity=None), None)
    await session.set_quote(PAIR, Decimal("50000"))

    assert (await session.fetch_portfolio(PAIR)).equity == Decimal("10000")


async def test_paper_connection_check_validates_the_local_account_without_a_quote_or_order():
    from cryptotrader.venues.paper import PaperVenueAdapter

    session = await PaperVenueAdapter().connect(paper_connection(), None)

    await session.check_connection()

    assert session._account.orders == {}


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


def test_code_registry_constructs_all_three_production_venue_factories():
    expected = {"okx", "bybit", "paper"}
    from cryptotrader.configuration.registry import get_extension_registry

    by_name = get_extension_registry().venues

    assert expected <= set(by_name)
    adapters = {name: by_name[name].factory() for name in expected}
    assert {name: adapter.adapter_id for name, adapter in adapters.items()} == {
        "okx": "okx",
        "bybit": "bybit",
        "paper": "paper",
    }

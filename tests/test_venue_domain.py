"""Platform-neutral venue domain contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from decimal import Decimal

import pytest

from cryptotrader.pair import Pair
from cryptotrader.venues.models import OrderIntent, ProtectionSpec, VenueQuote
from tests.factories.runtime_config import connection


@pytest.mark.parametrize("environment", ["paper", "demo", "testnet", "live"])
def test_connection_accepts_only_declared_environments(environment):
    overrides = {"credential_ref": "live-credentials"} if environment == "live" else {}

    assert connection("c", environment, **overrides).environment == environment


def test_runtime_rejects_undeclared_environment():
    from cryptotrader.runtime_config.models import validate_runtime_document
    from tests.factories.runtime_config import (
        INSTALLED_ADAPTERS,
        INSTALLED_MARKET_SOURCES,
        INSTALLED_SIGNALS,
        runtime_document,
    )

    with pytest.raises(ValueError, match="environment"):
        validate_runtime_document(
            runtime_document(connections=(connection("c", "sandbox"),)),
            INSTALLED_SIGNALS,
            INSTALLED_ADAPTERS,
            INSTALLED_MARKET_SOURCES,
        )


@pytest.mark.asyncio
async def test_live_connection_draft_can_be_saved_without_credentials_but_is_not_eligible():
    from datetime import UTC, datetime
    from types import SimpleNamespace

    from cryptotrader.decision.readiness import book_scope
    from cryptotrader.runtime_config.models import RuntimeConfigSnapshot, validate_runtime_document
    from cryptotrader.venues.models import VenueConnection
    from tests.factories.runtime_config import (
        INSTALLED_ADAPTERS,
        INSTALLED_MARKET_SOURCES,
        INSTALLED_SIGNALS,
        allocation,
        book,
        runtime_document,
    )

    document = runtime_document(
        connections=(VenueConnection("live", "Live", "okx", "live", True, None, 1, "isolated", False),),
        books=(book("real", "real", allocation("live")),),
    )
    validate_runtime_document(document, INSTALLED_SIGNALS, INSTALLED_ADAPTERS, INSTALLED_MARKET_SOURCES)
    assert document.execution.connections[0].credential_ref is None
    assert document.execution.live_order_execution_enabled is False
    scope = await book_scope(RuntimeConfigSnapshot(1, document, datetime.now(UTC)), SimpleNamespace())
    assert scope[0].eligible is False
    assert {reason.code for reason in scope[0].reasons} == {"real_authorization_missing", "credentials_missing"}


def test_connection_parameters_default_to_an_immutable_empty_mapping():
    first = connection("first", "demo")
    second = connection("second", "demo")

    assert first.parameters == {}
    assert first.parameters is not second.parameters
    with pytest.raises(TypeError):
        first.parameters["leak"] = True


def test_capabilities_are_immutable_and_validate_declared_values():
    from cryptotrader.venues.models import VenueCapabilities

    capabilities = VenueCapabilities(
        market_types=frozenset({"spot", "swap"}),
        native_protection=True,
        hedge_mode=False,
        reduce_only=True,
        supported_order_types=frozenset({"market", "limit"}),
    )

    assert capabilities.market_types == frozenset({"spot", "swap"})
    with pytest.raises(FrozenInstanceError):
        capabilities.native_protection = False
    with pytest.raises(ValueError, match="market_types"):
        VenueCapabilities(frozenset({"margin"}), True, False, True, frozenset({"market"}))
    with pytest.raises(ValueError, match="unknown capability"):
        VenueCapabilities(
            frozenset({"spot"}),
            False,
            False,
            True,
            frozenset({"market"}),
            unknown_fields=frozenset({"native_protection"}),
        )


def test_venue_values_keep_decimal_precision_and_open_state_is_immutable():
    from cryptotrader.venues.models import (
        ConnectionPosition,
        NormalizedOrder,
        OpenVenueState,
        OrderIntent,
        ProtectionSpec,
        ProtectionState,
        VenueQuote,
    )

    pair = Pair.parse("BTC/USDT:USDT")
    quote = VenueQuote(pair, Decimal("49999.99"), Decimal("50000.01"), Decimal("50000.00"))
    position = ConnectionPosition(pair, Decimal("0.1"), Decimal("5000.000"), Decimal("50000.00"))
    intent = OrderIntent(pair, "buy", Decimal("0.1"), "market", None, False)
    order = NormalizedOrder(
        "order-1",
        pair,
        "buy",
        "market",
        Decimal("0.1"),
        Decimal("0.1"),
        Decimal("50000.00"),
        "filled",
        False,
    )
    spec = ProtectionSpec(pair, "long", Decimal("0.1"), Decimal("48000"), Decimal("55000"))
    protection = ProtectionState(
        ("stop-1", "take-1"),
        pair,
        "long",
        Decimal("0.1"),
        Decimal("48000"),
        Decimal("55000"),
        True,
        False,
    )
    state = OpenVenueState(position, (order,), (protection,))

    assert quote.last == Decimal("50000.00")
    assert intent.amount == Decimal("0.1")
    assert order.average_price == Decimal("50000.00")
    assert spec.stop_loss == Decimal("48000")
    assert state.open_orders == (order,)
    assert state.triggered_protections == ()
    with pytest.raises(FrozenInstanceError):
        state.position = position


def test_open_venue_state_rejects_orders_and_protections_from_another_pair():
    from cryptotrader.venues.models import ConnectionPosition, NormalizedOrder, OpenVenueState, ProtectionState

    pair = Pair.parse("BTC/USDT:USDT")
    other_pair = Pair.parse("ETH/USDT:USDT")
    position = ConnectionPosition(pair, Decimal("1"), Decimal("100"), Decimal("100"))
    other_order = NormalizedOrder(
        "other-order",
        other_pair,
        "buy",
        "market",
        Decimal("1"),
        Decimal("0"),
        None,
        "open",
        False,
    )
    other_protection = ProtectionState(
        ("other-stop",),
        other_pair,
        "long",
        Decimal("1"),
        Decimal("90"),
        None,
        True,
        False,
    )

    with pytest.raises(ValueError, match="open order pairs"):
        OpenVenueState(position, (other_order,), ())
    with pytest.raises(ValueError, match="protection pairs"):
        OpenVenueState(position, (), (other_protection,))


@pytest.mark.parametrize(
    ("side", "stop_loss", "take_profit"),
    [
        ("long", Decimal("100"), None),
        ("long", None, Decimal("100")),
        ("short", Decimal("100"), None),
        ("short", None, Decimal("100")),
    ],
)
def test_protection_spec_rejects_prices_on_the_wrong_side_of_execution(side, stop_loss, take_profit):
    spec = ProtectionSpec(Pair.parse("BTC/USDT:USDT"), side, Decimal("1"), stop_loss, take_profit)

    with pytest.raises(ValueError, match="protection price geometry"):
        spec.validate_geometry(Decimal("100"))


@pytest.mark.parametrize(
    ("side", "stop_loss", "take_profit"),
    [
        ("long", Decimal("99"), None),
        ("long", None, Decimal("101")),
        ("short", Decimal("101"), None),
        ("short", None, Decimal("99")),
    ],
)
def test_protection_spec_accepts_one_logically_positioned_price(side, stop_loss, take_profit):
    spec = ProtectionSpec(Pair.parse("BTC/USDT:USDT"), side, Decimal("1"), stop_loss, take_profit)

    spec.validate_geometry(Decimal("100"))


@pytest.mark.parametrize(
    "factory",
    [
        lambda pair: VenueQuote(pair, 49999.99, Decimal("50000.01"), Decimal("50000")),
        lambda pair: OrderIntent(pair, "buy", 0.1, "market", None, False),
        lambda pair: ProtectionSpec(pair, "long", Decimal("0.1"), 48000.0, Decimal("55000")),
    ],
)
def test_money_amount_and_price_fields_reject_binary_float(factory):
    with pytest.raises(ValueError, match="Decimal"):
        factory(Pair.parse("BTC/USDT:USDT"))


def test_venue_protocols_are_structural_and_contract_helper_checks_capabilities():  # noqa: C901
    from cryptotrader.venues.models import VenueCapabilities
    from cryptotrader.venues.protocol import VenueAdapter, VenueSession
    from tests.contracts.venue_adapter import assert_venue_contract

    class FakeSession:
        connection_id = "paper-local"
        capabilities = VenueCapabilities(frozenset({"spot", "swap"}), True, False, True, frozenset({"market"}))

        async def check_connection(self):
            return None

        async def list_instruments(self):
            return ()

        async def fetch_account(self):
            return None

        async def fetch_fills(self, cursor):
            return None

        async def fetch_funding(self, cursor):
            return None

        async def fetch_portfolio(self, pair):
            return None

        async def fetch_quote(self, pair):
            return None

        async def normalize_amount(self, pair, base_amount):
            return base_amount

        async def minimum_amount(self, pair, reference_price, minimum_quote_notional):
            return None

        async def place_order(self, intent):
            return None

        async def cancel_order(self, order_id, pair):
            return None

        async def find_order(self, pair, *, order_id=None, client_order_id=None):
            return None

        async def normalize_protection(self, spec):
            return spec

        async def replace_protection(self, spec):
            return None

        async def cancel_protection(self, protection_ids):
            return None

        async def list_open_state(self, pair):
            return None

        async def close(self):
            return None

    class FakeAdapter:
        adapter_id = "paper"

        def capabilities(self, environment):
            return VenueCapabilities(frozenset({"spot", "swap"}), True, False, True, frozenset({"market"}))

        async def connect(self, connection, credentials):
            return FakeSession()

    adapter = FakeAdapter()
    session = FakeSession()

    assert isinstance(adapter, VenueAdapter)
    assert isinstance(session, VenueSession)
    assert_venue_contract(FakeAdapter, ("paper",))


def test_shared_venue_contract_rejects_synchronous_connect():
    from cryptotrader.venues.models import VenueCapabilities
    from tests.contracts.venue_adapter import assert_venue_contract

    class SyncConnectAdapter:
        adapter_id = "sync"

        def capabilities(self, environment):
            return VenueCapabilities(frozenset({"swap"}), True, False, True, frozenset({"market"}))

        def connect(self, connection, credentials):
            raise NotImplementedError

    with pytest.raises(AssertionError, match="connect must be async"):
        assert_venue_contract(SyncConnectAdapter, ("paper",))

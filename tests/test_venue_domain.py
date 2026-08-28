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


def test_connection_rejects_undeclared_environment():
    with pytest.raises(ValueError, match="environment"):
        connection("c", "sandbox")


def test_live_connection_requires_credential_reference():
    from cryptotrader.venues.models import VenueConnection

    with pytest.raises(ValueError, match="credential_ref"):
        VenueConnection("live", "Live", "okx", "live", True, None, 1, "isolated")


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


def test_venue_protocols_are_structural_and_contract_helper_checks_capabilities():
    from cryptotrader.venues.models import VenueCapabilities
    from cryptotrader.venues.protocol import VenueAdapter, VenueSession
    from tests.contracts.venue_adapter import assert_venue_contract

    class FakeSession:
        connection_id = "paper-local"

        async def fetch_portfolio(self, pair):
            return None

        async def fetch_quote(self, pair):
            return None

        async def place_order(self, intent):
            return None

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

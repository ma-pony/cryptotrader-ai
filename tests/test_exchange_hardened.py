"""Regression coverage for the hardened Decimal venue boundary."""

from decimal import Decimal

import pytest

from cryptotrader.pair import Pair
from cryptotrader.runtime_config.secrets import CredentialPayload
from cryptotrader.venues.models import OrderIntent
from cryptotrader.venues.protocol import VenueAdapter, VenueSession
from tests.factories.runtime_config import connection
from tests.fakes.ccxt_client import FakeCcxtFactory


async def _session():
    from cryptotrader.venues.okx import OkxVenueAdapter

    adapter = OkxVenueAdapter(client_factory=FakeCcxtFactory("okx"))
    session = await adapter.connect(
        connection("okx-demo", "demo", adapter_id="okx", credential_ref="credentials"),
        CredentialPayload(
            values={
                "api_key": "key",  # pragma: allowlist secret
                "secret": "secret",  # pragma: allowlist secret
                "passphrase": "passphrase",  # pragma: allowlist secret
            }
        ),
    )
    return adapter, session


@pytest.mark.asyncio
async def test_okx_adapter_and_session_are_structural_task_4_protocols():
    adapter, session = await _session()

    assert isinstance(adapter, VenueAdapter)
    assert isinstance(session, VenueSession)


@pytest.mark.asyncio
async def test_okx_precision_never_returns_binary_float_above_session_boundary():
    _, session = await _session()

    order = await session.place_order(
        OrderIntent(Pair.parse("BTC/USDT:USDT"), "buy", Decimal("0.123"), "market", None, False)
    )

    assert order.amount == Decimal("0.12")
    assert order.filled_amount == Decimal("0.12")
    assert isinstance(order.average_price, Decimal)

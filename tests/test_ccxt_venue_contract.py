"""Shared executable contract for both CCXT venue implementations."""

import pytest

from cryptotrader.pair import Pair
from cryptotrader.runtime_config.secrets import CredentialPayload
from cryptotrader.venues.ccxt_base import VenueOperationError
from tests.contracts.venue_adapter import assert_ccxt_session_contract, assert_venue_contract
from tests.factories.runtime_config import connection
from tests.fakes.ccxt_client import FakeCcxtFactory


@pytest.mark.parametrize(
    ("adapter_module", "adapter_name", "exchange_id", "environments"),
    [
        ("cryptotrader.venues.okx", "OkxVenueAdapter", "okx", ("demo", "live")),
        ("cryptotrader.venues.bybit", "BybitVenueAdapter", "bybit", ("testnet", "demo", "live")),
    ],
)
def test_ccxt_adapters_implement_the_task_4_adapter_contract(adapter_module, adapter_name, exchange_id, environments):
    import importlib

    adapter_class = getattr(importlib.import_module(adapter_module), adapter_name)
    assert_venue_contract(lambda: adapter_class(client_factory=FakeCcxtFactory(exchange_id)), environments)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("adapter_module", "adapter_name", "exchange_id", "environment"),
    [
        ("cryptotrader.venues.okx", "OkxVenueAdapter", "okx", "demo"),
        ("cryptotrader.venues.bybit", "BybitVenueAdapter", "bybit", "testnet"),
    ],
)
async def test_ccxt_adapters_share_portfolio_order_protection_and_close_contract(
    adapter_module, adapter_name, exchange_id, environment
):
    import importlib

    adapter_class = getattr(importlib.import_module(adapter_module), adapter_name)
    await assert_ccxt_session_contract(lambda: adapter_class(client_factory=FakeCcxtFactory(exchange_id)), environment)


@pytest.mark.asyncio
async def test_ccxt_error_boundary_does_not_expose_raw_exchange_message_or_cause():
    from cryptotrader.venues.okx import OkxVenueAdapter

    adapter = OkxVenueAdapter(client_factory=FakeCcxtFactory("okx"))
    session = await adapter.connect(
        connection("okx-demo", "demo", adapter_id="okx", credential_ref="credentials"),
        CredentialPayload(api_key="key", secret="secret", passphrase="passphrase"),  # pragma: allowlist secret
    )

    async def fail_with_raw_response(*_args):
        raise RuntimeError("raw-response-with-secret")

    session.client.fetch_ticker = fail_with_raw_response
    with pytest.raises(VenueOperationError) as caught:
        await session.fetch_quote(Pair.parse("BTC/USDT:USDT"))

    assert str(caught.value) == "okx-demo: fetch quote failed"
    assert caught.value.__cause__ is None

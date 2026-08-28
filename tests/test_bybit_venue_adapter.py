"""Bybit environment, order, and native protection behavior."""

from decimal import Decimal

import pytest

from cryptotrader.pair import Pair
from cryptotrader.runtime_config.secrets import CredentialPayload
from cryptotrader.venues.models import OrderIntent, ProtectionSpec
from tests.factories.runtime_config import connection
from tests.fakes.ccxt_client import FakeCcxtFactory


async def _connect(environment: str = "testnet"):
    from cryptotrader.venues.bybit import BybitVenueAdapter

    factory = FakeCcxtFactory("bybit")
    adapter = BybitVenueAdapter(client_factory=factory)
    session = await adapter.connect(
        connection("bybit", environment, adapter_id="bybit", credential_ref="credentials"),
        CredentialPayload(api_key="key", secret="secret"),  # pragma: allowlist secret
    )
    return adapter, session, factory


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("environment", "hostname"),
    [("testnet", "api-testnet.bybit.com"), ("demo", "api-demo.bybit.com"), ("live", "api.bybit.com")],
)
async def test_bybit_environment_maps_to_official_endpoint(environment, hostname):
    _, session, _ = await _connect(environment)

    assert hostname in session.client.urls["api"]["public"]


@pytest.mark.asyncio
async def test_bybit_uses_unified_account_order_params_for_hedge_position():
    _, session, factory = await _connect()
    intent = OrderIntent(Pair.parse("BTC/USDT:USDT"), "sell", Decimal("0.1"), "market", None, True)

    order = await session.place_order(intent)

    create = next(payload for name, payload in session.client.calls if name == "create_order")
    assert factory.configs[0]["options"]["defaultType"] == "swap"
    assert create[3] == 10.0
    assert create[5] == {"positionIdx": 1, "reduceOnly": True}
    assert order.reduce_only is True


@pytest.mark.asyncio
async def test_bybit_uses_official_zero_position_index_for_one_way_mode():
    _, session, _ = await _connect()
    session.client.position_contracts = "0"
    session.client.position_index = 0
    intent = OrderIntent(Pair.parse("BTC/USDT:USDT"), "buy", Decimal("0.1"), "market", None, False)

    await session.place_order(intent)

    create = next(payload for name, payload in session.client.calls if name == "create_order")
    assert create[5] == {"positionIdx": 0}


@pytest.mark.asyncio
async def test_bybit_protection_uses_trading_stop_and_requires_position_query_confirmation():
    _, session, _ = await _connect()
    spec = ProtectionSpec(Pair.parse("BTC/USDT:USDT"), "long", Decimal("0.02"), Decimal("48000"), Decimal("55000"))

    protection = await session.replace_protection(spec)

    request = next(payload for name, payload in session.client.calls if name == "private_post_v5_position_trading_stop")
    assert request == {
        "category": "linear",
        "symbol": "BTCUSDT",
        "tpslMode": "Full",
        "positionIdx": 1,
        "stopLoss": "48000.0",
        "takeProfit": "55000.0",
        "slTriggerBy": "LastPrice",
        "tpTriggerBy": "LastPrice",
        "slOrderType": "Market",
        "tpOrderType": "Market",
    }
    names = [name for name, _ in session.client.calls]
    create_index = names.index("private_post_v5_position_trading_stop")
    assert "fetch_positions" in names[create_index + 1 :]
    assert protection.active is True


@pytest.mark.asyncio
async def test_bybit_rejects_unconfirmed_protection():
    from cryptotrader.venues.ccxt_base import VenueOperationError

    _, session, _ = await _connect()
    session.client.confirm_protection = False
    spec = ProtectionSpec(Pair.parse("BTC/USDT:USDT"), "long", Decimal("0.02"), Decimal("48000"), Decimal("55000"))

    with pytest.raises(VenueOperationError, match="not confirmed"):
        await session.replace_protection(spec)

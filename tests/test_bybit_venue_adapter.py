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
    _, _, factory = await _connect(environment)

    assert hostname in factory.clients[-1].urls["api"]["public"]


@pytest.mark.asyncio
async def test_bybit_uses_unified_account_order_params_for_hedge_position():
    _, session, factory = await _connect()
    intent = OrderIntent(Pair.parse("BTC/USDT:USDT"), "sell", Decimal("0.1"), "market", None, True)

    order = await session.place_order(intent)

    create = next(payload for name, payload in factory.clients[-1].calls if name == "create_order")
    assert factory.configs[0]["options"]["defaultType"] == "swap"
    assert create[3] == 0.1
    assert create[5] == {"positionIdx": 1, "reduceOnly": True}
    assert order.reduce_only is True


@pytest.mark.asyncio
async def test_bybit_uses_official_zero_position_index_for_one_way_mode():
    _, session, factory = await _connect()
    factory.clients[-1].position_contracts = "0"
    factory.clients[-1].position_index = 0
    factory.clients[-1].hedged = False
    intent = OrderIntent(Pair.parse("BTC/USDT:USDT"), "buy", Decimal("0.1"), "market", None, False)

    await session.place_order(intent)

    create = next(payload for name, payload in factory.clients[-1].calls if name == "create_order")
    assert create[5] == {"positionIdx": 0}


@pytest.mark.asyncio
async def test_bybit_protection_uses_trading_stop_and_requires_position_query_confirmation():
    _, session, factory = await _connect()
    spec = ProtectionSpec(Pair.parse("BTC/USDT:USDT"), "long", Decimal("0.02"), Decimal("48000"), Decimal("55000"))

    protection = await session.replace_protection(spec)

    request = next(
        payload for name, payload in factory.clients[-1].calls if name == "private_post_v5_position_trading_stop"
    )
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
    names = [name for name, _ in factory.clients[-1].calls]
    create_index = names.index("private_post_v5_position_trading_stop")
    assert "fetch_positions" in names[create_index + 1 :]
    assert protection.active is True


@pytest.mark.asyncio
async def test_bybit_rejects_unconfirmed_protection():
    from cryptotrader.venues.ccxt_base import VenueOperationError

    _, session, factory = await _connect()
    factory.clients[-1].confirm_protection = False
    spec = ProtectionSpec(Pair.parse("BTC/USDT:USDT"), "long", Decimal("0.02"), Decimal("48000"), Decimal("55000"))

    with pytest.raises(VenueOperationError, match="not confirmed"):
        await session.replace_protection(spec)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("hedged", "side", "reduce_only", "expected"),
    [
        (False, "buy", False, 0),
        (False, "sell", True, 0),
        (True, "buy", False, 1),
        (True, "sell", False, 2),
        (True, "sell", True, 1),
        (True, "buy", True, 2),
    ],
)
async def test_bybit_position_index_comes_from_official_mode_and_order_intent(hedged, side, reduce_only, expected):
    _, session, factory = await _connect()
    client = factory.clients[-1]
    client.hedged = hedged
    client.position_index = 2 if side == "buy" else 1
    client.position_side = "short" if client.position_index == 2 else "long"

    await session.place_order(
        OrderIntent(Pair.parse("BTC/USDT:USDT"), side, Decimal("0.1"), "market", None, reduce_only)
    )

    request = next(payload for name, payload in client.calls if name == "create_order")
    assert request[5]["positionIdx"] == expected


@pytest.mark.asyncio
async def test_bybit_one_sided_full_protection_explicitly_clears_old_omitted_side():
    _, session, factory = await _connect()
    pair = Pair.parse("BTC/USDT:USDT")
    await session.replace_protection(ProtectionSpec(pair, "long", Decimal("0.02"), Decimal("48000"), Decimal("55000")))

    replacement = await session.replace_protection(
        ProtectionSpec(pair, "long", Decimal("0.02"), Decimal("47000"), None)
    )

    request = [
        payload for name, payload in factory.clients[-1].calls if name == "private_post_v5_position_trading_stop"
    ][-1]
    assert request["stopLoss"] == "47000.0"
    assert request["takeProfit"] == "0"
    assert replacement.stop_loss == Decimal("47000")
    assert replacement.take_profit is None


@pytest.mark.asyncio
async def test_bybit_position_index_zero_is_preserved_when_reading_protection():
    _, session, factory = await _connect()
    client = factory.clients[-1]
    client.hedged = False
    client.position_index = 0
    await session.replace_protection(
        ProtectionSpec(Pair.parse("BTC/USDT:USDT"), "long", Decimal("0.02"), Decimal("48000"), None)
    )

    protection = (await session.list_open_state(Pair.parse("BTC/USDT:USDT"))).protections[0]

    assert protection.protection_ids == ("bybit-position:BTCUSDT:0",)


@pytest.mark.asyncio
async def test_bybit_invalid_native_return_code_is_normalized_without_raw_payload():
    from cryptotrader.venues.ccxt_base import VenueOperationError

    _, session, factory = await _connect()

    async def malformed(_params):
        return {"retCode": "raw-secret-value", "result": {"credential": "raw-secret-value"}}

    factory.clients[-1].private_post_v5_position_trading_stop = malformed
    spec = ProtectionSpec(
        Pair.parse("BTC/USDT:USDT"),
        "long",
        Decimal("0.02"),
        Decimal("48000"),
        None,
    )

    with pytest.raises(VenueOperationError) as caught:
        await session.replace_protection(spec)

    assert "raw-secret-value" not in str(caught.value)
    assert caught.value.__cause__ is None


@pytest.mark.asyncio
async def test_bybit_one_way_short_does_not_confirm_requested_long_protection():
    from cryptotrader.venues.ccxt_base import VenueOperationError

    _, session, factory = await _connect()
    client = factory.clients[-1]
    client.hedged = False
    client.position_index = 0
    client.position_side = "short"
    spec = ProtectionSpec(
        Pair.parse("BTC/USDT:USDT"),
        "long",
        Decimal("0.02"),
        Decimal("48000"),
        Decimal("55000"),
    )

    with pytest.raises(VenueOperationError, match="not confirmed"):
        await session.replace_protection(spec)


@pytest.mark.asyncio
async def test_bybit_one_way_short_confirms_requested_short_protection():
    _, session, factory = await _connect()
    client = factory.clients[-1]
    client.hedged = False
    client.position_index = 0
    client.position_side = "short"
    spec = ProtectionSpec(
        Pair.parse("BTC/USDT:USDT"),
        "short",
        Decimal("0.02"),
        Decimal("56000"),
        Decimal("45000"),
    )

    protection = await session.replace_protection(spec)

    assert protection.position_side == "short"

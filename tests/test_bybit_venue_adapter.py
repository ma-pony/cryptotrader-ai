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
        CredentialPayload(values={"api_key": "key", "secret": "secret"}),  # pragma: allowlist secret
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
async def test_bybit_spot_order_keeps_the_common_client_order_identifier():
    _, session, factory = await _connect()
    await session.place_order(
        OrderIntent(Pair.parse("BTC/USDT"), "buy", Decimal("0.1"), "market", None, False, "CTSPOTBYBITO")
    )
    create = next(payload for name, payload in factory.clients[-1].calls if name == "create_order")
    assert create[5] == {"orderLinkId": "CTSPOTBYBITO"}


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


@pytest.mark.asyncio
async def test_bybit_conditional_reduce_only_order_is_grouped_as_protection():
    from tests.fakes.account_client import account_session

    session, client = await account_session("bybit")
    original = client.private_get_v5_order_realtime

    async def conditional_close(params):
        response = await original(params)
        for row in response["result"]["list"]:
            if row["orderId"] == "protect1":
                row["stopOrderType"] = "Stop"
                row["closeOnTrigger"] = True
        return response

    client.private_get_v5_order_realtime = conditional_close
    snapshot = await session.fetch_account()
    assert next(order for order in snapshot.orders if order.venue_order_id == "protect1").protection


@pytest.mark.asyncio
async def test_bybit_missing_fee_currency_and_option_cash_flow_do_not_become_known_profit():
    from tests.fakes.account_client import account_session
    from tests.test_account_read_contract import collect_pages

    session, client = await account_session("bybit")
    original = client.private_get_v5_execution_list

    async def no_currency(params):
        response = await original(params)
        for row in response["result"]["list"]:
            row["feeCurrency"] = ""
        return response

    client.private_get_v5_execution_list = no_currency
    fills, _ = await collect_pages(session.fetch_fills)
    fill = next(fill for fill in fills if fill.venue_fill_id == "trade1")
    assert fill.fee.amount is None
    assert "currency" in fill.fee.unavailable_reason
    option = next(fill for fill in fills if fill.venue_fill_id == "eth-option-fill")
    assert option.realized_pnl.amount is None
    assert option.realized_pnl.unavailable_reason
    assert option.instrument.tradable is False


@pytest.mark.asyncio
async def test_bybit_explicit_cancel_can_target_real_protection_id_from_full_account_read():
    from tests.fakes.account_client import account_session

    session, client = await account_session("bybit")
    pending = {"protect1"}

    async def cancel_order(order_id, symbol):
        assert symbol == "BTC/USDT:USDT"
        pending.remove(order_id)

    client.cancel_order = cancel_order
    snapshot = await session.fetch_account()
    assert pending == {"protect1"}
    protective_order = next(order for order in snapshot.orders if order.protection)
    await session.cancel_protection((protective_order.venue_order_id,))
    assert pending == set()


@pytest.mark.asyncio
async def test_bybit_full_account_protection_cancel_failure_is_safe_and_does_not_forget_target():
    from cryptotrader.venues.protocol import VenueOperationError
    from tests.fakes.account_client import account_session

    session, client = await account_session("bybit")
    pending = {"protect1"}

    async def rejected(_order_id, _symbol):
        raise RuntimeError("fixture-private-message")

    client.cancel_order = rejected
    await session.fetch_account()
    with pytest.raises(VenueOperationError) as caught:
        await session.cancel_protection(("protect1",))
    assert "fixture-private-message" not in str(caught.value)
    assert pending == {"protect1"}

    async def accepted(order_id, _symbol):
        pending.remove(order_id)

    client.cancel_order = accepted
    await session.cancel_protection(("protect1",))
    assert pending == set()

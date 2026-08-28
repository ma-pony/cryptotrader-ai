"""OKX environment and venue-native parameter behavior."""

from decimal import Decimal

import pytest

from cryptotrader.pair import Pair
from cryptotrader.runtime_config.secrets import CredentialPayload
from cryptotrader.venues.models import OrderIntent, ProtectionSpec
from tests.factories.runtime_config import connection
from tests.fakes.ccxt_client import FakeCcxtFactory


async def _connect(environment: str = "demo"):
    from cryptotrader.venues.okx import OkxVenueAdapter

    factory = FakeCcxtFactory("okx")
    adapter = OkxVenueAdapter(client_factory=factory)
    session = await adapter.connect(
        connection("okx", environment, adapter_id="okx", credential_ref="credentials"),
        CredentialPayload(api_key="key", secret="secret", passphrase="passphrase"),  # pragma: allowlist secret
    )
    return adapter, session, factory


@pytest.mark.asyncio
async def test_okx_demo_sets_simulated_header_only_for_demo():
    _, demo, _ = await _connect("demo")
    _, live, _ = await _connect("live")

    assert demo.client.headers["x-simulated-trading"] == "1"
    assert "x-simulated-trading" not in live.client.headers


@pytest.mark.asyncio
async def test_okx_client_uses_passphrase_and_scoped_market_types():
    _, _, factory = await _connect()

    assert factory.configs[0]["password"] == "passphrase"  # pragma: allowlist secret
    assert factory.configs[0]["options"]["fetchMarkets"] == ["spot", "swap"]


@pytest.mark.asyncio
async def test_okx_swap_order_uses_contracts_td_mode_position_side_and_reduce_only():
    _, session, _ = await _connect()
    intent = OrderIntent(Pair.parse("BTC/USDT:USDT"), "sell", Decimal("0.1"), "market", None, True)

    order = await session.place_order(intent)

    create = next(payload for name, payload in session.client.calls if name == "create_order")
    assert create[3] == 10.0
    assert create[5] == {"tdMode": "isolated", "posSide": "long", "reduceOnly": True}
    assert order.amount == Decimal("0.1")
    assert order.filled_amount == Decimal("0.1")
    assert order.reduce_only is True


@pytest.mark.asyncio
async def test_okx_protection_is_returned_only_after_pending_algo_query_confirms_it():
    _, session, _ = await _connect()
    spec = ProtectionSpec(Pair.parse("BTC/USDT:USDT"), "long", Decimal("0.02"), Decimal("48000"), Decimal("55000"))

    protection = await session.replace_protection(spec)

    names = [name for name, _ in session.client.calls]
    create_index = names.index("private_post_trade_order_algo")
    assert "private_get_trade_orders_algo_pending" in names[create_index + 1 :]
    assert protection.protection_ids == ("algo-1",)
    assert protection.active is True


@pytest.mark.asyncio
async def test_okx_rejects_unconfirmed_protection():
    from cryptotrader.venues.ccxt_base import VenueOperationError

    _, session, _ = await _connect()
    session.client.confirm_protection = False
    spec = ProtectionSpec(Pair.parse("BTC/USDT:USDT"), "long", Decimal("0.02"), Decimal("48000"), Decimal("55000"))

    with pytest.raises(VenueOperationError, match="not confirmed"):
        await session.replace_protection(spec)


@pytest.mark.asyncio
async def test_okx_replacement_confirms_new_protection_before_cancelling_old():
    _, session, _ = await _connect()
    pair = Pair.parse("BTC/USDT:USDT")
    old = await session.replace_protection(
        ProtectionSpec(pair, "long", Decimal("0.02"), Decimal("48000"), Decimal("55000"))
    )
    session.client.calls.clear()

    replacement = await session.replace_protection(
        ProtectionSpec(pair, "long", Decimal("0.02"), Decimal("47000"), Decimal("56000"))
    )

    names = [name for name, _ in session.client.calls]
    create_index = names.index("private_post_trade_order_algo")
    confirm_index = names.index("private_get_trade_orders_algo_pending", create_index + 1)
    cancel_index = names.index("private_post_trade_cancel_algos")
    assert create_index < confirm_index < cancel_index
    assert old.protection_ids == ("algo-1",)
    assert replacement.protection_ids == ("algo-2",)
    assert (await session.list_open_state(pair)).protections == (replacement,)

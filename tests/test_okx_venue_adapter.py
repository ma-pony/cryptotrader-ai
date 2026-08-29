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
    _, _, demo_factory = await _connect("demo")
    _, _, live_factory = await _connect("live")

    assert demo_factory.clients[-1].headers["x-simulated-trading"] == "1"
    assert "x-simulated-trading" not in live_factory.clients[-1].headers


@pytest.mark.asyncio
async def test_okx_client_uses_passphrase_and_scoped_market_types():
    _, _, factory = await _connect()

    assert factory.configs[0]["password"] == "passphrase"  # pragma: allowlist secret
    assert factory.configs[0]["options"]["fetchMarkets"] == ["spot", "swap"]


@pytest.mark.asyncio
async def test_okx_hedge_reduce_keeps_normalized_semantics_without_exchange_reduce_only_param():
    _, session, factory = await _connect()
    intent = OrderIntent(Pair.parse("BTC/USDT:USDT"), "sell", Decimal("0.1"), "market", None, True)

    order = await session.place_order(intent)

    create = next(payload for name, payload in factory.clients[-1].calls if name == "create_order")
    assert create[3] == 10.0
    assert create[5] == {"tdMode": "isolated", "posSide": "long"}
    assert order.amount == Decimal("0.1")
    assert order.filled_amount == Decimal("0.1")
    assert order.reduce_only is True


@pytest.mark.asyncio
async def test_okx_protection_is_returned_only_after_pending_algo_query_confirms_it():
    _, session, factory = await _connect()
    spec = ProtectionSpec(Pair.parse("BTC/USDT:USDT"), "long", Decimal("0.02"), Decimal("48000"), Decimal("55000"))

    protection = await session.replace_protection(spec)

    names = [name for name, _ in factory.clients[-1].calls]
    create_index = names.index("private_post_trade_order_algo")
    assert "private_get_trade_orders_algo_pending" in names[create_index + 1 :]
    assert protection.protection_ids == ("algo-1",)
    assert protection.active is True


@pytest.mark.asyncio
async def test_okx_normalized_protection_spec_matches_the_subsequent_readback_exactly():
    _, session, _factory = await _connect()
    pair = Pair.parse("BTC/USDT:USDT")
    raw = ProtectionSpec(pair, "long", Decimal("0.029"), Decimal("48000.09"), Decimal("55000.09"))

    normalized = await session.normalize_protection(raw)
    protection = await session.replace_protection(normalized)

    assert normalized.amount == Decimal("0.02")
    assert normalized.stop_loss == Decimal("48000.0")
    assert normalized.take_profit == Decimal("55000.0")
    assert protection.amount == normalized.amount
    assert protection.stop_loss == normalized.stop_loss
    assert protection.take_profit == normalized.take_profit


@pytest.mark.asyncio
async def test_okx_rejects_unconfirmed_protection():
    from cryptotrader.venues.ccxt_base import VenueOperationError

    _, session, factory = await _connect()
    factory.clients[-1].confirm_protection = False
    spec = ProtectionSpec(Pair.parse("BTC/USDT:USDT"), "long", Decimal("0.02"), Decimal("48000"), Decimal("55000"))

    with pytest.raises(VenueOperationError, match="not confirmed"):
        await session.replace_protection(spec)


@pytest.mark.asyncio
async def test_okx_replacement_confirms_new_protection_before_cancelling_old():
    _, session, factory = await _connect()
    pair = Pair.parse("BTC/USDT:USDT")
    old = await session.replace_protection(
        ProtectionSpec(pair, "long", Decimal("0.02"), Decimal("48000"), Decimal("55000"))
    )
    factory.clients[-1].calls.clear()

    replacement = await session.replace_protection(
        ProtectionSpec(pair, "long", Decimal("0.02"), Decimal("47000"), Decimal("56000"))
    )

    names = [name for name, _ in factory.clients[-1].calls]
    create_index = names.index("private_post_trade_order_algo")
    confirm_index = names.index("private_get_trade_orders_algo_pending", create_index + 1)
    cancel_index = names.index("private_post_trade_cancel_algos")
    assert create_index < confirm_index < cancel_index
    assert old.protection_ids == ("algo-1",)
    assert replacement.protection_ids == ("algo-2",)
    assert (await session.list_open_state(pair)).protections == (replacement,)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("hedged", "side", "amount", "reduce_only", "expected"),
    [
        pytest.param(False, "buy", "0.1", False, {"tdMode": "isolated", "posSide": "net"}, id="net-open"),
        pytest.param(
            False,
            "sell",
            "0.04",
            True,
            {"tdMode": "isolated", "posSide": "net", "reduceOnly": True},
            id="net-reduce",
        ),
        pytest.param(
            False,
            "sell",
            "0.02",
            True,
            {"tdMode": "isolated", "posSide": "net", "reduceOnly": True},
            id="net-close",
        ),
        pytest.param(True, "buy", "0.1", False, {"tdMode": "isolated", "posSide": "long"}, id="hedge-open-long"),
        pytest.param(True, "sell", "0.1", False, {"tdMode": "isolated", "posSide": "short"}, id="hedge-open-short"),
        pytest.param(True, "sell", "0.04", True, {"tdMode": "isolated", "posSide": "long"}, id="hedge-reduce-long"),
        pytest.param(True, "buy", "0.04", True, {"tdMode": "isolated", "posSide": "short"}, id="hedge-reduce-short"),
        pytest.param(True, "sell", "0.02", True, {"tdMode": "isolated", "posSide": "long"}, id="hedge-close-long"),
        pytest.param(True, "buy", "0.02", True, {"tdMode": "isolated", "posSide": "short"}, id="hedge-close-short"),
    ],
)
async def test_okx_order_params_follow_account_mode_and_open_reduce_close_intent(
    hedged, side, amount, reduce_only, expected
):
    _, session, factory = await _connect()
    client = factory.clients[-1]
    client.hedged = hedged
    client.position_side = "short" if side == "buy" else "long"

    await session.place_order(
        OrderIntent(Pair.parse("BTC/USDT:USDT"), side, Decimal(amount), "market", None, reduce_only)
    )

    request = next(payload for name, payload in client.calls if name == "create_order")
    assert request[5] == expected


@pytest.mark.asyncio
async def test_okx_canary_client_order_id_uses_the_strict_common_platform_format():
    _, session, factory = await _connect()
    client_order_id = "CT0123456789ABCDEFO"

    await session.place_order(
        OrderIntent(Pair.parse("BTC/USDT:USDT"), "buy", Decimal("0.1"), "market", None, False, client_order_id)
    )

    request = next(payload for name, payload in factory.clients[-1].calls if name == "create_order")
    assert request[5]["clOrdId"] == client_order_id
    assert len(request[5]["clOrdId"]) <= 32
    assert request[5]["clOrdId"].isalnum()


@pytest.mark.asyncio
async def test_okx_spot_order_keeps_the_common_client_order_identifier():
    _, session, factory = await _connect()
    await session.place_order(
        OrderIntent(Pair.parse("BTC/USDT"), "buy", Decimal("0.1"), "market", None, False, "CTSPOTOKXO")
    )
    request = next(payload for name, payload in factory.clients[-1].calls if name == "create_order")
    assert request[5] == {"clOrdId": "CTSPOTOKXO"}


@pytest.mark.asyncio
async def test_minimum_amount_never_rounds_below_platform_amount_or_cost_limit():
    _, session, factory = await _connect()
    client = factory.clients[-1]
    client.markets["BTC/USDT:USDT"]["limits"] = {
        "amount": {"min": "0.1004"},
        "cost": {"min": "0.1004"},
    }
    client.markets["BTC/USDT:USDT"]["precision"]["amount"] = "0.01"

    amount = await session.minimum_amount(Pair.parse("BTC/USDT:USDT"), Decimal("1"), Decimal("0.1004"))

    assert amount >= Decimal("0.1004")
    assert amount * Decimal("1") >= Decimal("0.1004")


@pytest.mark.asyncio
async def test_minimum_amount_retries_after_precision_rounds_first_candidate_to_zero(monkeypatch):
    from cryptotrader.venues.ccxt_base import VenueOperationError

    _, session, _factory = await _connect()
    original = session.normalize_amount
    calls = 0

    async def rounds_first_candidate_to_zero(pair, amount):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise VenueOperationError("okx: amount rounds to zero")
        return await original(pair, amount)

    monkeypatch.setattr(session, "normalize_amount", rounds_first_candidate_to_zero)

    amount = await session.minimum_amount(Pair.parse("BTC/USDT:USDT"), Decimal("100"), Decimal("10"))

    assert calls >= 2
    assert amount * Decimal("100") >= Decimal("10")


@pytest.mark.asyncio
async def test_okx_replacing_long_protection_does_not_cancel_short_leg():
    _, session, factory = await _connect()
    pair = Pair.parse("BTC/USDT:USDT")
    client = factory.clients[-1]
    long_old = await session.replace_protection(
        ProtectionSpec(pair, "long", Decimal("0.02"), Decimal("48000"), Decimal("55000"))
    )
    short = await session.replace_protection(
        ProtectionSpec(pair, "short", Decimal("0.02"), Decimal("56000"), Decimal("45000"))
    )

    replacement = await session.replace_protection(
        ProtectionSpec(pair, "long", Decimal("0.02"), Decimal("47000"), Decimal("57000"))
    )

    active_ids = {item.protection_ids[0] for item in (await session.list_open_state(pair)).protections}
    assert long_old.protection_ids[0] not in active_ids
    assert short.protection_ids[0] in active_ids
    assert replacement.protection_ids[0] in active_ids
    cancelled = {
        row["algoId"] for name, payload in client.calls if name == "private_post_trade_cancel_algos" for row in payload
    }
    assert short.protection_ids[0] not in cancelled


@pytest.mark.asyncio
async def test_okx_active_protection_never_fabricates_trigger_event():
    _, session, factory = await _connect()
    pair = Pair.parse("BTC/USDT:USDT")
    await session.replace_protection(ProtectionSpec(pair, "long", Decimal("0.02"), Decimal("48000"), Decimal("55000")))
    factory.clients[-1]._okx_algos[0]["state"] = "triggered"

    protection = (await session.list_open_state(pair)).protections[0]

    assert protection.active is True
    assert protection.triggered is False

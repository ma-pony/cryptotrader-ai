"""Regression coverage for OKX native algo OCO through the venue session."""

from decimal import Decimal

import pytest

from cryptotrader.pair import Pair
from cryptotrader.runtime_config.secrets import CredentialPayload
from cryptotrader.venues.models import ProtectionSpec
from tests.factories.runtime_config import connection
from tests.fakes.ccxt_client import FakeCcxtFactory

_CLIENTS = {}


async def _session():
    from cryptotrader.venues.okx import OkxVenueAdapter

    factory = FakeCcxtFactory("okx")
    adapter = OkxVenueAdapter(client_factory=factory)
    session = await adapter.connect(
        connection("okx-demo", "demo", adapter_id="okx", credential_ref="credentials"),
        CredentialPayload(api_key="key", secret="secret", passphrase="passphrase"),  # pragma: allowlist secret
    )
    _CLIENTS[session] = factory.clients[-1]
    return session


@pytest.mark.asyncio
async def test_okx_algo_oco_preserves_contract_size_precision_and_platform_params():
    session = await _session()
    spec = ProtectionSpec(Pair.parse("BTC/USDT:USDT"), "long", Decimal("0.02"), Decimal("48000"), Decimal("55000"))

    await session.replace_protection(spec)

    request = next(payload for name, payload in _CLIENTS[session].calls if name == "private_post_trade_order_algo")
    assert request == {
        "instId": "BTC-USDT-SWAP",
        "tdMode": "isolated",
        "side": "sell",
        "posSide": "long",
        "ordType": "oco",
        "sz": "2",
        "reduceOnly": "true",
        "slTriggerPx": "48000.0",
        "slTriggerPxType": "last",
        "slOrdPx": "-1",
        "tpTriggerPx": "55000.0",
        "tpTriggerPxType": "last",
        "tpOrdPx": "-1",
    }


@pytest.mark.asyncio
async def test_okx_algo_cancel_is_idempotent_after_query_confirmation():
    session = await _session()
    spec = ProtectionSpec(Pair.parse("BTC/USDT:USDT"), "long", Decimal("0.02"), Decimal("48000"), Decimal("55000"))
    protection = await session.replace_protection(spec)

    await session.cancel_protection(protection.protection_ids)
    await session.cancel_protection(protection.protection_ids)

    assert (await session.list_open_state(spec.pair)).protections == ()


@pytest.mark.asyncio
async def test_okx_algo_snaps_base_amount_to_contract_lot_size():
    session = await _session()
    spec = ProtectionSpec(Pair.parse("BTC/USDT:USDT"), "long", Decimal("0.025"), Decimal("48000"), Decimal("55000"))

    protection = await session.replace_protection(spec)

    request = next(payload for name, payload in _CLIENTS[session].calls if name == "private_post_trade_order_algo")
    assert request["sz"] == "2"
    assert protection.amount == Decimal("0.02")


@pytest.mark.asyncio
async def test_okx_algo_rejects_amount_that_rounds_to_zero_before_request():
    from cryptotrader.venues.ccxt_base import VenueOperationError

    session = await _session()
    spec = ProtectionSpec(Pair.parse("BTC/USDT:USDT"), "long", Decimal("0.001"), Decimal("48000"), Decimal("55000"))

    with pytest.raises(VenueOperationError, match="rounds to zero"):
        await session.replace_protection(spec)

    assert not any(name == "private_post_trade_order_algo" for name, _ in _CLIENTS[session].calls)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        {"code": "50001", "msg": "service unavailable", "data": []},
        {"code": "0", "data": [{"algoId": "", "sCode": "51000", "sMsg": "param error"}]},
    ],
)
async def test_okx_algo_business_rejections_are_normalized(response):
    from cryptotrader.venues.ccxt_base import VenueOperationError

    session = await _session()

    async def reject(_params):
        return response

    _CLIENTS[session].private_post_trade_order_algo = reject
    spec = ProtectionSpec(Pair.parse("BTC/USDT:USDT"), "long", Decimal("0.02"), Decimal("48000"), Decimal("55000"))

    with pytest.raises(VenueOperationError, match=r"protection .*rejected"):
        await session.replace_protection(spec)


@pytest.mark.asyncio
async def test_okx_algo_unknown_pair_fails_before_native_request():
    from cryptotrader.venues.ccxt_base import VenueOperationError

    session = await _session()
    spec = ProtectionSpec(Pair.parse("DOGE/USDT:USDT"), "long", Decimal("10"), Decimal("0.1"), Decimal("0.2"))

    with pytest.raises(VenueOperationError, match="unknown market"):
        await session.replace_protection(spec)


@pytest.mark.asyncio
@pytest.mark.parametrize("contract_size", [None, "0", "invalid"])
async def test_okx_swap_fails_closed_for_invalid_contract_size(contract_size):
    from cryptotrader.venues.ccxt_base import VenueOperationError

    session = await _session()
    _CLIENTS[session].markets["BTC/USDT:USDT"]["contractSize"] = contract_size
    spec = ProtectionSpec(Pair.parse("BTC/USDT:USDT"), "long", Decimal("0.02"), Decimal("48000"), Decimal("55000"))

    with pytest.raises(VenueOperationError, match="contract size"):
        await session.replace_protection(spec)


@pytest.mark.asyncio
async def test_okx_cancel_swallows_official_already_gone_code():
    session = await _session()
    spec = ProtectionSpec(Pair.parse("BTC/USDT:USDT"), "long", Decimal("0.02"), Decimal("48000"), Decimal("55000"))
    protection = await session.replace_protection(spec)

    async def already_gone(_params):
        raise RuntimeError("OKX 51400: Algo order does not exist")

    _CLIENTS[session].private_post_trade_cancel_algos = already_gone
    await session.cancel_protection(protection.protection_ids)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        {"code": "50001", "msg": "system error", "data": []},
        {"code": "0", "data": [{"algoId": "algo-1", "sCode": "51000", "sMsg": "rejected"}]},
    ],
)
async def test_okx_cancel_business_rejections_are_normalized(response):
    from cryptotrader.venues.ccxt_base import VenueOperationError

    session = await _session()
    spec = ProtectionSpec(Pair.parse("BTC/USDT:USDT"), "long", Decimal("0.02"), Decimal("48000"), Decimal("55000"))
    protection = await session.replace_protection(spec)

    async def reject(_params):
        return response

    _CLIENTS[session].private_post_trade_cancel_algos = reject
    with pytest.raises(VenueOperationError, match=r"cancel OKX protection .*rejected"):
        await session.cancel_protection(protection.protection_ids)


@pytest.mark.asyncio
async def test_okx_pending_algo_query_rejection_is_normalized():
    from cryptotrader.venues.ccxt_base import VenueOperationError

    session = await _session()

    async def reject(_params):
        return {"code": "50001", "msg": "down", "data": []}

    _CLIENTS[session].private_get_trade_orders_algo_pending = reject
    with pytest.raises(VenueOperationError, match="protection query rejected"):
        await session.list_open_state(Pair.parse("BTC/USDT:USDT"))

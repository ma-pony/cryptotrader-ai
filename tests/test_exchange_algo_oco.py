"""Phase 2A — LiveExchange OKX algo OCO API methods (place / cancel / list).

Pure unit tests with mocked ccxt response; no sandbox network call.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cryptotrader.execution.exchange import LiveExchange
from cryptotrader.models import Order


def _make_okx_exchange() -> LiveExchange:
    with patch("ccxt.async_support.okx") as mock_cls:
        mock_inst = MagicMock()
        mock_inst.load_markets = AsyncMock()
        # Markets loaded with DOGE perp swap so _to_okx_inst_id works.
        mock_inst.markets = {
            "DOGE/USDT:USDT": {"id": "DOGE-USDT-SWAP"},
            "BTC/USDT:USDT": {"id": "BTC-USDT-SWAP"},
        }
        # Identity precision helpers — real ccxt rounds to lotSz/tickSz.
        mock_inst.amount_to_precision = MagicMock(side_effect=lambda pair, a: str(a))
        mock_inst.price_to_precision = MagicMock(side_effect=lambda pair, x: str(x))
        mock_cls.return_value = mock_inst
        return LiveExchange(
            "okx",
            "k",
            "s",
            sandbox=True,
            passphrase="p",
            margin_mode="isolated",
        )


def test_live_protection_capability_is_explicit_per_implemented_adapter():
    assert _make_okx_exchange().supports_protection_orders() is True
    with patch("ccxt.async_support.binance") as mock_cls:
        mock_cls.return_value = MagicMock()
        binance = LiveExchange("binance", "k", "s", sandbox=True)
    assert binance.supports_protection_orders() is False


@pytest.mark.asyncio
async def test_reduce_only_order_reaches_exchange_params():
    ex = _make_okx_exchange()
    ex._markets_loaded = True
    ex.get_balance = AsyncMock(return_value={"USDT": 10_000.0})
    ex._derive_pos_side = AsyncMock(return_value="long")
    ex._exchange.create_order = AsyncMock(return_value={"id": "close-1", "status": "closed"})

    await ex.place_order(
        Order(
            pair="BTC/USDT:USDT",
            side="sell",
            amount=0.1,
            price=100.0,
            reduce_only=True,
        )
    )

    params = ex._exchange.create_order.await_args.args[-1]
    assert params["reduceOnly"] is True


# ── place_algo_oco ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_place_algo_oco_success_returns_algo_id():
    ex = _make_okx_exchange()
    ex._markets_loaded = True
    ex._exchange.private_post_trade_order_algo = AsyncMock(
        return_value={
            "code": "0",
            "data": [{"algoId": "abc123", "sCode": "0", "sMsg": ""}],
        }
    )

    algo_id = await ex.place_algo_oco(
        "DOGE/USDT:USDT",
        side="sell",
        amount=10.0,
        sl_trigger_px=0.108,
        tp_trigger_px=0.120,
        pos_side="long",
    )

    assert algo_id == "abc123"
    ex._exchange.private_post_trade_order_algo.assert_awaited_once()
    params = ex._exchange.private_post_trade_order_algo.await_args.args[0]
    assert params["instId"] == "DOGE-USDT-SWAP"
    assert params["tdMode"] == "isolated"
    assert params["side"] == "sell"
    assert params["posSide"] == "long"
    assert params["ordType"] == "oco"
    assert params["sz"] == "10.0"
    assert params["reduceOnly"] == "true"
    assert params["slTriggerPx"] == "0.108"
    assert params["slOrdPx"] == "-1"
    assert params["tpTriggerPx"] == "0.12"
    assert params["tpOrdPx"] == "-1"


@pytest.mark.asyncio
async def test_place_algo_oco_snaps_sz_to_lot_size():
    """Regression: OKX rejects code 51121 when sz is not a multiple of lotSz.

    Verifies place_algo_oco routes amount through ccxt's
    ``amount_to_precision`` before constructing ``sz`` — matches the
    snapping that ``place_order`` already does for the entry leg.
    """
    ex = _make_okx_exchange()
    ex._markets_loaded = True
    # DOGE-USDT-SWAP has lotSz=1 — caller passes a fractional fill (the
    # bug surfaced 2026-05-14 with sz=4562.824176).
    ex._exchange.amount_to_precision = MagicMock(return_value="4562")
    ex._exchange.price_to_precision = MagicMock(side_effect=["0.108", "0.12"])
    ex._exchange.private_post_trade_order_algo = AsyncMock(
        return_value={"code": "0", "data": [{"algoId": "ok", "sCode": "0", "sMsg": ""}]}
    )

    await ex.place_algo_oco(
        "DOGE/USDT:USDT",
        side="buy",
        amount=4562.824176,
        sl_trigger_px=0.108,
        tp_trigger_px=0.120,
        pos_side="short",
    )

    ex._exchange.amount_to_precision.assert_called_once_with("DOGE/USDT:USDT", 4562.824176)
    params = ex._exchange.private_post_trade_order_algo.await_args.args[0]
    assert params["sz"] == "4562"
    assert params["slTriggerPx"] == "0.108"
    assert params["tpTriggerPx"] == "0.12"


@pytest.mark.asyncio
async def test_place_algo_oco_converts_base_amount_to_contracts():
    ex = _make_okx_exchange()
    ex._markets_loaded = True
    ex._exchange.markets["BTC/USDT:USDT"]["contractSize"] = 0.01
    ex._exchange.private_post_trade_order_algo = AsyncMock(
        return_value={"code": "0", "data": [{"algoId": "ok", "sCode": "0", "sMsg": ""}]}
    )

    await ex.place_algo_oco(
        "BTC/USDT:USDT",
        side="sell",
        amount=0.1,
        sl_trigger_px=90_000,
        tp_trigger_px=120_000,
        pos_side="long",
    )

    ex._exchange.amount_to_precision.assert_called_once_with("BTC/USDT:USDT", 10.0)
    params = ex._exchange.private_post_trade_order_algo.await_args.args[0]
    assert params["sz"] == "10.0"


@pytest.mark.asyncio
async def test_place_algo_oco_rejects_when_rounded_to_zero():
    """If snapping rounds sz below lotSz to 0, raise ValueError early."""
    ex = _make_okx_exchange()
    ex._markets_loaded = True
    ex._exchange.amount_to_precision = MagicMock(return_value="0")
    ex._exchange.price_to_precision = MagicMock(side_effect=["0.108", "0.12"])
    ex._exchange.private_post_trade_order_algo = AsyncMock()

    with pytest.raises(ValueError, match="rounds to 0"):
        await ex.place_algo_oco(
            "DOGE/USDT:USDT",
            side="buy",
            amount=0.1,
            sl_trigger_px=0.108,
            tp_trigger_px=0.120,
            pos_side="short",
        )
    ex._exchange.private_post_trade_order_algo.assert_not_called()


@pytest.mark.asyncio
async def test_place_algo_oco_top_level_error_raises():
    ex = _make_okx_exchange()
    ex._markets_loaded = True
    ex._exchange.private_post_trade_order_algo = AsyncMock(
        return_value={"code": "50001", "msg": "service unavailable", "data": []}
    )

    with pytest.raises(RuntimeError, match="OKX algo OCO rejected"):
        await ex.place_algo_oco(
            "DOGE/USDT:USDT",
            side="sell",
            amount=10.0,
            sl_trigger_px=0.108,
            tp_trigger_px=0.120,
            pos_side="long",
        )


@pytest.mark.asyncio
async def test_place_algo_oco_leg_level_error_raises():
    """OKX returns code=0 but per-leg sCode != 0 (e.g. 51000 param error)."""
    ex = _make_okx_exchange()
    ex._markets_loaded = True
    ex._exchange.private_post_trade_order_algo = AsyncMock(
        return_value={
            "code": "0",
            "data": [{"algoId": "", "sCode": "51000", "sMsg": "param error"}],
        }
    )

    with pytest.raises(RuntimeError, match=r"OKX algo OCO leg rejected.*51000"):
        await ex.place_algo_oco(
            "BTC/USDT:USDT",
            side="buy",
            amount=1.0,
            sl_trigger_px=82000,
            tp_trigger_px=78000,
            pos_side="short",
        )


@pytest.mark.asyncio
async def test_place_algo_oco_non_okx_raises_not_implemented():
    with patch("ccxt.async_support.binance") as mock_cls:
        mock_inst = MagicMock()
        mock_cls.return_value = mock_inst
        ex = LiveExchange("binance", "k", "s", sandbox=True)
    with pytest.raises(NotImplementedError, match="OKX-only"):
        await ex.place_algo_oco("BTC/USDT", side="sell", amount=1, sl_trigger_px=1, tp_trigger_px=2, pos_side="long")


@pytest.mark.asyncio
async def test_place_algo_oco_unknown_pair_raises():
    ex = _make_okx_exchange()
    ex._markets_loaded = True
    with pytest.raises(ValueError, match="Cannot resolve OKX instId"):
        await ex.place_algo_oco(
            "UNKNOWN/USDT:USDT",
            side="sell",
            amount=1,
            sl_trigger_px=1,
            tp_trigger_px=2,
            pos_side="long",
        )


# ── cancel_algo ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cancel_algo_success():
    ex = _make_okx_exchange()
    ex._markets_loaded = True
    ex._exchange.private_post_trade_cancel_algos = AsyncMock(
        return_value={"code": "0", "data": [{"algoId": "abc123", "sCode": "0"}]}
    )

    await ex.cancel_algo("abc123", "DOGE/USDT:USDT")

    ex._exchange.private_post_trade_cancel_algos.assert_awaited_once()
    params = ex._exchange.private_post_trade_cancel_algos.await_args.args[0]
    assert params == [{"algoId": "abc123", "instId": "DOGE-USDT-SWAP"}]


@pytest.mark.asyncio
async def test_cancel_algo_already_gone_swallowed():
    """OKX 51400 / 51401 → already triggered / not exist → swallow as success."""
    ex = _make_okx_exchange()
    ex._markets_loaded = True
    ex._exchange.private_post_trade_cancel_algos = AsyncMock(
        side_effect=Exception("OKX 51400: Algo order does not exist")
    )

    # Should not raise
    await ex.cancel_algo("dead123", "DOGE/USDT:USDT")


@pytest.mark.asyncio
async def test_cancel_algo_real_error_propagates():
    ex = _make_okx_exchange()
    ex._markets_loaded = True
    ex._exchange.private_post_trade_cancel_algos = AsyncMock(side_effect=Exception("OKX 50001: System error"))

    with pytest.raises(Exception, match="50001"):
        await ex.cancel_algo("abc123", "DOGE/USDT:USDT")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        {"code": "50001", "msg": "system error", "data": []},
        {
            "code": "0",
            "data": [{"algoId": "abc123", "sCode": "51000", "sMsg": "cancel rejected"}],
        },
    ],
)
async def test_cancel_algo_business_rejection_propagates(response):
    ex = _make_okx_exchange()
    ex._markets_loaded = True
    ex._exchange.private_post_trade_cancel_algos = AsyncMock(return_value=response)

    with pytest.raises(RuntimeError, match=r"cancel.*failed"):
        await ex.cancel_algo("abc123", "DOGE/USDT:USDT")


# ── list_pending_algos ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_pending_algos_returns_list():
    ex = _make_okx_exchange()
    ex._markets_loaded = True
    ex._exchange.private_get_trade_orders_algo_pending = AsyncMock(
        return_value={
            "code": "0",
            "data": [
                {"algoId": "abc", "instId": "DOGE-USDT-SWAP", "ordType": "oco"},
                {"algoId": "def", "instId": "BTC-USDT-SWAP", "ordType": "oco"},
            ],
        }
    )

    algos = await ex.list_pending_algos()

    assert len(algos) == 2
    assert algos[0]["algoId"] == "abc"
    # Verify it was called with the OCO filter
    params = ex._exchange.private_get_trade_orders_algo_pending.await_args.args[0]
    assert params == {"ordType": "oco"}


@pytest.mark.asyncio
async def test_list_pending_algos_filtered_by_pair():
    ex = _make_okx_exchange()
    ex._markets_loaded = True
    ex._exchange.private_get_trade_orders_algo_pending = AsyncMock(return_value={"code": "0", "data": []})

    await ex.list_pending_algos(pair="DOGE/USDT:USDT")

    params = ex._exchange.private_get_trade_orders_algo_pending.await_args.args[0]
    assert params == {"ordType": "oco", "instId": "DOGE-USDT-SWAP"}


@pytest.mark.asyncio
async def test_list_pending_algos_error_raises():
    ex = _make_okx_exchange()
    ex._markets_loaded = True
    ex._exchange.private_get_trade_orders_algo_pending = AsyncMock(return_value={"code": "50001", "msg": "down"})

    with pytest.raises(RuntimeError, match="OKX list algos failed"):
        await ex.list_pending_algos()

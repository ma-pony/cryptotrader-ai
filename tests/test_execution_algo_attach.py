"""Phase 2B — _attach_okx_algo_protect: post-fill OKX algo OCO integration.

Covers:
  - long/short entry + valid SL/TP → places algo, returns ids
  - close action → cancels stale algos, places nothing
  - hold action → no-op (defensive; place_order wouldn't call this anyway)
  - missing SL or TP → no algo
  - paper exchange (no place_algo_oco method) → no-op
  - algo placement failure → soft fail, returns empty dict
  - list_pending_algos failure → continues to placement
  - cancel_algo failure on stale id → continues
  - contract_size conversion (filled base units → algo sz contracts)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from cryptotrader.nodes.execution import _attach_okx_algo_protect


def _make_okx_mock_exchange(
    *,
    contract_size: float = 1.0,
    total_position_amount: float | None = None,
) -> MagicMock:
    """Mock that quacks like LiveExchange for OKX (has place_algo_oco etc.).

    ``total_position_amount`` populates ``get_positions()[pair]['amount']`` to
    simulate the current total position size returned by OKX. Set to
    ``None`` (default) to make get_positions return no entry for the pair,
    which exercises the fallback-to-filled_amount path used when there's no
    pre-existing position (fresh open).
    """
    ex = MagicMock()
    ex.place_algo_oco = AsyncMock(return_value="okx-algo-xyz")
    ex.cancel_algo = AsyncMock(return_value=None)
    ex.list_pending_algos = AsyncMock(return_value=[])
    positions = {}
    if total_position_amount is not None:
        positions["DOGE/USDT:USDT"] = {"amount": total_position_amount, "side": "short"}
    ex.get_positions = AsyncMock(return_value=positions)
    # Markets dict (accessed via exchange._exchange.markets) for contract_size lookup
    inner = MagicMock()
    inner.markets = {"DOGE/USDT:USDT": {"contractSize": str(contract_size)}}
    ex._exchange = inner
    return ex


def _make_paper_exchange() -> MagicMock:
    """No place_algo_oco method — duck-type fails the hasattr() check."""
    return MagicMock(spec=["create_order", "fetch_balance"])


# ── Entry path: long ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_long_entry_with_valid_sl_tp_places_algo():
    ex = _make_okx_mock_exchange(contract_size=1000.0)
    verdict = {"action": "long", "stop_loss": 0.108, "take_profit": 0.120}

    result = await _attach_okx_algo_protect(
        exchange=ex,
        pair="DOGE/USDT:USDT",
        verdict=verdict,
        filled_amount=10_000.0,  # base units (DOGE coins)
    )

    assert result == {
        "algo_id": "okx-algo-xyz",
        "stop_loss_price": 0.108,
        "take_profit_price": 0.120,
    }
    ex.place_algo_oco.assert_awaited_once()
    call = ex.place_algo_oco.await_args
    assert call.args[0] == "DOGE/USDT:USDT"
    assert call.kwargs["side"] == "sell"  # closing a long
    assert call.kwargs["pos_side"] == "long"
    assert call.kwargs["amount"] == pytest.approx(10.0)  # 10000 / 1000 contracts
    assert call.kwargs["sl_trigger_px"] == 0.108
    assert call.kwargs["tp_trigger_px"] == 0.120


@pytest.mark.asyncio
async def test_short_entry_with_valid_sl_tp_places_algo():
    ex = _make_okx_mock_exchange(contract_size=1.0)
    verdict = {"action": "short", "stop_loss": 95.0, "take_profit": 88.0}

    result = await _attach_okx_algo_protect(
        exchange=ex,
        pair="DOGE/USDT:USDT",
        verdict=verdict,
        filled_amount=5.0,
    )

    assert result["algo_id"] == "okx-algo-xyz"
    call = ex.place_algo_oco.await_args
    assert call.kwargs["side"] == "buy"  # closing a short
    assert call.kwargs["pos_side"] == "short"
    assert call.kwargs["amount"] == 5.0  # no conversion for contract_size=1


# ── Cancel-on-flip ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_entry_cancels_existing_algos_before_placing_new():
    ex = _make_okx_mock_exchange()
    ex.list_pending_algos = AsyncMock(
        return_value=[
            {"algoId": "old-algo-1", "instId": "DOGE-USDT-SWAP"},
            {"algoId": "old-algo-2", "instId": "DOGE-USDT-SWAP"},
        ]
    )

    await _attach_okx_algo_protect(
        exchange=ex,
        pair="DOGE/USDT:USDT",
        verdict={"action": "long", "stop_loss": 0.108, "take_profit": 0.120},
        filled_amount=10.0,
    )

    assert ex.cancel_algo.await_count == 2
    cancelled_ids = [call.args[0] for call in ex.cancel_algo.await_args_list]
    assert cancelled_ids == ["old-algo-1", "old-algo-2"]
    # And new algo still placed
    ex.place_algo_oco.assert_awaited_once()


@pytest.mark.asyncio
async def test_close_action_cancels_pending_no_new_algo():
    ex = _make_okx_mock_exchange()
    ex.list_pending_algos = AsyncMock(return_value=[{"algoId": "stale-protect", "instId": "DOGE-USDT-SWAP"}])

    result = await _attach_okx_algo_protect(
        exchange=ex,
        pair="DOGE/USDT:USDT",
        verdict={"action": "close"},
        filled_amount=5.0,
    )

    assert result == {}
    ex.cancel_algo.assert_awaited_once_with("stale-protect", "DOGE/USDT:USDT")
    ex.place_algo_oco.assert_not_awaited()


# ── No-op paths ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_paper_exchange_returns_empty_dict():
    ex = _make_paper_exchange()
    result = await _attach_okx_algo_protect(
        exchange=ex,
        pair="DOGE/USDT:USDT",
        verdict={"action": "long", "stop_loss": 0.108, "take_profit": 0.120},
        filled_amount=10.0,
    )
    assert result == {}


@pytest.mark.asyncio
async def test_missing_sl_returns_empty_dict_no_algo_placed():
    ex = _make_okx_mock_exchange()
    result = await _attach_okx_algo_protect(
        exchange=ex,
        pair="DOGE/USDT:USDT",
        verdict={"action": "long", "take_profit": 0.120},  # stop_loss missing
        filled_amount=10.0,
    )
    assert result == {}
    ex.place_algo_oco.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_tp_returns_empty_dict():
    ex = _make_okx_mock_exchange()
    result = await _attach_okx_algo_protect(
        exchange=ex,
        pair="DOGE/USDT:USDT",
        verdict={"action": "long", "stop_loss": 0.108},
        filled_amount=10.0,
    )
    assert result == {}
    ex.place_algo_oco.assert_not_awaited()


@pytest.mark.asyncio
async def test_hold_action_returns_empty_dict():
    """Defensive: place_order would not call this for hold, but if invoked,
    it must do nothing (no cancel of unrelated algos)."""
    ex = _make_okx_mock_exchange()
    ex.list_pending_algos = AsyncMock(return_value=[])
    result = await _attach_okx_algo_protect(
        exchange=ex,
        pair="DOGE/USDT:USDT",
        verdict={"action": "hold"},
        filled_amount=0.0,
    )
    assert result == {}
    ex.place_algo_oco.assert_not_awaited()


# ── Soft-fail behaviour ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_algo_placement_failure_soft_fails():
    """OKX rejection during place_algo_oco does not raise; returns empty dict."""
    ex = _make_okx_mock_exchange()
    ex.place_algo_oco = AsyncMock(side_effect=RuntimeError("OKX 50001 system error"))

    result = await _attach_okx_algo_protect(
        exchange=ex,
        pair="DOGE/USDT:USDT",
        verdict={"action": "long", "stop_loss": 0.108, "take_profit": 0.120},
        filled_amount=10.0,
    )
    assert result == {}


@pytest.mark.asyncio
async def test_list_pending_algos_failure_continues_to_placement():
    """If listing fails (transient), we skip cancel and proceed to place."""
    ex = _make_okx_mock_exchange()
    ex.list_pending_algos = AsyncMock(side_effect=Exception("transient network"))

    result = await _attach_okx_algo_protect(
        exchange=ex,
        pair="DOGE/USDT:USDT",
        verdict={"action": "long", "stop_loss": 0.108, "take_profit": 0.120},
        filled_amount=10.0,
    )
    assert result["algo_id"] == "okx-algo-xyz"
    ex.place_algo_oco.assert_awaited_once()


@pytest.mark.asyncio
async def test_cancel_algo_failure_continues_to_placement():
    """One stale-algo cancel failure does not block placement of the new one."""
    ex = _make_okx_mock_exchange()
    ex.list_pending_algos = AsyncMock(return_value=[{"algoId": "stuck-id", "instId": "DOGE-USDT-SWAP"}])
    ex.cancel_algo = AsyncMock(side_effect=Exception("OKX 50001 system error"))

    result = await _attach_okx_algo_protect(
        exchange=ex,
        pair="DOGE/USDT:USDT",
        verdict={"action": "long", "stop_loss": 0.108, "take_profit": 0.120},
        filled_amount=10.0,
    )
    assert result["algo_id"] == "okx-algo-xyz"
    ex.place_algo_oco.assert_awaited_once()


@pytest.mark.asyncio
async def test_add_to_position_sizes_oco_to_total_not_increment():
    """Regression for OCO sz bug (2026-05-17): on add-to-position the OCO must
    protect the *total* current position, not just the incremental fill.

    Before the fix: AI add of 50 new contracts to an existing 85 → OCO sz=50
    (only the new add protected; 85 contracts naked). Observed cumulatively
    across SOL/DOGE for many cycles, with 99%+ of position unguarded.
    """
    # Existing 85,000 DOGE total (short). Each contract = 1000 DOGE → 85
    # contracts. New fill adds another 50,000 DOGE (50 contracts) → total 135.
    ex = _make_okx_mock_exchange(contract_size=1000.0, total_position_amount=-135_000.0)

    result = await _attach_okx_algo_protect(
        exchange=ex,
        pair="DOGE/USDT:USDT",
        verdict={"action": "short", "stop_loss": 0.115, "take_profit": 0.105},
        filled_amount=50_000.0,  # increment ONLY — bug used this directly
    )

    assert result["algo_id"] == "okx-algo-xyz"
    call = ex.place_algo_oco.await_args
    # sz must be total contracts (135), not increment-only (50).
    assert call.kwargs["amount"] == pytest.approx(135.0)
    assert call.kwargs["side"] == "buy"  # closing a short
    assert call.kwargs["pos_side"] == "short"
    ex.get_positions.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_positions_failure_falls_back_to_filled_amount():
    """When OKX position read fails, OCO sz falls back to filled_amount.

    Protects the new fill at minimum. Soft-fail mirroring `_attach_okx_algo_protect`
    overall design (algo failure never rolls back the entry leg).
    """
    ex = _make_okx_mock_exchange(contract_size=1000.0)
    ex.get_positions = AsyncMock(side_effect=Exception("transient OKX read err"))

    result = await _attach_okx_algo_protect(
        exchange=ex,
        pair="DOGE/USDT:USDT",
        verdict={"action": "short", "stop_loss": 0.115, "take_profit": 0.105},
        filled_amount=50_000.0,
    )

    assert result["algo_id"] == "okx-algo-xyz"
    # Fallback uses filled_amount when get_positions errored.
    assert ex.place_algo_oco.await_args.kwargs["amount"] == pytest.approx(50.0)


@pytest.mark.asyncio
async def test_non_okx_live_exchange_not_implemented_silent_skip():
    """LiveExchange on a non-OKX venue: list_pending_algos raises NotImplementedError.
    We should silently skip without WARN spam."""
    ex = MagicMock()
    ex.place_algo_oco = AsyncMock()
    ex.cancel_algo = AsyncMock()
    ex.list_pending_algos = AsyncMock(side_effect=NotImplementedError("OKX-only"))

    result = await _attach_okx_algo_protect(
        exchange=ex,
        pair="BTC/USDT:USDT",
        verdict={"action": "long", "stop_loss": 80000, "take_profit": 82000},
        filled_amount=1.0,
    )
    assert result == {}
    ex.place_algo_oco.assert_not_awaited()

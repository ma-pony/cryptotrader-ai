"""PaperExchange concurrency safety tests (task 3.2).

Coverage scenarios:
- _lock attribute exists and has the correct type
- Concurrent place_order() writes yield consistent final balance
- get_balance() uses Lock for snapshot read
- get_positions() uses Lock for snapshot read
- cancel_order() / get_order() / fetch_open_orders() are Lock-protected reads
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from cryptotrader.models import Order

# -- Factory helpers --


def _make_order(side: str = "buy", amount: float = 0.1, price: float = 10000.0) -> Order:
    return Order(pair="BTC/USDT", side=side, amount=amount, price=price)


def _default_cfg() -> Any:
    """Minimal config mock."""
    cfg = MagicMock()
    cfg.backtest.initial_capital = 10000.0
    cfg.backtest.slippage_base = 0.001
    cfg.backtest.fee_bps = 10.0
    return cfg


def _make_exchange(**kwargs) -> Any:
    with patch("cryptotrader.config.load_config", return_value=_default_cfg()):
        from cryptotrader.execution.simulator import PaperExchange

        return PaperExchange(**kwargs)


# -- Structural tests --


def test_lock_exists():
    """PaperExchange.__init__ must create an asyncio.Lock instance."""
    ex = _make_exchange(initial_balances={"USDT": 10000.0})
    assert hasattr(ex, "_lock"), "_lock attribute missing"
    assert isinstance(ex._lock, asyncio.Lock), "_lock must be asyncio.Lock"


# -- Concurrent write consistency (place_order already protected) --


@pytest.mark.asyncio
async def test_concurrent_place_orders_balance_consistency():
    """Concurrent place_order() calls must produce a consistent final balance (no race)."""
    ex = _make_exchange(initial_balances={"USDT": 10000.0})
    order = _make_order(side="buy", amount=0.01, price=1000.0)

    tasks = [asyncio.create_task(ex.place_order(order)) for _ in range(10)]
    results = await asyncio.gather(*tasks)

    filled = [r for r in results if r["status"] == "filled"]

    balance = (await ex.get_balance()).get("USDT", 0.0)
    assert balance >= 0, f"Balance went negative ({balance}), race condition detected"
    assert len(results) == 10, "Some orders were lost"

    spent = sum(r["amount"] * r["price"] * (1 + ex._fee_bps / 10000) for r in filled)
    assert abs(balance + spent - 10000.0) < 1.0, f"Balance not conserved: balance={balance}, spent={spent}"


@pytest.mark.asyncio
async def test_concurrent_buy_sell_no_negative_balance():
    """After concurrent buys and sells, USDT and asset balances must stay non-negative."""
    ex = _make_exchange(initial_balances={"USDT": 10000.0, "BTC": 1.0})
    ex._cost_basis["BTC"] = {"total_cost": 10000.0, "total_amount": 1.0}

    buy_order = _make_order(side="buy", amount=0.01, price=1000.0)
    sell_order = _make_order(side="sell", amount=0.05, price=1000.0)

    tasks = [asyncio.create_task(ex.place_order(buy_order)) for _ in range(5)] + [
        asyncio.create_task(ex.place_order(sell_order)) for _ in range(5)
    ]
    await asyncio.gather(*tasks)

    balance = await ex.get_balance()
    usdt = balance.get("USDT", 0.0)
    btc = balance.get("BTC", 0.0)
    assert usdt >= -1e-9, f"USDT balance went negative: {usdt}"
    assert btc >= -1e-9, f"BTC balance went negative: {btc}"


# -- Read methods must wait for Lock (snapshot read verification) --


async def _assert_method_waits_for_lock(ex: Any, coro_factory: Any, lock_hold_seconds: float = 0.15) -> None:
    """Assert that the given coroutine factory blocks while the Lock is held.

    Strategy:
    1. Acquire the Lock and hold it for lock_hold_seconds.
    2. Launch the target method after lock_hold_seconds * 0.25 seconds.
    3. Record when the Lock is released and when the method completes.
    4. If method_end >= lock_release, the method waited for the Lock.
    """
    lock_release_time: list[float] = []
    method_end_time: list[float] = []

    async def hold_lock() -> None:
        async with ex._lock:
            await asyncio.sleep(lock_hold_seconds)
            lock_release_time.append(time.monotonic())

    async def call_target() -> None:
        await asyncio.sleep(lock_hold_seconds * 0.25)  # wait until lock is held
        await coro_factory()
        method_end_time.append(time.monotonic())

    await asyncio.gather(hold_lock(), call_target())

    assert lock_release_time, "hold_lock did not run"
    assert method_end_time, "target method did not run"

    release_t = lock_release_time[0]
    end_t = method_end_time[0]

    # Method must complete after the Lock is released (5 ms scheduling tolerance)
    assert end_t >= release_t - 0.005, (
        f"Method finished while Lock was still held "
        f"(method_end={end_t:.4f} < lock_release={release_t:.4f}). "
        f"The method is NOT protected by async with self._lock."
    )


@pytest.mark.asyncio
async def test_get_balance_waits_for_lock():
    """get_balance() must acquire the Lock for a consistent snapshot read."""
    ex = _make_exchange(initial_balances={"USDT": 10000.0})
    await _assert_method_waits_for_lock(ex, lambda: ex.get_balance())


@pytest.mark.asyncio
async def test_get_positions_waits_for_lock():
    """get_positions() must acquire the Lock for a consistent snapshot read."""
    ex = _make_exchange(initial_balances={"USDT": 10000.0, "BTC": 0.1})
    ex._cost_basis["BTC"] = {"total_cost": 10000.0, "total_amount": 0.1}
    await _assert_method_waits_for_lock(ex, lambda: ex.get_positions())


@pytest.mark.asyncio
async def test_cancel_order_waits_for_lock():
    """cancel_order() must acquire the Lock before mutating order state."""
    ex = _make_exchange(initial_balances={"USDT": 10000.0})
    order = _make_order(side="buy", amount=0.01, price=100.0)
    result = await ex.place_order(order)
    order_id = result["id"]
    # Reset to filled so cancel is allowed again
    ex._orders[order_id]["status"] = "filled"
    await _assert_method_waits_for_lock(ex, lambda: ex.cancel_order(order_id))


@pytest.mark.asyncio
async def test_get_order_waits_for_lock():
    """get_order() must acquire the Lock before reading order data."""
    ex = _make_exchange(initial_balances={"USDT": 10000.0})
    order = _make_order(side="buy", amount=0.01, price=100.0)
    result = await ex.place_order(order)
    order_id = result["id"]
    await _assert_method_waits_for_lock(ex, lambda: ex.get_order(order_id))


@pytest.mark.asyncio
async def test_fetch_open_orders_waits_for_lock():
    """fetch_open_orders() must acquire the Lock before iterating orders."""
    ex = _make_exchange(initial_balances={"USDT": 10000.0})
    await _assert_method_waits_for_lock(ex, lambda: ex.fetch_open_orders())


# -- Functional correctness --


@pytest.mark.asyncio
async def test_get_balance_returns_correct_snapshot():
    """get_balance() snapshot read returns the correct values."""
    ex = _make_exchange(initial_balances={"USDT": 5000.0})
    balance = await ex.get_balance()
    assert balance.get("USDT") == pytest.approx(5000.0)


@pytest.mark.asyncio
async def test_get_balance_excludes_zero_values():
    """get_balance() omits assets with zero balance."""
    ex = _make_exchange(initial_balances={"USDT": 5000.0, "BTC": 0.0})
    balance = await ex.get_balance()
    assert "BTC" not in balance, "Zero-balance asset must not appear in result"


@pytest.mark.asyncio
async def test_cancel_order_sets_cancelled_status():
    """cancel_order() sets order status to cancelled."""
    ex = _make_exchange(initial_balances={"USDT": 10000.0})
    order = _make_order(side="buy", amount=0.01, price=100.0)
    placed = await ex.place_order(order)
    order_id = placed["id"]

    cancelled = await ex.cancel_order(order_id)
    assert cancelled["status"] == "cancelled"


@pytest.mark.asyncio
async def test_get_order_raises_for_unknown_id():
    """get_order() raises ValueError for unknown order_id."""
    ex = _make_exchange(initial_balances={"USDT": 10000.0})
    with pytest.raises(ValueError, match="not found"):
        await ex.get_order("nonexistent-id")


@pytest.mark.asyncio
async def test_fetch_open_orders_returns_only_open():
    """fetch_open_orders() returns only orders with status=open."""
    ex = _make_exchange(initial_balances={"USDT": 10000.0})
    order = _make_order(side="buy", amount=0.01, price=100.0)
    await ex.place_order(order)

    # place_order fills immediately (status=filled), so open list is empty
    open_orders = await ex.fetch_open_orders()
    assert isinstance(open_orders, list)
    assert all(o["status"] == "open" for o in open_orders)


# -- Deadlock prevention --


@pytest.mark.asyncio
async def test_sequential_calls_no_deadlock():
    """Sequential calls to all public methods must not deadlock."""
    ex = _make_exchange(initial_balances={"USDT": 10000.0})
    order = _make_order(side="buy", amount=0.01, price=100.0)

    for _ in range(3):
        r = await ex.place_order(order)
        assert r["status"] in {"filled", "failed"}

    balance = await ex.get_balance()
    assert isinstance(balance, dict)

    positions = await ex.get_positions()
    assert isinstance(positions, dict)

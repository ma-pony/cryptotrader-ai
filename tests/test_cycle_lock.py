"""Tests for the per-pair cycle lock used to prevent concurrent runs.

Production observation 2026-05-02: a manual ``arena run`` started while the
launchd scheduler was processing a freshly-restarted cycle produced two
concurrent ETH/USDT close decisions 426ms apart. The lock guarantees that
only one cycle for a given pair can be in flight at a time.
"""

from __future__ import annotations

import asyncio

import pytest

from cryptotrader.cycle_lock import cycle_lock
from cryptotrader.risk.state import RedisStateManager


@pytest.fixture
def rsm():
    # No redis_url -> memory fallback (RedisStateManager._mem). Same NX semantics.
    return RedisStateManager(None)


@pytest.mark.asyncio
async def test_acquire_when_free(rsm):
    async with cycle_lock(rsm, "BTC/USDT") as acquired:
        assert acquired is True


@pytest.mark.asyncio
async def test_second_holder_is_blocked_while_first_is_active(rsm):
    async with cycle_lock(rsm, "BTC/USDT") as outer:
        assert outer is True
        async with cycle_lock(rsm, "BTC/USDT") as inner:
            assert inner is False


@pytest.mark.asyncio
async def test_lock_released_on_normal_exit(rsm):
    async with cycle_lock(rsm, "BTC/USDT") as a1:
        assert a1 is True
    # Lock cleared — fresh acquire succeeds.
    async with cycle_lock(rsm, "BTC/USDT") as a2:
        assert a2 is True


@pytest.mark.asyncio
async def test_lock_released_even_when_block_raises(rsm):
    async def _holder_that_raises():
        async with cycle_lock(rsm, "BTC/USDT") as acquired:
            assert acquired is True
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await _holder_that_raises()
    # The exception propagated, but the lock was still released.
    async with cycle_lock(rsm, "BTC/USDT") as a2:
        assert a2 is True


@pytest.mark.asyncio
async def test_different_pairs_do_not_block_each_other(rsm):
    async with cycle_lock(rsm, "BTC/USDT") as a_btc:
        assert a_btc is True
        async with cycle_lock(rsm, "ETH/USDT") as a_eth:
            assert a_eth is True


@pytest.mark.asyncio
async def test_contended_concurrent_acquire_only_one_wins(rsm):
    """Simulate the production race: two coroutines try to enter the same pair."""
    pair = "BTC/USDT"
    held_at_same_time = 0
    max_concurrent = 0
    lock = asyncio.Lock()  # protects the counter, NOT the cycle_lock under test

    async def worker():
        nonlocal held_at_same_time, max_concurrent
        async with cycle_lock(rsm, pair) as acquired:
            if not acquired:
                return
            async with lock:
                held_at_same_time += 1
                max_concurrent = max(max_concurrent, held_at_same_time)
            await asyncio.sleep(0.01)
            async with lock:
                held_at_same_time -= 1

    await asyncio.gather(*(worker() for _ in range(10)))
    assert max_concurrent == 1, f"two holders ran simultaneously: max_concurrent={max_concurrent}"


@pytest.mark.asyncio
async def test_release_is_owner_checked(rsm):
    """If holder A's TTL expires and B acquires, A's release must NOT wipe B's key."""
    # Manually exercise the underlying primitives to simulate TTL expiry.
    key = "cycle_lock:BTC/USDT"
    a_owner = "owner-a"
    b_owner = "owner-b"

    assert await rsm.try_acquire_lock(key, a_owner, ttl=300) is True
    # Simulate A's TTL expiry by deleting the key out from under A.
    await rsm._redis.delete(key) if rsm._redis else rsm._mem.delete(key)
    # B grabs it.
    assert await rsm.try_acquire_lock(key, b_owner, ttl=300) is True
    # A's late release must be a no-op.
    assert await rsm.release_lock(key, a_owner) is False
    # B's key still present.
    assert await rsm.get(key) == b_owner

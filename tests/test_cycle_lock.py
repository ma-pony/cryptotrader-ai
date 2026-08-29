"""Redis-only execution lease primitives."""

from __future__ import annotations

import asyncio

import pytest

from cryptotrader.cycle_lock import cycle_lock
from cryptotrader.risk.state import RedisStateManager


class _Redis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    async def set(self, key, value, *, nx, ex):
        del ex
        if nx and key in self.values:
            return False
        self.values[key] = value
        return True

    async def get(self, key):
        return self.values.get(key)

    async def delete(self, key):
        return int(self.values.pop(key, None) is not None)

    async def eval(self, script, numkeys, key, owner_id):
        assert numkeys == 1
        assert "redis.call('GET', KEYS[1]) == ARGV[1]" in script
        if self.values.get(key) != owner_id:
            return 0
        del self.values[key]
        return 1


def _strict_state() -> RedisStateManager:
    state = RedisStateManager(None)
    state._redis = _Redis()
    return state


@pytest.mark.asyncio
async def test_execution_lock_refuses_missing_redis_without_memory_fallback():
    state = RedisStateManager(None)

    with pytest.raises(RuntimeError, match="Redis is required"):
        async with cycle_lock(state, "BTC/USDT"):
            pass
    assert state._mem.get("cycle_lock:BTC/USDT") is None


@pytest.mark.asyncio
async def test_same_pair_concurrent_execution_admits_exactly_one_holder():
    state = _strict_state()
    entered = 0
    maximum = 0
    guard = asyncio.Lock()

    async def worker():
        nonlocal entered, maximum
        async with cycle_lock(state, "BTC/USDT") as acquired:
            if not acquired:
                return
            async with guard:
                entered += 1
                maximum = max(maximum, entered)
            await asyncio.sleep(0)
            async with guard:
                entered -= 1

    await asyncio.gather(worker(), worker())
    assert maximum == 1
    assert state._mem.get("cycle_lock:BTC/USDT") is None


@pytest.mark.asyncio
async def test_stale_execution_lease_release_cannot_delete_a_reassigned_redis_key():
    """The comparison and deletion must occur in one Redis-side operation."""

    class ReassignedLeaseRedis(_Redis):
        async def get(self, key):
            previous = await super().get(key)
            self.values[key] = "new-owner"
            return previous

        async def eval(self, script, numkeys, key, owner_id):
            self.values[key] = "new-owner"
            return await super().eval(script, numkeys, key, owner_id)

    state = RedisStateManager(None)
    state._redis = ReassignedLeaseRedis()
    state._redis.values["cycle_lock:BTC/USDT"] = "old-owner"

    released = await state.release_strict_lock("cycle_lock:BTC/USDT", "old-owner")

    assert released is False
    assert state._redis.values["cycle_lock:BTC/USDT"] == "new-owner"

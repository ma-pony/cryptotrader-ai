"""Redis-only execution lease primitives."""

from __future__ import annotations

import asyncio

import pytest

from cryptotrader.cycle_lock import ExecutionLeaseUnavailableError, cycle_lock, execution_book_lease
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


@pytest.mark.asyncio
async def test_execution_book_lease_scopes_pool_and_does_not_translate_body_errors(monkeypatch):
    seen: list[str] = []

    class State:
        async def aclose(self):
            return None

    async def acquire(_state, pair):
        seen.append(pair)
        yield True

    from contextlib import asynccontextmanager

    monkeypatch.setattr("cryptotrader.cycle_lock.RedisStateManager", lambda _url: State(), raising=False)
    monkeypatch.setattr("cryptotrader.cycle_lock.cycle_lock", asynccontextmanager(acquire))

    with pytest.raises(RuntimeError, match="body failure"):
        async with execution_book_lease("redis://localhost/0", "simulation"):
            raise RuntimeError("body failure")
    assert seen == ["book:simulation"]


@pytest.mark.asyncio
async def test_execution_book_lease_uses_dedicated_contention_error(monkeypatch):
    class State:
        async def aclose(self):
            return None

    from contextlib import asynccontextmanager

    async def unavailable(_state, _pair):
        yield False

    monkeypatch.setattr("cryptotrader.cycle_lock.RedisStateManager", lambda _url: State(), raising=False)
    monkeypatch.setattr("cryptotrader.cycle_lock.cycle_lock", asynccontextmanager(unavailable))
    with pytest.raises(ExecutionLeaseUnavailableError):
        async with execution_book_lease("redis://localhost/0", "simulation", wait_timeout=0):
            pass


@pytest.mark.asyncio
async def test_execution_book_lease_closes_an_entered_false_context_before_raising(monkeypatch):
    closed = False

    class State:
        async def aclose(self):
            return None

    class FalseLease:
        async def __aenter__(self):
            return False

        async def __aexit__(self, *_args):
            nonlocal closed
            closed = True

    monkeypatch.setattr("cryptotrader.cycle_lock.RedisStateManager", lambda _url: State(), raising=False)
    monkeypatch.setattr("cryptotrader.cycle_lock.cycle_lock", lambda *_args: FalseLease())

    with pytest.raises(ExecutionLeaseUnavailableError):
        async with execution_book_lease("redis://localhost/0", "simulation", wait_timeout=0):
            pass
    assert closed is True


@pytest.mark.asyncio
async def test_execution_book_lease_cancellation_owns_release_and_redis_close(monkeypatch):
    released = False
    redis_closed = False
    release_started = asyncio.Event()
    release_gate = asyncio.Event()

    class State:
        async def aclose(self):
            nonlocal redis_closed
            redis_closed = True

    class Lease:
        async def __aenter__(self):
            return True

        async def __aexit__(self, *_args):
            nonlocal released
            release_started.set()
            await release_gate.wait()
            released = True

    monkeypatch.setattr("cryptotrader.cycle_lock.RedisStateManager", lambda _url: State(), raising=False)
    monkeypatch.setattr("cryptotrader.cycle_lock.cycle_lock", lambda *_args: Lease())

    async def worker():
        async with execution_book_lease("redis://localhost/0", "simulation"):
            await asyncio.Event().wait()

    task = asyncio.create_task(worker())
    await asyncio.sleep(0)
    task.cancel()
    await release_started.wait()
    task.cancel()
    release_gate.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert release_started.is_set()
    assert released is True
    assert redis_closed is True


@pytest.mark.asyncio
async def test_execution_book_lease_closes_redis_when_lease_exit_fails(monkeypatch):
    redis_closed = False

    class State:
        async def aclose(self):
            nonlocal redis_closed
            redis_closed = True

    class Lease:
        async def __aenter__(self):
            return True

        async def __aexit__(self, *_args):
            raise RuntimeError("release failed")

    monkeypatch.setattr("cryptotrader.cycle_lock.RedisStateManager", lambda _url: State(), raising=False)
    monkeypatch.setattr("cryptotrader.cycle_lock.cycle_lock", lambda *_args: Lease())
    with pytest.raises(RuntimeError, match="release failed"):
        async with execution_book_lease("redis://localhost/0", "simulation"):
            pass
    assert redis_closed is True

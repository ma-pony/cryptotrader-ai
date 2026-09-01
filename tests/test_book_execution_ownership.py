"""Strict distributed pool admission; independent books never share a pair mutex."""

import asyncio

import pytest


class StrictRedis:
    def __init__(self):
        self.values = {}
        self.contended = asyncio.Event()

    async def try_acquire_strict_lock(self, key, owner, ttl):
        if key in self.values:
            self.contended.set()
            return False
        self.values[key] = owner
        return True

    async def release_strict_lock(self, key, owner):
        if self.values.get(key) == owner:
            del self.values[key]
            return True
        return False

    async def aclose(self):
        pass


@pytest.mark.asyncio
async def test_same_book_excludes_another_pair_and_other_book_progresses(monkeypatch):
    import cryptotrader.execution_ownership as module

    owner_type = getattr(module, "ExecutionOwnership", None)
    assert owner_type is not None, "execution ownership must be keyed by book, not pair"
    redis = StrictRedis()
    monkeypatch.setattr("cryptotrader.cycle_lock.RedisStateManager", lambda _: redis)
    owner = owner_type("redis://offline")
    entered = asyncio.Event()

    async def second():
        async with owner.book("a"):
            entered.set()

    async with owner.book("a"):
        task = asyncio.create_task(second())
        await asyncio.wait_for(redis.contended.wait(), 1)
        assert not entered.is_set()
        async with owner.book("b"):
            other_book_was_allowed_to_progress = True
    assert other_book_was_allowed_to_progress
    await asyncio.wait_for(task, 1)
    assert entered.is_set()
    assert redis.values == {}


@pytest.mark.asyncio
async def test_wait_timeout_and_redis_error_are_fail_closed(monkeypatch):
    from cryptotrader.cycle_lock import ExecutionLeaseUnavailableError, execution_book_lease

    redis = StrictRedis()
    redis.values["cycle_lock:book:a"] = "other-owner"
    monkeypatch.setattr("cryptotrader.cycle_lock.RedisStateManager", lambda _: redis)
    with pytest.raises(ExecutionLeaseUnavailableError, match="timed out"):
        async with execution_book_lease("redis://offline", "a", wait_timeout=0):
            pytest.fail("timed out owner admitted")
    assert redis.values["cycle_lock:book:a"] == "other-owner"

    async def unavailable(*_):
        raise RuntimeError("Redis unavailable")

    redis.try_acquire_strict_lock = unavailable
    with pytest.raises(ExecutionLeaseUnavailableError, match="Redis unavailable"):
        async with execution_book_lease("redis://offline", "b"):
            pytest.fail("Redis failure admitted")


@pytest.mark.asyncio
async def test_cancel_waiter_does_not_release_another_owner(monkeypatch):
    from cryptotrader.execution_ownership import ExecutionOwnership

    redis = StrictRedis()
    monkeypatch.setattr("cryptotrader.cycle_lock.RedisStateManager", lambda _: redis)
    owner = ExecutionOwnership("redis://offline")

    async def waiting():
        async with owner.book("a"):
            pytest.fail("cancelled waiter admitted")

    async with owner.book("a"):
        task = asyncio.create_task(waiting())
        await asyncio.wait_for(redis.contended.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert "cycle_lock:book:a" in redis.values


@pytest.mark.asyncio
async def test_unassigned_connection_has_its_own_shared_write_scope(monkeypatch):
    from cryptotrader.execution_ownership import ExecutionOwnership

    redis = StrictRedis()
    monkeypatch.setattr("cryptotrader.cycle_lock.RedisStateManager", lambda _: redis)
    owner = ExecutionOwnership("redis://offline")
    entered = asyncio.Event()

    async def next_writer():
        async with owner.connection("canary"):
            entered.set()

    async with owner.connection("canary"):
        task = asyncio.create_task(next_writer())
        await asyncio.wait_for(redis.contended.wait(), 1)
        assert not entered.is_set()
        async with owner.book("canary"):
            assert set(redis.values) == {"cycle_lock:connection:canary", "cycle_lock:book:canary"}
    await asyncio.wait_for(task, 1)
    assert entered.is_set()
    assert not redis.values

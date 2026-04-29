"""Tests for risk/state.py — _MemoryStore, RedisStateManager memory fallback & Redis paths."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis.exceptions import RedisError

from cryptotrader.risk.state import RedisStateManager, _MemoryStore


class TestMemoryStore:
    def test_get_set(self):
        m = _MemoryStore()
        assert m.get("k") is None
        m.set("k", "v")
        assert m.get("k") == "v"

    def test_set_with_expiry(self):
        m = _MemoryStore()
        m.set("k", "v", ex=3600)
        assert m.get("k") == "v"

    def test_expiry_eviction(self):
        m = _MemoryStore()
        m.set("k", "v", ex=1)
        with patch.object(time, "monotonic", return_value=time.monotonic() + 2):
            assert m.get("k") is None

    def test_incr_new_key(self):
        m = _MemoryStore()
        assert m.incr("c") == 1
        assert m.incr("c") == 2
        assert m.incr("c") == 3

    def test_incr_preserves_expiry(self):
        m = _MemoryStore()
        m.set("c", "5", ex=3600)
        val = m.incr("c")
        assert val == 6

    def test_expire(self):
        m = _MemoryStore()
        m.set("k", "v")
        m.expire("k", 1)
        assert m.get("k") == "v"
        with patch.object(time, "monotonic", return_value=time.monotonic() + 2):
            assert m.get("k") is None

    def test_expire_nonexistent_key(self):
        m = _MemoryStore()
        m.expire("nope", 10)

    def test_delete(self):
        m = _MemoryStore()
        m.set("k", "v")
        m.delete("k")
        assert m.get("k") is None

    def test_delete_nonexistent(self):
        m = _MemoryStore()
        m.delete("nope")

    def test_list_rpush_and_lrange(self):
        m = _MemoryStore()
        m.list_rpush("l", "a")
        m.list_rpush("l", "b")
        m.list_rpush("l", "c")
        assert m.list_lrange("l", 0, -1) == ["a", "b", "c"]
        assert m.list_lrange("l", 0, 1) == ["a", "b"]
        assert m.list_lrange("l", 1, 2) == ["b", "c"]

    def test_list_lrange_empty(self):
        m = _MemoryStore()
        assert m.list_lrange("nope", 0, -1) == []

    def test_list_llen(self):
        m = _MemoryStore()
        assert m.list_llen("l") == 0
        m.list_rpush("l", "x")
        m.list_rpush("l", "y")
        assert m.list_llen("l") == 2

    def test_list_ltrim(self):
        m = _MemoryStore()
        for i in range(5):
            m.list_rpush("l", str(i))
        m.list_ltrim("l", -3, -1)
        assert m.list_lrange("l", 0, -1) == ["2", "3", "4"]

    def test_list_ltrim_empty(self):
        m = _MemoryStore()
        m.list_ltrim("nope", 0, -1)

    def test_list_delete(self):
        m = _MemoryStore()
        m.list_rpush("l", "a")
        m.list_delete("l")
        assert m.list_llen("l") == 0

    def test_list_delete_nonexistent(self):
        m = _MemoryStore()
        m.list_delete("nope")

    def test_list_rpush_with_expiry(self):
        m = _MemoryStore()
        m.list_rpush("l", "a", ex=3600)
        assert m.list_llen("l") == 1
        with patch.object(time, "monotonic", return_value=time.monotonic() + 7200):
            assert m.list_llen("l") == 0

    def test_list_rpush_existing_with_new_expiry(self):
        m = _MemoryStore()
        m.list_rpush("l", "a")
        m.list_rpush("l", "b", ex=100)
        assert m.list_llen("l") == 2


class TestRedisStateManagerMemoryFallback:
    """Test RedisStateManager with redis_url=None (pure memory fallback)."""

    @pytest.mark.asyncio
    async def test_init_no_redis(self):
        mgr = RedisStateManager(None)
        assert not mgr.available

    @pytest.mark.asyncio
    async def test_init_disabled(self):
        mgr = RedisStateManager("DISABLED")
        assert not mgr.available

    @pytest.mark.asyncio
    async def test_ping_no_redis(self):
        mgr = RedisStateManager(None)
        assert await mgr.ping() is False

    @pytest.mark.asyncio
    async def test_get_set(self):
        mgr = RedisStateManager(None)
        assert await mgr.get("k") is None
        await mgr.set("k", "v")
        assert await mgr.get("k") == "v"

    @pytest.mark.asyncio
    async def test_set_with_expiry(self):
        mgr = RedisStateManager(None)
        await mgr.set("k", "v", ex=3600)
        assert await mgr.get("k") == "v"

    @pytest.mark.asyncio
    async def test_incr(self):
        mgr = RedisStateManager(None)
        assert await mgr.incr("c") == 1
        assert await mgr.incr("c") == 2

    @pytest.mark.asyncio
    async def test_expire(self):
        mgr = RedisStateManager(None)
        await mgr.set("k", "v")
        await mgr.expire("k", 3600)
        assert await mgr.get("k") == "v"

    @pytest.mark.asyncio
    async def test_set_cooldown(self):
        mgr = RedisStateManager(None)
        await mgr.set_cooldown("BTC/USDT", 5)
        assert await mgr.get("cooldown:BTC/USDT") == "1"

    @pytest.mark.asyncio
    async def test_set_post_loss_cooldown(self):
        mgr = RedisStateManager(None)
        await mgr.set_post_loss_cooldown(10)
        assert await mgr.get("cooldown:post_loss") == "1"

    @pytest.mark.asyncio
    async def test_incr_trade_count_and_get(self):
        mgr = RedisStateManager(None)
        await mgr.incr_trade_count()
        hourly, daily = await mgr.get_trade_counts()
        assert hourly == 1
        assert daily == 1
        await mgr.incr_trade_count()
        hourly, daily = await mgr.get_trade_counts()
        assert hourly == 2
        assert daily == 2

    @pytest.mark.asyncio
    async def test_circuit_breaker(self):
        mgr = RedisStateManager(None)
        assert await mgr.is_circuit_breaker_active() is False
        await mgr.set_circuit_breaker()
        assert await mgr.is_circuit_breaker_active() is True
        await mgr.reset_circuit_breaker()
        assert await mgr.is_circuit_breaker_active() is False

    @pytest.mark.asyncio
    async def test_buffer_push_and_range(self):
        mgr = RedisStateManager(None)
        await mgr.buffer_push("buf", "a", max_size=3, ttl=3600)
        await mgr.buffer_push("buf", "b", max_size=3, ttl=3600)
        await mgr.buffer_push("buf", "c", max_size=3, ttl=3600)
        assert await mgr.buffer_range("buf", 0, -1) == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_buffer_push_trims(self):
        mgr = RedisStateManager(None)
        for i in range(5):
            await mgr.buffer_push("buf", str(i), max_size=3, ttl=3600)
        assert await mgr.buffer_len("buf") == 3
        assert await mgr.buffer_range("buf", 0, -1) == ["2", "3", "4"]

    @pytest.mark.asyncio
    async def test_buffer_len(self):
        mgr = RedisStateManager(None)
        assert await mgr.buffer_len("buf") == 0
        await mgr.buffer_push("buf", "x", max_size=10, ttl=3600)
        assert await mgr.buffer_len("buf") == 1

    @pytest.mark.asyncio
    async def test_buffer_delete(self):
        mgr = RedisStateManager(None)
        await mgr.buffer_push("buf", "x", max_size=10, ttl=3600)
        await mgr.buffer_delete("buf")
        assert await mgr.buffer_len("buf") == 0

    @pytest.mark.asyncio
    async def test_buffer_set_ttl(self):
        mgr = RedisStateManager(None)
        await mgr.buffer_push("buf", "x", max_size=10, ttl=3600)
        await mgr.buffer_set_ttl("buf", 3600)
        assert await mgr.buffer_len("buf") == 1

    @pytest.mark.asyncio
    async def test_publish_in_proc(self):
        mgr = RedisStateManager(None)
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=10)
        mgr._in_proc_queues.setdefault("ch", set()).add(q)
        await mgr.publish("ch", "hello")
        assert q.get_nowait() == "hello"

    @pytest.mark.asyncio
    async def test_publish_full_queue(self):
        mgr = RedisStateManager(None)
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=1)
        q.put_nowait("fill")
        mgr._in_proc_queues.setdefault("ch", set()).add(q)
        await mgr.publish("ch", "overflow")
        assert q.qsize() == 1


class TestRedisStateManagerWithMockRedis:
    """Test RedisStateManager when a mock Redis client is injected."""

    def _make_mgr(self, mock_redis=None):
        mgr = RedisStateManager(None)
        mgr._redis = mock_redis or AsyncMock()
        return mgr

    @pytest.mark.asyncio
    async def test_available_with_redis(self):
        mgr = self._make_mgr()
        assert mgr.available is True

    @pytest.mark.asyncio
    async def test_ping_success(self):
        mock_r = AsyncMock()
        mock_r.ping = AsyncMock(return_value=True)
        mgr = self._make_mgr(mock_r)
        assert await mgr.ping() is True

    @pytest.mark.asyncio
    async def test_ping_failure(self):
        mock_r = AsyncMock()
        mock_r.ping = AsyncMock(side_effect=Exception("down"))
        mgr = self._make_mgr(mock_r)
        assert await mgr.ping() is False

    @pytest.mark.asyncio
    async def test_get_from_redis(self):
        mock_r = AsyncMock()
        mock_r.get = AsyncMock(return_value=b"hello")
        mgr = self._make_mgr(mock_r)
        assert await mgr.get("k") == "hello"

    @pytest.mark.asyncio
    async def test_get_redis_returns_str(self):
        mock_r = AsyncMock()
        mock_r.get = AsyncMock(return_value="world")
        mgr = self._make_mgr(mock_r)
        assert await mgr.get("k") == "world"

    @pytest.mark.asyncio
    async def test_get_redis_error_fallback(self):
        mock_r = AsyncMock()
        mock_r.get = AsyncMock(side_effect=RedisError("fail"))
        mgr = self._make_mgr(mock_r)
        mgr._mem.set("k", "fallback_val")
        assert await mgr.get("k") == "fallback_val"

    @pytest.mark.asyncio
    async def test_set_to_redis(self):
        mock_r = AsyncMock()
        mock_r.set = AsyncMock()
        mgr = self._make_mgr(mock_r)
        await mgr.set("k", "v", ex=60)
        mock_r.set.assert_called_once_with("k", "v", ex=60)

    @pytest.mark.asyncio
    async def test_set_redis_error_fallback(self):
        mock_r = AsyncMock()
        mock_r.set = AsyncMock(side_effect=RedisError("fail"))
        mgr = self._make_mgr(mock_r)
        await mgr.set("k", "v")
        assert mgr._mem.get("k") == "v"

    @pytest.mark.asyncio
    async def test_incr_from_redis(self):
        mock_r = AsyncMock()
        mock_r.incr = AsyncMock(return_value=5)
        mgr = self._make_mgr(mock_r)
        assert await mgr.incr("c") == 5

    @pytest.mark.asyncio
    async def test_incr_redis_error_fallback(self):
        mock_r = AsyncMock()
        mock_r.incr = AsyncMock(side_effect=RedisError("fail"))
        mgr = self._make_mgr(mock_r)
        assert await mgr.incr("c") == 1

    @pytest.mark.asyncio
    async def test_expire_from_redis(self):
        mock_r = AsyncMock()
        mock_r.expire = AsyncMock()
        mgr = self._make_mgr(mock_r)
        await mgr.expire("k", 3600)
        mock_r.expire.assert_called_once_with("k", 3600)

    @pytest.mark.asyncio
    async def test_expire_redis_error_fallback(self):
        mock_r = AsyncMock()
        mock_r.expire = AsyncMock(side_effect=RedisError("fail"))
        mgr = self._make_mgr(mock_r)
        mgr._mem.set("k", "v")
        await mgr.expire("k", 3600)

    @pytest.mark.asyncio
    async def test_reset_circuit_breaker_redis(self):
        mock_r = AsyncMock()
        mock_r.delete = AsyncMock()
        mgr = self._make_mgr(mock_r)
        await mgr.reset_circuit_breaker()
        mock_r.delete.assert_called_once_with("circuit_breaker:active")

    @pytest.mark.asyncio
    async def test_reset_circuit_breaker_redis_error(self):
        mock_r = AsyncMock()
        mock_r.delete = AsyncMock(side_effect=RedisError("fail"))
        mgr = self._make_mgr(mock_r)
        mgr._mem.set("circuit_breaker:active", "1")
        await mgr.reset_circuit_breaker()
        assert mgr._mem.get("circuit_breaker:active") is None

    @pytest.mark.asyncio
    async def test_buffer_push_redis(self):
        mock_pipe = AsyncMock()
        mock_r = AsyncMock()
        mock_r.pipeline = MagicMock(return_value=mock_pipe)
        mgr = self._make_mgr(mock_r)
        await mgr.buffer_push("buf", "val", max_size=10, ttl=3600)
        mock_pipe.rpush.assert_called_once_with("buf", "val")
        mock_pipe.ltrim.assert_called_once_with("buf", -10, -1)
        mock_pipe.expire.assert_called_once_with("buf", 3600)
        mock_pipe.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_buffer_push_redis_error(self):
        mock_pipe = AsyncMock()
        mock_pipe.execute = AsyncMock(side_effect=RedisError("fail"))
        mock_r = AsyncMock()
        mock_r.pipeline = MagicMock(return_value=mock_pipe)
        mgr = self._make_mgr(mock_r)
        await mgr.buffer_push("buf", "val", max_size=10, ttl=3600)
        assert mgr._mem.list_llen("buf") == 1

    @pytest.mark.asyncio
    async def test_buffer_range_redis(self):
        mock_r = AsyncMock()
        mock_r.lrange = AsyncMock(return_value=[b"a", b"b"])
        mgr = self._make_mgr(mock_r)
        result = await mgr.buffer_range("buf", 0, -1)
        assert result == ["a", "b"]

    @pytest.mark.asyncio
    async def test_buffer_range_redis_error(self):
        mock_r = AsyncMock()
        mock_r.lrange = AsyncMock(side_effect=RedisError("fail"))
        mgr = self._make_mgr(mock_r)
        result = await mgr.buffer_range("buf", 0, -1)
        assert result == []

    @pytest.mark.asyncio
    async def test_buffer_len_redis(self):
        mock_r = AsyncMock()
        mock_r.llen = AsyncMock(return_value=5)
        mgr = self._make_mgr(mock_r)
        assert await mgr.buffer_len("buf") == 5

    @pytest.mark.asyncio
    async def test_buffer_len_redis_error(self):
        mock_r = AsyncMock()
        mock_r.llen = AsyncMock(side_effect=RedisError("fail"))
        mgr = self._make_mgr(mock_r)
        assert await mgr.buffer_len("buf") == 0

    @pytest.mark.asyncio
    async def test_buffer_set_ttl_redis(self):
        mock_r = AsyncMock()
        mock_r.expire = AsyncMock()
        mgr = self._make_mgr(mock_r)
        await mgr.buffer_set_ttl("buf", 3600)
        mock_r.expire.assert_called_once_with("buf", 3600)

    @pytest.mark.asyncio
    async def test_buffer_set_ttl_redis_error(self):
        mock_r = AsyncMock()
        mock_r.expire = AsyncMock(side_effect=RedisError("fail"))
        mgr = self._make_mgr(mock_r)
        await mgr.buffer_set_ttl("buf", 3600)

    @pytest.mark.asyncio
    async def test_buffer_delete_redis(self):
        mock_r = AsyncMock()
        mock_r.delete = AsyncMock()
        mgr = self._make_mgr(mock_r)
        await mgr.buffer_delete("buf")
        mock_r.delete.assert_called_once_with("buf")

    @pytest.mark.asyncio
    async def test_buffer_delete_redis_error(self):
        mock_r = AsyncMock()
        mock_r.delete = AsyncMock(side_effect=RedisError("fail"))
        mgr = self._make_mgr(mock_r)
        mgr._mem.list_rpush("buf", "x")
        await mgr.buffer_delete("buf")
        assert mgr._mem.list_llen("buf") == 0

    @pytest.mark.asyncio
    async def test_publish_to_redis(self):
        mock_r = AsyncMock()
        mock_r.publish = AsyncMock()
        mgr = self._make_mgr(mock_r)
        await mgr.publish("ch", "msg")
        mock_r.publish.assert_called_once_with("ch", "msg")

    @pytest.mark.asyncio
    async def test_publish_redis_error_still_sends_inproc(self):
        mock_r = AsyncMock()
        mock_r.publish = AsyncMock(side_effect=RedisError("fail"))
        mgr = self._make_mgr(mock_r)
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=10)
        mgr._in_proc_queues.setdefault("ch", set()).add(q)
        await mgr.publish("ch", "msg")
        assert q.get_nowait() == "msg"

    @pytest.mark.asyncio
    async def test_init_redis_connection_error(self):
        with patch("cryptotrader.risk.state.redis.from_url", side_effect=Exception("bad url")):
            mgr = RedisStateManager("redis://bad-host:6379")
        assert mgr._redis is None
        assert not mgr.available

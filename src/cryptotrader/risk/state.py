"""Redis state manager for risk checks — real connection with TTL support.

Falls back to in-memory counters when Redis is unavailable so that
cooldown and rate-limit checks still function (best-effort, single-process).
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING

from cryptotrader._compat import UTC

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

import redis.asyncio as redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)


class _MemoryStore:
    """Minimal in-memory key-value store with TTL expiry."""

    def __init__(self) -> None:
        self._data: dict[str, tuple[str, float | None]] = {}  # key -> (value, expire_ts)
        self._lists: dict[str, tuple[list[str], float | None]] = {}  # key -> (items, expire_ts)

    def _evict(self, key: str) -> None:
        entry = self._data.get(key)
        if entry and entry[1] is not None and time.monotonic() > entry[1]:
            del self._data[key]

    def get(self, key: str) -> str | None:
        self._evict(key)
        entry = self._data.get(key)
        return entry[0] if entry else None

    def set(self, key: str, value: str, ex: int | None = None) -> None:
        expire_ts = (time.monotonic() + ex) if ex else None
        self._data[key] = (value, expire_ts)

    def incr(self, key: str) -> int:
        self._evict(key)
        entry = self._data.get(key)
        val = int(entry[0]) + 1 if entry else 1
        expire_ts = entry[1] if entry else None
        self._data[key] = (str(val), expire_ts)
        return val

    def expire(self, key: str, seconds: int) -> None:
        entry = self._data.get(key)
        if entry:
            self._data[key] = (entry[0], time.monotonic() + seconds)

    def ttl(self, key: str) -> int:
        """Remaining TTL in seconds. -2 if missing, -1 if no expiry."""
        self._evict(key)
        entry = self._data.get(key)
        if entry is None:
            return -2
        if entry[1] is None:
            return -1
        remaining = entry[1] - time.monotonic()
        return max(0, int(remaining))

    def delete(self, key: str) -> None:
        self._data.pop(key, None)

    def _evict_list(self, key: str) -> None:
        entry = self._lists.get(key)
        if entry and entry[1] is not None and time.monotonic() > entry[1]:
            del self._lists[key]

    def list_rpush(self, key: str, value: str, ex: int | None = None) -> None:
        self._evict_list(key)
        entry = self._lists.get(key)
        if entry:
            entry[0].append(value)
            if ex is not None:
                self._lists[key] = (entry[0], time.monotonic() + ex)
        else:
            expire_ts = (time.monotonic() + ex) if ex else None
            self._lists[key] = ([value], expire_ts)

    def list_lrange(self, key: str, start: int, end: int) -> list[str]:
        self._evict_list(key)
        entry = self._lists.get(key)
        if not entry:
            return []
        items = entry[0]
        end = len(items) + end + 1 if end < 0 else end + 1
        return items[start:end]

    def list_llen(self, key: str) -> int:
        self._evict_list(key)
        entry = self._lists.get(key)
        return len(entry[0]) if entry else 0

    def list_ltrim(self, key: str, start: int, end: int) -> None:
        self._evict_list(key)
        entry = self._lists.get(key)
        if not entry:
            return
        items = entry[0]
        end = len(items) + end + 1 if end < 0 else end + 1
        self._lists[key] = (items[start:end], entry[1])

    def list_delete(self, key: str) -> None:
        self._lists.pop(key, None)


class RedisStateManager:
    def __init__(self, redis_url: str | None) -> None:
        self._redis: redis.Redis | None = None
        self._mem = _MemoryStore()
        self._in_proc_queues: dict[str, set[asyncio.Queue[str]]] = {}
        if redis_url and redis_url != "DISABLED":
            try:
                self._redis = redis.from_url(redis_url)
            except Exception:
                logger.warning("Redis connection failed, using memory fallback", exc_info=True)
                self._redis = None

    @property
    def available(self) -> bool:
        return self._redis is not None

    async def ping(self) -> bool:
        """Check if Redis is currently reachable."""
        if self._redis is None:
            return False
        try:
            await self._redis.ping()
            self._last_ping_error: dict[str, str] | None = None
            return True
        except Exception as e:
            logger.warning("Redis ping failed: %s: %s", type(e).__name__, e, exc_info=True)
            self._last_ping_error = {"type": type(e).__name__, "msg": str(e)}
            return False

    async def get(self, key: str) -> str | None:
        if self._redis is not None:
            try:
                val = await self._redis.get(key)
                return val.decode() if isinstance(val, bytes) else val
            except RedisError:
                logger.debug("Redis get failed for %s, using memory fallback", key)
        return self._mem.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        if self._redis is not None:
            try:
                await self._redis.set(key, value, ex=ex)
                return
            except RedisError:
                logger.debug("Redis set failed for %s, using memory fallback", key)
        self._mem.set(key, value, ex=ex)

    async def incr(self, key: str) -> int | None:
        if self._redis is not None:
            try:
                return await self._redis.incr(key)
            except RedisError:
                logger.debug("Redis incr failed for %s, using memory fallback", key)
        return self._mem.incr(key)

    async def expire(self, key: str, seconds: int) -> None:
        if self._redis is not None:
            try:
                await self._redis.expire(key, seconds)
                return
            except RedisError:
                logger.debug("Redis expire failed for %s, using memory fallback", key)
        self._mem.expire(key, seconds)

    async def ttl(self, key: str) -> int:
        """Return seconds remaining until ``key`` expires.

        Mirrors the Redis TTL contract:
          -2 = key does not exist
          -1 = key exists but has no expiry
          N  = N seconds remaining (always >= 0)
        """
        if self._redis is not None:
            try:
                raw = await self._redis.ttl(key)
                return int(raw)
            except RedisError:
                logger.debug("Redis ttl failed for %s, using memory fallback", key)
        return self._mem.ttl(key)

    async def set_cooldown(self, pair: str, minutes: int) -> None:
        await self.set(f"cooldown:{pair}", "1", ex=minutes * 60)

    async def set_post_loss_cooldown(self, minutes: int) -> None:
        await self.set("cooldown:post_loss", "1", ex=minutes * 60)

    async def incr_trade_count(self) -> None:
        now = datetime.now(UTC)
        hourly_key = f"trades:hourly:{now.strftime('%Y%m%d%H')}"
        daily_key = f"trades:daily:{now.strftime('%Y%m%d')}"
        await self.incr(hourly_key)
        await self.expire(hourly_key, 3600)
        await self.incr(daily_key)
        await self.expire(daily_key, 86400)

    async def get_trade_counts(self) -> tuple[int, int]:
        now = datetime.now(UTC)
        hourly = await self.get(f"trades:hourly:{now.strftime('%Y%m%d%H')}")
        daily = await self.get(f"trades:daily:{now.strftime('%Y%m%d')}")
        return int(hourly or 0), int(daily or 0)

    async def set_circuit_breaker(self, ttl_seconds: int = 86400) -> None:
        await self.set("circuit_breaker:active", "1", ex=ttl_seconds)

    async def is_circuit_breaker_active(self) -> bool:
        val = await self.get("circuit_breaker:active")
        return val is not None

    async def reset_circuit_breaker(self) -> None:
        if self._redis is not None:
            try:
                await self._redis.delete("circuit_breaker:active")
                return
            except RedisError:
                logger.debug("Redis delete failed for circuit_breaker, using memory fallback")
        self._mem.delete("circuit_breaker:active")

    # ── Lock primitives ──
    #
    # SETNX-with-TTL + owner-check release. Used by ``cycle_lock`` to keep
    # ``arena run`` and the launchd scheduler from racing each other on the
    # same pair (production observation 2026-05-02: a manual ``arena run``
    # while the scheduler restarted produced two concurrent ETH close orders
    # 426 ms apart — only one filled, but a second BUY would have doubled
    # exposure).

    async def try_acquire_lock(self, key: str, owner_id: str, ttl: int) -> bool:
        """Atomic SET NX EX. Returns True if we now own the lock."""
        if self._redis is not None:
            try:
                # set(nx=True) returns True only when the key was created.
                acquired = await self._redis.set(key, owner_id, nx=True, ex=ttl)
                return bool(acquired)
            except RedisError:
                logger.debug("Redis SET NX failed for %s, using memory fallback", key)
        # Memory path: emulate NX semantics.
        if self._mem.get(key) is not None:
            return False
        self._mem.set(key, owner_id, ex=ttl)
        return True

    async def release_lock(self, key: str, owner_id: str) -> bool:
        """Delete only if we still own the key. Returns True on actual delete.

        Owner check prevents one process from releasing another's lock when a
        prior holder's TTL expired and a new holder took over.
        """
        current = await self.get(key)
        if current != owner_id:
            return False
        if self._redis is not None:
            try:
                await self._redis.delete(key)
                return True
            except RedisError:
                logger.debug("Redis DEL failed for %s, using memory fallback", key)
        self._mem.delete(key)
        return True

    # ── List (buffer) operations ──

    async def buffer_push(self, key: str, value: str, max_size: int, ttl: int) -> None:
        if self._redis is not None:
            try:
                pipe = self._redis.pipeline()
                pipe.rpush(key, value)
                pipe.ltrim(key, -max_size, -1)
                pipe.expire(key, ttl)
                await pipe.execute()
                return
            except RedisError:
                logger.debug("Redis buffer_push failed for %s, using memory fallback", key)
        self._mem.list_rpush(key, value, ex=ttl)
        if self._mem.list_llen(key) > max_size:
            self._mem.list_ltrim(key, -max_size, -1)

    async def buffer_range(self, key: str, start: int, end: int) -> list[str]:
        if self._redis is not None:
            try:
                raw = await self._redis.lrange(key, start, end)
                return [v.decode() if isinstance(v, bytes) else v for v in raw]
            except RedisError:
                logger.debug("Redis buffer_range failed for %s, using memory fallback", key)
        return self._mem.list_lrange(key, start, end)

    async def buffer_len(self, key: str) -> int:
        if self._redis is not None:
            try:
                return await self._redis.llen(key)
            except RedisError:
                logger.debug("Redis buffer_len failed for %s, using memory fallback", key)
        return self._mem.list_llen(key)

    async def buffer_set_ttl(self, key: str, ttl_s: int) -> None:
        if self._redis is not None:
            try:
                await self._redis.expire(key, ttl_s)
                return
            except RedisError:
                logger.debug("Redis buffer_set_ttl failed for %s, using memory fallback", key)
        self._mem.expire(key, ttl_s)

    async def buffer_delete(self, key: str) -> None:
        if self._redis is not None:
            try:
                await self._redis.delete(key)
                return
            except RedisError:
                logger.debug("Redis buffer_delete failed for %s, using memory fallback", key)
        self._mem.list_delete(key)

    # ── Pub/Sub operations ──

    async def publish(self, channel: str, message: str) -> None:
        if self._redis is not None:
            try:
                await self._redis.publish(channel, message)
            except RedisError:
                logger.debug("Redis publish failed for %s, using memory fallback", channel)
        for q in self._in_proc_queues.get(channel, set()):
            try:
                q.put_nowait(message)
            except asyncio.QueueFull:
                logger.debug("In-proc queue full for channel %s, dropping message", channel)

    async def subscribe_iter(self, channel: str) -> AsyncIterator[str]:
        if self._redis is not None:
            try:
                pubsub = self._redis.pubsub()
                await pubsub.subscribe(channel)
                try:
                    async for msg in pubsub.listen():
                        if msg["type"] == "message":
                            data = msg["data"]
                            yield data.decode() if isinstance(data, bytes) else data
                finally:
                    await pubsub.unsubscribe(channel)
                    await pubsub.close()
                return
            except RedisError:
                logger.debug("Redis subscribe failed for %s, using memory fallback", channel)
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=100)
        subs = self._in_proc_queues.setdefault(channel, set())
        subs.add(q)
        try:
            while True:
                yield await q.get()
        finally:
            subs.discard(q)

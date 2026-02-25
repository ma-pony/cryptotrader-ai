"""Redis state manager for risk checks — real connection with TTL support.

Falls back to in-memory counters when Redis is unavailable so that
cooldown and rate-limit checks still function (best-effort, single-process).
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime

import redis.asyncio as redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)


class _MemoryStore:
    """Minimal in-memory key-value store with TTL expiry."""

    def __init__(self) -> None:
        self._data: dict[str, tuple[str, float | None]] = {}  # key -> (value, expire_ts)

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

    def delete(self, key: str) -> None:
        self._data.pop(key, None)


class RedisStateManager:
    def __init__(self, redis_url: str | None) -> None:
        self._redis: redis.Redis | None = None
        self._mem = _MemoryStore()
        if redis_url:
            try:
                self._redis = redis.from_url(redis_url)
            except Exception:
                self._redis = None

    @property
    def available(self) -> bool:
        return self._redis is not None

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

    async def set_circuit_breaker(self) -> None:
        await self.set("circuit_breaker:active", "1")

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

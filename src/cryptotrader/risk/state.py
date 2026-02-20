"""Redis state manager for risk checks — real connection with TTL support."""

from __future__ import annotations

import logging
from datetime import datetime

import redis.asyncio as redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)


class RedisStateManager:
    def __init__(self, redis_url: str | None) -> None:
        self._redis: redis.Redis | None = None
        if redis_url:
            try:
                self._redis = redis.from_url(redis_url)
            except Exception:
                self._redis = None

    @property
    def available(self) -> bool:
        return self._redis is not None

    async def get(self, key: str) -> str | None:
        try:
            val = await self._redis.get(key)
            return val.decode() if isinstance(val, bytes) else val
        except (RedisError, AttributeError):
            return None

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        try:
            await self._redis.set(key, value, ex=ex)
        except (RedisError, AttributeError):
            pass

    async def incr(self, key: str) -> int | None:
        try:
            return await self._redis.incr(key)
        except (RedisError, AttributeError):
            return None

    async def expire(self, key: str, seconds: int) -> None:
        try:
            await self._redis.expire(key, seconds)
        except (RedisError, AttributeError):
            pass

    async def set_cooldown(self, pair: str, minutes: int) -> None:
        await self.set(f"cooldown:{pair}", "1", ex=minutes * 60)

    async def incr_trade_count(self) -> None:
        now = datetime.utcnow()
        hourly_key = f"trades:hourly:{now.strftime('%Y%m%d%H')}"
        daily_key = f"trades:daily:{now.strftime('%Y%m%d')}"
        await self.incr(hourly_key)
        await self.expire(hourly_key, 3600)
        await self.incr(daily_key)
        await self.expire(daily_key, 86400)

    async def get_trade_counts(self) -> tuple[int, int]:
        now = datetime.utcnow()
        hourly = await self.get(f"trades:hourly:{now.strftime('%Y%m%d%H')}")
        daily = await self.get(f"trades:daily:{now.strftime('%Y%m%d')}")
        return int(hourly or 0), int(daily or 0)

    async def set_circuit_breaker(self) -> None:
        await self.set("circuit_breaker:active", "1")

    async def is_circuit_breaker_active(self) -> bool:
        val = await self.get("circuit_breaker:active")
        return val is not None

    async def reset_circuit_breaker(self) -> None:
        try:
            await self._redis.delete("circuit_breaker:active")
        except (RedisError, AttributeError):
            pass

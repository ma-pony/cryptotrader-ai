"""Redis state manager for risk checks."""

from __future__ import annotations

import redis.asyncio as redis
from redis.exceptions import RedisError


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
            return await self._redis.get(key)
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

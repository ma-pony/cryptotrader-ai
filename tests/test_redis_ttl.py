"""Unit tests for RedisStateManager.ttl + _MemoryStore.ttl sentinel semantics."""

from __future__ import annotations

import time

import pytest

from cryptotrader.risk.state import RedisStateManager, _MemoryStore


class TestMemoryStoreTTL:
    def test_missing_key_returns_minus_two(self) -> None:
        m = _MemoryStore()
        assert m.ttl("never-set") == -2

    def test_key_without_expiry_returns_minus_one(self) -> None:
        m = _MemoryStore()
        m.set("persistent", "value")  # no ex → no expiry
        assert m.ttl("persistent") == -1

    def test_key_with_expiry_returns_positive_seconds(self) -> None:
        m = _MemoryStore()
        m.set("ephemeral", "value", ex=60)
        ttl = m.ttl("ephemeral")
        assert 50 <= ttl <= 60  # allow small slack

    def test_expired_key_returns_minus_two(self) -> None:
        m = _MemoryStore()
        m.set("soon", "value", ex=1)
        time.sleep(1.1)
        assert m.ttl("soon") == -2


class TestRedisStateManagerTTLFallback:
    """When Redis isn't configured, ttl delegates to the in-memory backend."""

    @pytest.mark.asyncio
    async def test_memory_mode_ttl_missing(self) -> None:
        rsm = RedisStateManager(None)  # Redis disabled → memory fallback
        assert await rsm.ttl("nothing") == -2

    @pytest.mark.asyncio
    async def test_memory_mode_ttl_with_cooldown(self) -> None:
        rsm = RedisStateManager(None)
        await rsm.set_cooldown("BTC/USDT", minutes=5)
        ttl = await rsm.ttl("cooldown:BTC/USDT")
        assert 290 <= ttl <= 300  # 5 min ± small slack

    @pytest.mark.asyncio
    async def test_memory_mode_ttl_post_loss(self) -> None:
        rsm = RedisStateManager(None)
        await rsm.set_post_loss_cooldown(minutes=10)
        assert await rsm.ttl("cooldown:post_loss") > 590

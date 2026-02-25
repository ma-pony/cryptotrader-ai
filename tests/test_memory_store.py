"""Tests for RedisStateManager in-memory fallback (_MemoryStore)."""

import time

import pytest

from cryptotrader.risk.state import RedisStateManager


@pytest.fixture
def rsm():
    """RedisStateManager with no Redis — pure in-memory mode."""
    return RedisStateManager(None)


@pytest.mark.asyncio
async def test_get_set_basic(rsm):
    assert await rsm.get("key1") is None
    await rsm.set("key1", "val1")
    assert await rsm.get("key1") == "val1"


@pytest.mark.asyncio
async def test_set_with_ttl_expires(rsm):
    await rsm.set("key2", "val2", ex=1)
    assert await rsm.get("key2") == "val2"
    time.sleep(1.1)
    assert await rsm.get("key2") is None


@pytest.mark.asyncio
async def test_incr_new_key(rsm):
    result = await rsm.incr("counter")
    assert result == 1
    result = await rsm.incr("counter")
    assert result == 2


@pytest.mark.asyncio
async def test_expire_sets_ttl(rsm):
    await rsm.set("key3", "val3")
    await rsm.expire("key3", 1)
    assert await rsm.get("key3") == "val3"
    time.sleep(1.1)
    assert await rsm.get("key3") is None


@pytest.mark.asyncio
async def test_cooldown_roundtrip(rsm):
    await rsm.set_cooldown("BTC/USDT", 1)
    assert await rsm.get("cooldown:BTC/USDT") == "1"


@pytest.mark.asyncio
async def test_trade_counts(rsm):
    hourly, daily = await rsm.get_trade_counts()
    assert hourly == 0
    assert daily == 0
    await rsm.incr_trade_count()
    await rsm.incr_trade_count()
    hourly, daily = await rsm.get_trade_counts()
    assert hourly == 2
    assert daily == 2


@pytest.mark.asyncio
async def test_circuit_breaker(rsm):
    assert not await rsm.is_circuit_breaker_active()
    await rsm.set_circuit_breaker()
    assert await rsm.is_circuit_breaker_active()
    await rsm.reset_circuit_breaker()
    assert not await rsm.is_circuit_breaker_active()

"""Tests for EventBuffer — push, replay, overflow, TTL, memory fallback."""

from __future__ import annotations

import pytest

from cryptotrader.chat.event_buffer import EventBuffer
from cryptotrader.chat.event_bus import SSEEnvelope
from cryptotrader.risk.state import RedisStateManager


@pytest.fixture
def state_mgr():
    return RedisStateManager(None)


@pytest.fixture
def buffer(state_mgr):
    return EventBuffer("test-session", state_mgr, ttl_s=300, max_size=10)


@pytest.mark.asyncio
async def test_push_and_range_after(buffer):
    env = SSEEnvelope(event_id=1, type="node_done", ts="2026-01-01T00:00:00Z", session_id="test-session", data={"x": 1})
    await buffer.push(env)
    result = await buffer.range_after(0)
    assert len(result) == 1
    assert result[0].event_id == 1
    assert result[0].type == "node_done"


@pytest.mark.asyncio
async def test_range_after_filters_by_event_id(buffer):
    for i in range(1, 6):
        env = SSEEnvelope(event_id=i, type="node_done", ts="2026-01-01T00:00:00Z", session_id="test-session")
        await buffer.push(env)
    result = await buffer.range_after(3)
    assert len(result) == 2
    assert [r.event_id for r in result] == [4, 5]


@pytest.mark.asyncio
async def test_range_after_empty(buffer):
    result = await buffer.range_after(0)
    assert result == []


@pytest.mark.asyncio
async def test_next_event_id_increments(buffer):
    id1 = await buffer.next_event_id()
    id2 = await buffer.next_event_id()
    id3 = await buffer.next_event_id()
    assert id1 < id2 < id3


@pytest.mark.asyncio
async def test_overflow_protection(state_mgr):
    buf = EventBuffer("overflow-test", state_mgr, ttl_s=300, max_size=3)
    for i in range(5):
        env = SSEEnvelope(event_id=i + 1, type="test", ts="2026-01-01T00:00:00Z", session_id="overflow-test")
        await buf.push(env)
    result = await buf.range_after(0)
    assert len(result) <= 3


@pytest.mark.asyncio
async def test_exists(buffer):
    assert not await buffer.exists()
    env = SSEEnvelope(event_id=1, type="test", ts="2026-01-01T00:00:00Z", session_id="test-session")
    await buffer.push(env)
    assert await buffer.exists()


@pytest.mark.asyncio
async def test_mark_done(buffer):
    await buffer.mark_done("stream_done")
    result = await buffer.range_after(0)
    assert len(result) == 1
    assert result[0].type == "stream_done"

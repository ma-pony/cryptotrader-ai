"""Tests for chat control API endpoints — interrupt, steer, watch."""

from __future__ import annotations

import asyncio

import pytest

from cryptotrader.chat.event_buffer import EventBuffer
from cryptotrader.chat.event_bus import EventBus
from cryptotrader.chat.task_manager import BackgroundTaskManager
from cryptotrader.config import ChatConfig
from cryptotrader.risk.state import RedisStateManager


@pytest.fixture(autouse=True)
def _reset_singleton():
    BackgroundTaskManager.reset()
    yield
    BackgroundTaskManager.reset()


@pytest.fixture
def state_mgr():
    return RedisStateManager(None)


def _make_bus(session_id: str, state_mgr: RedisStateManager) -> EventBus:
    buf = EventBuffer(session_id, state_mgr)
    return EventBus(session_id, buf)


async def _long_coro():
    await asyncio.sleep(10)


@pytest.mark.asyncio
async def test_interrupt_returns_received(state_mgr):
    from api.routes.chat_control import interrupt_analysis

    config = ChatConfig(max_concurrent_tasks=5)
    mgr = BackgroundTaskManager.get_instance(config)
    bus = _make_bus("s1", state_mgr)
    mgr.create("s1", "BTC/USDT", _long_coro(), "chat", bus)

    resp = await interrupt_analysis("s1")
    assert resp.type == "interrupt_received"
    assert resp.session_id == "s1"


@pytest.mark.asyncio
async def test_interrupt_noop_when_already_interrupted(state_mgr):
    from api.routes.chat_control import interrupt_analysis

    config = ChatConfig(max_concurrent_tasks=5)
    mgr = BackgroundTaskManager.get_instance(config)
    bus = _make_bus("s1", state_mgr)
    mgr.create("s1", "BTC/USDT", _long_coro(), "chat", bus)

    await interrupt_analysis("s1")
    resp = await interrupt_analysis("s1")
    assert resp.type == "interrupt_noop"


@pytest.mark.asyncio
async def test_interrupt_404_for_unknown():
    from fastapi import HTTPException

    from api.routes.chat_control import interrupt_analysis

    config = ChatConfig(max_concurrent_tasks=5)
    BackgroundTaskManager.get_instance(config)
    with pytest.raises(HTTPException) as exc_info:
        await interrupt_analysis("nonexistent")
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_steer_queued(state_mgr):
    from api.routes.chat_control import SteerRequest, steer_agent

    config = ChatConfig(max_concurrent_tasks=5)
    mgr = BackgroundTaskManager.get_instance(config)
    bus = _make_bus("s1", state_mgr)
    mgr.create("s1", "BTC/USDT", _long_coro(), "chat", bus)

    req = SteerRequest(target="tech_agent", instruction="Focus on RSI divergence")
    resp = await steer_agent("s1", req)
    assert resp.type == "steer_queued"
    assert resp.target == "tech_agent"
    assert resp.queue_position >= 1


@pytest.mark.asyncio
async def test_steer_invalid_agent(state_mgr):
    from fastapi import HTTPException

    from api.routes.chat_control import SteerRequest, steer_agent

    config = ChatConfig(max_concurrent_tasks=5)
    mgr = BackgroundTaskManager.get_instance(config)
    bus = _make_bus("s1", state_mgr)
    mgr.create("s1", "BTC/USDT", _long_coro(), "chat", bus)

    req = SteerRequest(target="invalid_agent", instruction="test")
    with pytest.raises(HTTPException) as exc_info:
        await steer_agent("s1", req)
    assert exc_info.value.status_code == 422


@pytest.mark.asyncio
async def test_steer_too_late(state_mgr):
    from api.routes.chat_control import SteerRequest, steer_agent

    config = ChatConfig(max_concurrent_tasks=5)
    mgr = BackgroundTaskManager.get_instance(config)
    bus = _make_bus("s1", state_mgr)
    task = mgr.create("s1", "BTC/USDT", _long_coro(), "chat", bus)
    task.completed_agents.append("tech_agent")

    req = SteerRequest(target="tech_agent", instruction="too late")
    resp = await steer_agent("s1", req)
    assert resp.type == "steer_too_late"

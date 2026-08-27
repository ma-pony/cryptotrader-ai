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


async def _long_coro(_interrupt_event):
    await asyncio.sleep(10)


@pytest.mark.asyncio
async def test_interrupt_returns_received(state_mgr):
    from api.routes.chat_control import interrupt_analysis

    config = ChatConfig(max_concurrent_tasks=5)
    mgr = BackgroundTaskManager.get_instance(config)
    bus = _make_bus("s1", state_mgr)
    mgr.create("s1", "BTC/USDT", _long_coro, "chat", bus)

    resp = await interrupt_analysis("s1")
    assert resp.type == "interrupt_received"
    assert resp.session_id == "s1"


@pytest.mark.asyncio
async def test_interrupt_noop_when_already_interrupted(state_mgr):
    from api.routes.chat_control import interrupt_analysis

    config = ChatConfig(max_concurrent_tasks=5)
    mgr = BackgroundTaskManager.get_instance(config)
    bus = _make_bus("s1", state_mgr)
    mgr.create("s1", "BTC/USDT", _long_coro, "chat", bus)

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

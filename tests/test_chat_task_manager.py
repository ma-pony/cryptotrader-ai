"""Tests for BackgroundTaskManager — create, interrupt, concurrency, cleanup."""

from __future__ import annotations

import asyncio

import pytest

from cryptotrader.chat.event_buffer import EventBuffer
from cryptotrader.chat.event_bus import EventBus
from cryptotrader.chat.task_manager import BackgroundTaskManager, TooManyTasksError
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


async def _noop_coro():
    await asyncio.sleep(0.01)


async def _long_coro():
    await asyncio.sleep(10)


@pytest.mark.asyncio
async def test_create_and_get(state_mgr):
    config = ChatConfig(max_concurrent_tasks=5)
    mgr = BackgroundTaskManager.get_instance(config)
    bus = _make_bus("s1", state_mgr)
    task = mgr.create("s1", "BTC/USDT", _noop_coro(), "chat", bus)
    assert task.session_id == "s1"
    assert task.pair == "BTC/USDT"
    assert mgr.get("s1") is task


@pytest.mark.asyncio
async def test_concurrent_limit(state_mgr):
    config = ChatConfig(max_concurrent_tasks=2)
    mgr = BackgroundTaskManager.get_instance(config)
    mgr.create("s1", "BTC/USDT", _long_coro(), "chat", _make_bus("s1", state_mgr))
    mgr.create("s2", "ETH/USDT", _long_coro(), "chat", _make_bus("s2", state_mgr))
    with pytest.raises(TooManyTasksError):
        mgr.create("s3", "SOL/USDT", _long_coro(), "chat", _make_bus("s3", state_mgr))


@pytest.mark.asyncio
async def test_session_replacement(state_mgr):
    config = ChatConfig(max_concurrent_tasks=5)
    mgr = BackgroundTaskManager.get_instance(config)
    bus1 = _make_bus("s1", state_mgr)
    task1 = mgr.create("s1", "BTC/USDT", _long_coro(), "chat", bus1)
    bus2 = _make_bus("s1", state_mgr)
    task2 = mgr.create("s1", "BTC/USDT", _long_coro(), "chat", bus2)
    assert task1.interrupt_event.is_set()
    assert mgr.get("s1") is task2


@pytest.mark.asyncio
async def test_interrupt(state_mgr):
    config = ChatConfig(max_concurrent_tasks=5)
    mgr = BackgroundTaskManager.get_instance(config)
    bus = _make_bus("s1", state_mgr)
    task = mgr.create("s1", "BTC/USDT", _long_coro(), "chat", bus)
    assert not task.interrupt_event.is_set()
    assert mgr.interrupt("s1")
    assert task.interrupt_event.is_set()
    assert not mgr.interrupt("s1")


@pytest.mark.asyncio
async def test_task_done_marks_completed(state_mgr):
    config = ChatConfig(max_concurrent_tasks=5)
    mgr = BackgroundTaskManager.get_instance(config)
    bus = _make_bus("s1", state_mgr)
    mgr.create("s1", "BTC/USDT", _noop_coro(), "chat", bus)
    await asyncio.sleep(0.05)
    task = mgr.get("s1")
    assert task is not None
    assert task.completed


@pytest.mark.asyncio
async def test_get_nonexistent(state_mgr):
    config = ChatConfig(max_concurrent_tasks=5)
    mgr = BackgroundTaskManager.get_instance(config)
    assert mgr.get("nonexistent") is None


@pytest.mark.asyncio
async def test_interrupt_nonexistent(state_mgr):
    config = ChatConfig(max_concurrent_tasks=5)
    mgr = BackgroundTaskManager.get_instance(config)
    assert not mgr.interrupt("nonexistent")

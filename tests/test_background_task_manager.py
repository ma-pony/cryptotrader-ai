"""Canonical background-task admission, interruption, concurrency and cleanup."""

from __future__ import annotations

import asyncio
import json

import pytest

from cryptotrader.tasks import BackgroundTaskManager, ExecutionInProgressError, TooManyTasksError


@pytest.fixture(autouse=True)
def _reset_singleton():
    BackgroundTaskManager.reset()
    yield
    BackgroundTaskManager.reset()


async def _noop(_interrupt):
    await asyncio.sleep(0.01)


async def _long(_interrupt):
    await asyncio.sleep(10)


@pytest.mark.asyncio
async def test_shutdown_rejects_admission_while_execution_finishes():
    manager = BackgroundTaskManager()
    entered, release = asyncio.Event(), asyncio.Event()

    async def work(_interrupt):
        entered.set()
        await release.wait()

    owned = manager.create("owned", "BTC/USDT", work, "manual")
    await entered.wait()
    owned.orders_started = True
    shutdown = asyncio.create_task(manager.shutdown())
    await asyncio.sleep(0)
    try:
        assert not shutdown.done()
        with pytest.raises(RuntimeError, match="closed"):
            manager.create("late", "BTC/USDT", _noop, "manual")
    finally:
        release.set()
        await shutdown
    with pytest.raises(RuntimeError, match="closed"):
        manager.create("after", "BTC/USDT", _noop, "manual")


@pytest.mark.asyncio
async def test_create_get_and_completion():
    manager = BackgroundTaskManager.get_instance()
    task = manager.create("one", "BTC/USDT", _noop, "manual")
    assert manager.get("one") is task
    await task.task
    await asyncio.sleep(0)
    assert task.completed


@pytest.mark.asyncio
async def test_concurrency_limit():
    manager = BackgroundTaskManager.get_instance(max_concurrent_tasks=2)
    first = manager.create("one", "BTC/USDT", _long, "manual")
    second = manager.create("two", "ETH/USDT", _long, "manual")
    with pytest.raises(TooManyTasksError):
        manager.create("three", "SOL/USDT", _long, "manual")
    first.task.cancel()
    second.task.cancel()
    await asyncio.gather(first.task, second.task, return_exceptions=True)


@pytest.mark.asyncio
async def test_replacement_cannot_cancel_started_execution():
    manager = BackgroundTaskManager.get_instance()
    old = manager.create("one", "BTC/USDT", _long, "manual")
    old.orders_started = True
    with pytest.raises(ExecutionInProgressError):
        manager.create("one", "BTC/USDT", _long, "manual")
    assert manager.get("one") is old
    old.task.cancel()
    await asyncio.gather(old.task, return_exceptions=True)


@pytest.mark.asyncio
async def test_replaced_callback_cannot_complete_new_task():
    manager = BackgroundTaskManager.get_instance()
    entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def old_work(_interrupt):
        entered.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cleaning.set()
            await release.wait()
            raise

    old = manager.create("one", "BTC/USDT", old_work, "manual")
    await entered.wait()
    new = manager.create("one", "BTC/USDT", _long, "manual")
    await cleaning.wait()
    release.set()
    await asyncio.gather(old.task, return_exceptions=True)
    await asyncio.sleep(0)
    assert manager.get("one") is new
    assert not new.completed
    new.task.cancel()
    await asyncio.gather(new.task, return_exceptions=True)


@pytest.mark.asyncio
async def test_interrupt_is_idempotent_and_missing_is_false():
    manager = BackgroundTaskManager.get_instance()
    task = manager.create("one", "BTC/USDT", _long, "manual")
    assert manager.interrupt("one") is task
    assert manager.interrupt("one") is None
    assert manager.interrupt("missing") is None
    await asyncio.gather(task.task, return_exceptions=True)


@pytest.mark.asyncio
async def test_workflow_publisher_receives_canonical_source():
    published = []

    async def publisher(channel, payload):
        published.append((channel, json.loads(payload)))

    manager = BackgroundTaskManager.get_instance(workflow_publisher=publisher)
    task = manager.create("one", "BTC/USDT", _noop, "manual")
    await task.task
    await asyncio.sleep(0)
    assert published == [
        (
            "analysis:new_workflow",
            {"session_id": "one", "pair": "BTC/USDT", "trigger_source": "manual"},
        )
    ]

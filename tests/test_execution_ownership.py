"""Cancellation-safe lifecycle ownership waiter contract."""

from __future__ import annotations

import asyncio

import pytest

from cryptotrader.execution_ownership import wait_for_owned


@pytest.mark.asyncio
async def test_owned_waiter_defers_external_cancel_until_child_terminal() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    child_cancelled = False

    async def child() -> str:
        nonlocal child_cancelled
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            child_cancelled = True
            raise
        return "terminal"

    child_task = asyncio.create_task(child())
    owner = asyncio.create_task(wait_for_owned(child_task))
    await entered.wait()
    owner.cancel()
    await asyncio.sleep(0)

    assert not owner.done()
    assert child_cancelled is False

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await owner
    assert child_task.result() == "terminal"


@pytest.mark.asyncio
async def test_owned_waiter_absorbs_repeated_cancel_without_cancelling_child() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def child() -> None:
        entered.set()
        await release.wait()

    child_task = asyncio.create_task(child())
    owner = asyncio.create_task(wait_for_owned(child_task))
    await entered.wait()
    owner.cancel()
    await asyncio.sleep(0)
    owner.cancel()
    await asyncio.sleep(0)

    assert child_task.cancelled() is False
    assert owner.done() is False

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await owner
    assert child_task.cancelled() is False


@pytest.mark.asyncio
async def test_child_ordinary_failure_takes_priority_over_external_cancel() -> None:
    release = asyncio.Event()

    async def child() -> None:
        await release.wait()
        raise RuntimeError("owned failure")

    owner = asyncio.create_task(wait_for_owned(asyncio.create_task(child())))
    await asyncio.sleep(0)
    owner.cancel()
    release.set()

    with pytest.raises(RuntimeError, match="owned failure"):
        await owner


@pytest.mark.asyncio
async def test_child_control_flow_takes_priority_over_external_cancel() -> None:
    release = asyncio.Event()
    child_cancellation = asyncio.CancelledError("child control")

    async def child() -> None:
        await release.wait()
        raise child_cancellation

    owner = asyncio.create_task(wait_for_owned(asyncio.create_task(child())))
    await asyncio.sleep(0)
    owner.cancel()
    release.set()

    with pytest.raises(asyncio.CancelledError) as error:
        await owner
    assert error.value is child_cancellation

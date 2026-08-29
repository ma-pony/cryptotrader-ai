"""Cancellation-safe waiting for lifecycle-owned asynchronous work."""

from __future__ import annotations

import asyncio
from typing import TypeVar

T = TypeVar("T")


async def wait_for_owned(awaitable) -> T:
    """Wait for owned work to terminate before propagating caller cancellation."""

    completion = asyncio.ensure_future(awaitable)
    external_cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            result = await asyncio.shield(completion)
            break
        except asyncio.CancelledError as error:
            current = asyncio.current_task()
            if current is None or current.cancelling() == 0:
                return completion.result()
            if external_cancellation is None:
                external_cancellation = error
            if completion.done():
                result = completion.result()
                break
    if external_cancellation is not None:
        raise external_cancellation
    return result

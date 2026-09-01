"""Cancellation-safe waiting for lifecycle-owned asynchronous work."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TypeVar

T = TypeVar("T")


class ExecutionOwnership:
    """Strict Redis pool ownership; always enter after Runtime's application read lease."""

    def __init__(self, redis_url: str):
        self.redis_url = redis_url

    @asynccontextmanager
    async def book(self, book_id: str):
        from cryptotrader.cycle_lock import execution_book_lease

        if not isinstance(book_id, str) or not book_id.strip():
            raise ValueError("book_id is required")
        async with execution_book_lease(self.redis_url, book_id):
            yield

    @asynccontextmanager
    async def connection(self, connection_id: str):
        """For unassigned accounts only; assigned accounts must use their book owner."""
        from cryptotrader.cycle_lock import execution_connection_lease

        if not isinstance(connection_id, str) or not connection_id.strip():
            raise ValueError("connection_id is required")
        async with execution_connection_lease(self.redis_url, connection_id):
            yield


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

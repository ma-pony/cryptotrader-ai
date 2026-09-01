"""Pool-scoped admission inside the Runtime application lifecycle.

Strict Redis primitives have no in-memory fallback. A pool owner retains its
lease across full account refresh, risk validation and order finalization.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from time import monotonic
from typing import TYPE_CHECKING

from cryptotrader.execution_ownership import wait_for_owned
from cryptotrader.risk.state import RedisStateManager

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)


# The existing cycle time budget is 300s, plus 60s for finalization.
DEFAULT_CYCLE_LOCK_TTL = 360


class ExecutionLeaseUnavailableError(RuntimeError):
    """Redis pool admission failed or its bounded wait expired."""


@asynccontextmanager
async def cycle_lock(
    redis_state: RedisStateManager,
    pair: str,
    *,
    ttl: int = DEFAULT_CYCLE_LOCK_TTL,
) -> AsyncIterator[bool]:
    """Acquire ``cycle_lock:{pair}`` for the duration of the ``async with`` block.

    Yields ``True`` when the lock was acquired (caller should proceed) and
    ``False`` when another holder owns it (caller should skip the cycle and
    log). Always releases on exit, even on exception, but only if we still
    own the key — protects against the prior-holder-expired-then-re-acquired
    race.

    The owner identifier is unique to this lease and release is owner-checked.
    """
    key = f"cycle_lock:{pair}"
    owner_id = f"{os.getpid()}:{uuid.uuid4().hex}"

    acquired = await redis_state.try_acquire_strict_lock(key, owner_id, ttl)
    try:
        yield acquired
    finally:
        if acquired:
            try:
                await wait_for_owned(redis_state.release_strict_lock(key, owner_id))
            except Exception:
                logger.info("cycle_lock release failed for %s", key, exc_info=True)


@asynccontextmanager
async def execution_book_lease(redis_url: str, book_id: str, *, wait_timeout: float = 300) -> AsyncIterator[None]:
    """Acquire a strict Redis lease for the pool, independent of requested pair."""
    async with _execution_lease(redis_url, f"book:{book_id}", wait_timeout=wait_timeout):
        yield


@asynccontextmanager
async def execution_connection_lease(redis_url: str, connection_id: str) -> AsyncIterator[None]:
    """Serialize writes to an explicitly unassigned account (canary / manual exit)."""
    async with _execution_lease(redis_url, f"connection:{connection_id}"):
        yield


@asynccontextmanager
async def _execution_lease(redis_url: str, scope_key: str, *, wait_timeout: float = 300) -> AsyncIterator[None]:
    if not redis_url.strip():
        raise ExecutionLeaseUnavailableError("Redis is required for production execution lease")
    redis_state = RedisStateManager(redis_url)
    lease = cycle_lock(redis_state, scope_key)
    entered = False
    deadline = monotonic() + wait_timeout

    async def finalize() -> None:
        try:
            if entered:
                await lease.__aexit__(None, None, None)
        finally:
            await redis_state.aclose()

    try:
        try:
            while True:
                acquired = await lease.__aenter__()
                entered = True
                if acquired:
                    break
                await lease.__aexit__(None, None, None)
                entered = False
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise ExecutionLeaseUnavailableError(f"execution lease wait timed out for {scope_key}")
                await asyncio.sleep(min(0.05, remaining))
                lease = cycle_lock(redis_state, scope_key)
        except RuntimeError as error:
            raise ExecutionLeaseUnavailableError(str(error)) from error
        yield
    finally:
        await wait_for_owned(finalize())

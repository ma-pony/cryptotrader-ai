"""Pair-scoped cycle admission using the caller-provided Redis state manager.

Production order-bearing work enters this primitive only through
``Runtime.execution_lease``. Its strict Redis primitives have no in-memory
fallback, so degraded cross-process execution is never admitted.
"""

from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from cryptotrader.risk.state import RedisStateManager

logger = logging.getLogger(__name__)


# TTL: cycle_timeout_s caps a real cycle at 300s (then +60s margin for fee /
# OKX retry tail). 200s would be too aggressive; 360s leaves 1× headroom and
# bounds worst-case stale-lock wait to 6 minutes when the PID-alive check
# misses (e.g. permission-denied path).  Combined with stale-PID stealing
# (spec 021 E1) this is now belt-and-suspenders.
DEFAULT_CYCLE_LOCK_TTL = 360


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
                await redis_state.release_strict_lock(key, owner_id)
            except Exception:
                logger.info("cycle_lock release failed for %s", key, exc_info=True)

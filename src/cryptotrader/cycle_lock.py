"""Per-pair cycle lock to prevent concurrent runs on the same trading pair.

Production observation (2026-05-02): a manual ``arena run`` started while
the launchd scheduler was processing a freshly-restarted cycle produced two
ETH/USDT close decisions 426 ms apart, with the same pair entering the OKX
order pipeline twice. Only one filled (the second saw flat balance), but a
``long`` verdict in the same race would have doubled exposure — a single
restart-plus-manual-run window can't be assumed safe.

The lock is keyed by pair (``cycle_lock:<pair>``) with a TTL slightly longer
than the longest expected cycle. If Redis is unreachable, ``RedisStateManager``
falls back to in-process memory — single-process safety only, but the
launchd scheduler is single-process and ``arena run`` shares its memory
neither, so the practical guarantee in that mode degrades gracefully:
contending callers in the same process serialize, cross-process callers
do not (matches existing risk/state.py degraded-mode behavior).
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from cryptotrader.risk.state import RedisStateManager

logger = logging.getLogger(__name__)


# Default TTL: the launchd scheduler enforces graph_timeout_s (default 300),
# arena run does not but typically completes in <120s. 600s leaves headroom
# without leaving a stale lock on the floor for too long if a process crashes
# mid-cycle without running its release.
DEFAULT_CYCLE_LOCK_TTL = 600


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
    """
    key = f"cycle_lock:{pair}"
    owner_id = uuid.uuid4().hex

    acquired = await redis_state.try_acquire_lock(key, owner_id, ttl)
    try:
        yield acquired
    finally:
        if acquired:
            try:
                await redis_state.release_lock(key, owner_id)
            except Exception:
                logger.info("cycle_lock release failed for %s", key, exc_info=True)

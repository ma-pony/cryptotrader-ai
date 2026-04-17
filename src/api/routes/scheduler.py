"""Scheduler status endpoints.

Two routes are exposed:

- ``GET /scheduler/status`` — legacy public endpoint with APScheduler-flavored
  payload (``running``/``jobs``/``cycle_count``/…).
- ``GET /api/scheduler/status`` — contract endpoint (FR-802) returning the
  data-model shape consumed by the React Dashboard:
  ``{enabled, next_pair, next_run_at, redis_available}``.

When the scheduler is not running (API-only deployment), both routes
degrade gracefully (``running=False`` / ``next_*=null``) instead of 503.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

from fastapi import APIRouter, Request
from pydantic import BaseModel

if TYPE_CHECKING:
    from cryptotrader.scheduler import Scheduler

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/scheduler")
api_router = APIRouter(prefix="/api/scheduler", tags=["scheduler"])


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class SchedulerJobStatus(BaseModel):
    """Status of a single APScheduler job."""

    job_id: str
    name: str
    next_run_time: datetime | None
    pairs: list[str]


class SchedulerStatusResponse(BaseModel):
    """Full scheduler status response."""

    running: bool
    jobs: list[SchedulerJobStatus]
    cycle_count: int
    interval_minutes: int
    pairs: list[str]


# ---------------------------------------------------------------------------
# Internal helper — allows tests to override via patch
# ---------------------------------------------------------------------------


def _get_scheduler(request: Request) -> Scheduler | None:
    """Return the Scheduler instance from app.state, or None if not registered."""
    return getattr(request.app.state, "scheduler", None)


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.get("/status", response_model=SchedulerStatusResponse)
async def scheduler_status(request: Request) -> SchedulerStatusResponse:
    """Return the current scheduler status.

    When the scheduler is not started (API-only deployment), returns
    ``running=false`` with empty jobs rather than 503, so the Dashboard
    can degrade gracefully.
    """
    scheduler = _get_scheduler(request)

    if scheduler is None or not scheduler._scheduler.running:
        return SchedulerStatusResponse(
            running=False,
            jobs=[],
            cycle_count=0,
            interval_minutes=scheduler.interval_minutes if scheduler else 240,
            pairs=scheduler.pairs if scheduler else [],
        )

    # Scheduler is running — collect live job data
    raw_jobs = scheduler.jobs  # list[dict] from Scheduler.jobs property
    job_statuses: list[SchedulerJobStatus] = []
    for raw in raw_jobs:
        next_run_raw = raw.get("next_run_time")
        if isinstance(next_run_raw, str):
            try:
                next_run: datetime | None = datetime.fromisoformat(next_run_raw)
            except ValueError:
                logger.debug("Cannot parse next_run_time %r", next_run_raw)
                next_run = None
        elif isinstance(next_run_raw, datetime):
            next_run = next_run_raw
        else:
            next_run = None

        job_statuses.append(
            SchedulerJobStatus(
                job_id=raw.get("id", ""),
                name=raw.get("name", ""),
                next_run_time=next_run,
                pairs=scheduler.pairs,
            )
        )

    return SchedulerStatusResponse(
        running=True,
        jobs=job_statuses,
        cycle_count=scheduler._cycle_count,
        interval_minutes=scheduler.interval_minutes,
        pairs=scheduler.pairs,
    )


# ---------------------------------------------------------------------------
# Contract endpoint — /api/scheduler/status (FR-802)
# ---------------------------------------------------------------------------


class SchedulerContractStatus(BaseModel):
    """Data-model §2 SchedulerStatus shape for the React Dashboard."""

    enabled: bool
    next_pair: str | None
    next_run_at: datetime | None
    redis_available: bool


def _next_trading_run(scheduler: Scheduler | None) -> tuple[str | None, datetime | None]:
    """Return ``(next_pair, next_run_at)`` from the live scheduler, or (None, None)."""
    if scheduler is None:
        return (None, None)

    next_pair = scheduler.pairs[0] if scheduler.pairs else None

    next_run: datetime | None = None
    for raw in scheduler.jobs:
        if raw.get("id") != "trading_cycle":
            continue
        raw_ts = raw.get("next_run_time")
        if isinstance(raw_ts, datetime):
            next_run = raw_ts
        elif isinstance(raw_ts, str):
            try:
                next_run = datetime.fromisoformat(raw_ts)
            except ValueError:
                logger.debug("Cannot parse trading_cycle next_run_time %r", raw_ts)
        break
    return (next_pair, next_run)


@api_router.get("/status", response_model=SchedulerContractStatus)
async def scheduler_status_v2(request: Request) -> SchedulerContractStatus:
    """Return scheduler status in the data-model contract shape (FR-802)."""
    from cryptotrader.config import load_config
    from cryptotrader.risk.state import RedisStateManager

    config = load_config()
    rsm = RedisStateManager(config.infrastructure.redis_url)

    scheduler = _get_scheduler(request)
    next_pair, next_run_at = _next_trading_run(scheduler)

    return SchedulerContractStatus(
        enabled=bool(getattr(config.scheduler, "enabled", False)),
        next_pair=next_pair,
        next_run_at=next_run_at,
        redis_available=bool(rsm.available),
    )

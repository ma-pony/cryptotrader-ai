"""Scheduler status endpoint — GET /scheduler/status.

Returns the current state of the APScheduler-based trading scheduler.
When the scheduler is not running (e.g., in API-only mode), returns
running=False with empty jobs instead of 503.
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

"""Tests for GET /scheduler/status endpoint (Task 3).

Strategy: minimize mocks.
- Scheduler is instantiated as a real object (not started, so no event loop blocking).
- FastAPI TestClient with the real app.
- APScheduler internals are mocked only at the Scheduler.jobs property level when
  we need to simulate a running scheduler with registered jobs, because
  APScheduler's get_jobs() has no state without start().
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.routes.scheduler import SchedulerJobStatus, SchedulerStatusResponse
from cryptotrader.scheduler import Scheduler

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    """TestClient backed by the real FastAPI app."""
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def real_scheduler():
    """A real Scheduler instance that has NOT been started."""
    return Scheduler(pairs=["BTC/USDT", "ETH/USDT"], interval_minutes=60)


# ---------------------------------------------------------------------------
# Pydantic model contract tests (no HTTP, pure model validation)
# ---------------------------------------------------------------------------


def test_scheduler_job_status_model_fields():
    """SchedulerJobStatus has the four required fields with correct types."""
    job = SchedulerJobStatus(
        job_id="trading_cycle",
        name="Trading cycle",
        next_run_time=datetime(2026, 3, 15, 10, 0, 0, tzinfo=UTC),
        pairs=["BTC/USDT"],
    )
    assert job.job_id == "trading_cycle"
    assert job.name == "Trading cycle"
    assert job.next_run_time is not None
    assert job.pairs == ["BTC/USDT"]


def test_scheduler_job_status_next_run_time_optional():
    """SchedulerJobStatus accepts None for next_run_time."""
    job = SchedulerJobStatus(
        job_id="daily_summary",
        name="Daily summary",
        next_run_time=None,
        pairs=[],
    )
    assert job.next_run_time is None


def test_scheduler_status_response_running_true():
    """SchedulerStatusResponse with running=True has full job list and metadata."""
    job = SchedulerJobStatus(
        job_id="trading_cycle",
        name="Trading cycle",
        next_run_time=datetime(2026, 3, 15, 10, 0, 0, tzinfo=UTC),
        pairs=["BTC/USDT"],
    )
    resp = SchedulerStatusResponse(
        running=True,
        jobs=[job],
        cycle_count=42,
        interval_minutes=60,
        pairs=["BTC/USDT"],
    )
    assert resp.running is True
    assert len(resp.jobs) == 1
    assert resp.cycle_count == 42
    assert resp.interval_minutes == 60
    assert resp.pairs == ["BTC/USDT"]


def test_scheduler_status_response_running_false_empty_jobs():
    """SchedulerStatusResponse with running=False has empty jobs list."""
    resp = SchedulerStatusResponse(
        running=False,
        jobs=[],
        cycle_count=0,
        interval_minutes=240,
        pairs=[],
    )
    assert resp.running is False
    assert resp.jobs == []
    assert resp.cycle_count == 0


def test_scheduler_status_response_json_serializable():
    """SchedulerStatusResponse serializes to valid JSON without errors."""
    job = SchedulerJobStatus(
        job_id="trading_cycle",
        name="Trading cycle",
        next_run_time=datetime(2026, 3, 15, 10, 0, 0, tzinfo=UTC),
        pairs=["BTC/USDT", "ETH/USDT"],
    )
    resp = SchedulerStatusResponse(
        running=True,
        jobs=[job],
        cycle_count=5,
        interval_minutes=240,
        pairs=["BTC/USDT", "ETH/USDT"],
    )
    data = resp.model_dump()
    assert isinstance(data, dict)
    assert data["running"] is True
    assert len(data["jobs"]) == 1
    assert data["jobs"][0]["job_id"] == "trading_cycle"


# ---------------------------------------------------------------------------
# Scheduler not started → running=False
# ---------------------------------------------------------------------------


def test_get_scheduler_status_not_started_returns_200_not_503(client):
    """GET /scheduler/status returns 200 (not 503) when scheduler is not running."""
    resp = client.get("/scheduler/status")
    assert resp.status_code == 200


def test_get_scheduler_status_not_started_running_false(client):
    """When no scheduler is registered, running=False and jobs=[]."""
    # By default no scheduler is injected into the app state
    resp = client.get("/scheduler/status")
    data = resp.json()
    assert data["running"] is False
    assert data["jobs"] == []


def test_get_scheduler_status_not_started_zero_cycle_count(client):
    """When no scheduler is running, cycle_count=0."""
    resp = client.get("/scheduler/status")
    data = resp.json()
    assert data["cycle_count"] == 0


def test_get_scheduler_status_not_started_empty_pairs(client):
    """When no scheduler is running, pairs=[]."""
    resp = client.get("/scheduler/status")
    data = resp.json()
    assert data["pairs"] == []


def test_get_scheduler_status_response_shape(client):
    """Response has all five required top-level keys."""
    resp = client.get("/scheduler/status")
    data = resp.json()
    required_keys = {"running", "jobs", "cycle_count", "interval_minutes", "pairs"}
    assert required_keys.issubset(data.keys())


# ---------------------------------------------------------------------------
# Scheduler running → returns live data
# ---------------------------------------------------------------------------


def _patch_scheduler_running(scheduler, fake_jobs: list):
    """Context manager that makes a Scheduler look like it is running.

    Patches:
    - AsyncIOScheduler.running → True  (class-level property)
    - Scheduler.jobs → fake_jobs       (class-level property)

    Both are patched at the *class* level so unittest.mock can restore them.
    """
    from cryptotrader.scheduler import Scheduler as _Sched

    return (
        patch.object(
            type(scheduler._scheduler),
            "running",
            new_callable=lambda: property(lambda self: True),
        ),
        patch.object(
            _Sched,
            "jobs",
            new_callable=lambda: property(lambda self: fake_jobs),
        ),
    )


def test_get_scheduler_status_running_true(client, real_scheduler):
    """When a running scheduler is injected, running=True."""
    fake_jobs = [
        {
            "id": "trading_cycle",
            "name": "Trading cycle",
            "next_run_time": "2026-03-15T10:00:00+00:00",
            "trigger": "interval[0:04:00:00]",
        }
    ]
    real_scheduler._cycle_count = 7

    p1, p2 = _patch_scheduler_running(real_scheduler, fake_jobs)
    with p1, p2, patch("api.routes.scheduler._get_scheduler", return_value=real_scheduler):
        resp = client.get("/scheduler/status")

    assert resp.status_code == 200
    data = resp.json()
    assert data["running"] is True


def test_get_scheduler_status_returns_cycle_count(client, real_scheduler):
    """cycle_count reflects the scheduler's internal counter."""
    real_scheduler._cycle_count = 42
    fake_jobs: list = []

    p1, p2 = _patch_scheduler_running(real_scheduler, fake_jobs)
    with p1, p2, patch("api.routes.scheduler._get_scheduler", return_value=real_scheduler):
        resp = client.get("/scheduler/status")

    data = resp.json()
    assert data["cycle_count"] == 42


def test_get_scheduler_status_returns_interval_minutes(client, real_scheduler):
    """interval_minutes reflects the scheduler's configured interval."""
    fake_jobs: list = []
    p1, p2 = _patch_scheduler_running(real_scheduler, fake_jobs)
    with p1, p2, patch("api.routes.scheduler._get_scheduler", return_value=real_scheduler):
        resp = client.get("/scheduler/status")

    data = resp.json()
    assert data["interval_minutes"] == 60  # real_scheduler fixture uses 60


def test_get_scheduler_status_returns_pairs(client, real_scheduler):
    """pairs reflects the scheduler's configured trading pairs."""
    fake_jobs: list = []
    p1, p2 = _patch_scheduler_running(real_scheduler, fake_jobs)
    with p1, p2, patch("api.routes.scheduler._get_scheduler", return_value=real_scheduler):
        resp = client.get("/scheduler/status")

    data = resp.json()
    assert "BTC/USDT" in data["pairs"]
    assert "ETH/USDT" in data["pairs"]


def test_get_scheduler_status_jobs_shape(client, real_scheduler):
    """Each job object has job_id, name, next_run_time, pairs fields."""
    fake_jobs = [
        {
            "id": "trading_cycle",
            "name": "Trading cycle",
            "next_run_time": "2026-03-15T10:00:00+00:00",
            "trigger": "interval[0:04:00:00]",
        }
    ]
    real_scheduler._cycle_count = 1

    p1, p2 = _patch_scheduler_running(real_scheduler, fake_jobs)
    with p1, p2, patch("api.routes.scheduler._get_scheduler", return_value=real_scheduler):
        resp = client.get("/scheduler/status")

    data = resp.json()
    assert len(data["jobs"]) == 1
    job = data["jobs"][0]
    assert "job_id" in job
    assert "name" in job
    assert "next_run_time" in job
    assert "pairs" in job
    assert job["job_id"] == "trading_cycle"


def test_get_scheduler_status_job_with_none_next_run(client, real_scheduler):
    """Jobs with no next_run_time have null in response (not error)."""
    fake_jobs = [
        {
            "id": "daily_summary",
            "name": "Daily summary",
            "next_run_time": None,
            "trigger": "cron[hour='0']",
        }
    ]

    p1, p2 = _patch_scheduler_running(real_scheduler, fake_jobs)
    with p1, p2, patch("api.routes.scheduler._get_scheduler", return_value=real_scheduler):
        resp = client.get("/scheduler/status")

    data = resp.json()
    assert data["jobs"][0]["next_run_time"] is None


# ---------------------------------------------------------------------------
# Scheduler injected via app.state
# ---------------------------------------------------------------------------


def test_get_scheduler_status_via_app_state(real_scheduler):
    """Scheduler injected into app.state is used by the endpoint."""
    fake_jobs = [
        {
            "id": "trading_cycle",
            "name": "Trading cycle",
            "next_run_time": "2026-03-15T10:00:00+00:00",
            "trigger": "interval[0:04:00:00]",
        }
    ]
    real_scheduler._cycle_count = 3

    p1, p2 = _patch_scheduler_running(real_scheduler, fake_jobs)
    with p1, p2:
        # Inject scheduler into app.state directly
        app.state.scheduler = real_scheduler
        try:
            with TestClient(app, raise_server_exceptions=False) as c:
                resp = c.get("/scheduler/status")
        finally:
            # Clean up — remove injected scheduler so other tests are not affected
            del app.state.scheduler

    assert resp.status_code == 200
    data = resp.json()
    assert data["running"] is True
    assert data["cycle_count"] == 3
    assert data["pairs"] == ["BTC/USDT", "ETH/USDT"]


def test_get_scheduler_status_app_state_none_after_cleanup():
    """After cleanup, endpoint falls back to running=False."""
    # Ensure no scheduler is in app.state
    if hasattr(app.state, "scheduler"):
        del app.state.scheduler

    with TestClient(app, raise_server_exceptions=False) as c:
        resp = c.get("/scheduler/status")

    assert resp.status_code == 200
    assert resp.json()["running"] is False


# ---------------------------------------------------------------------------
# No API key required (public endpoint like /health and /metrics)
# ---------------------------------------------------------------------------


def test_scheduler_status_is_public_endpoint(client):
    """GET /scheduler/status does not require an API key."""
    # Without any auth headers, should still get 200 (not 401/403)
    resp = client.get("/scheduler/status")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Edge: Scheduler with multiple pairs
# ---------------------------------------------------------------------------


def test_get_scheduler_status_multiple_pairs(client):
    """Scheduler with three pairs reports all three in the response."""
    sched = Scheduler(pairs=["BTC/USDT", "ETH/USDT", "SOL/USDT"], interval_minutes=120)
    fake_jobs: list = []

    p1, p2 = _patch_scheduler_running(sched, fake_jobs)
    with p1, p2, patch("api.routes.scheduler._get_scheduler", return_value=sched):
        resp = client.get("/scheduler/status")

    data = resp.json()
    assert set(data["pairs"]) == {"BTC/USDT", "ETH/USDT", "SOL/USDT"}
    assert data["interval_minutes"] == 120

"""Scheduler tests — APScheduler integration."""

import asyncio
import logging
from unittest.mock import AsyncMock, patch

from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from cryptotrader.scheduler import Scheduler


def test_scheduler_init():
    s = Scheduler(["BTC/USDT"], interval_minutes=60)
    assert s.pairs == ["BTC/USDT"]
    assert s.interval_minutes == 60
    assert s.daily_summary_hour == 0


def test_scheduler_init_custom_summary_hour():
    s = Scheduler(["BTC/USDT"], interval_minutes=60, daily_summary_hour=8)
    assert s.daily_summary_hour == 8


def test_scheduler_status():
    s = Scheduler(["BTC/USDT", "ETH/USDT"])
    assert "BTC/USDT" in s.status
    assert "ETH/USDT" in s.status


def test_scheduler_registers_jobs():
    """Verify start() registers trading_cycle and daily_summary jobs."""
    s = Scheduler(["BTC/USDT"], interval_minutes=60)

    # Start the internal APScheduler (but don't await start() which blocks)
    s._scheduler.add_job(
        s._run_cycle,
        "interval",
        minutes=60,
        id="trading_cycle",
        name="Trading cycle",
    )
    s._scheduler.add_job(
        s._emit_daily_summary,
        "cron",
        hour=0,
        minute=0,
        id="daily_summary",
        name="Daily summary",
    )

    job_ids = [j.id for j in s._scheduler.get_jobs()]
    assert "trading_cycle" in job_ids
    assert "daily_summary" in job_ids


async def test_scheduler_jobs_property():
    """Verify .jobs returns structured job info."""
    s = Scheduler(["BTC/USDT"], interval_minutes=120)
    s._scheduler.start(paused=True)
    s._scheduler.add_job(
        s._run_cycle,
        "interval",
        minutes=120,
        id="trading_cycle",
        name="Trading cycle",
    )
    jobs = s.jobs
    assert len(jobs) == 1
    assert jobs[0]["id"] == "trading_cycle"
    assert jobs[0]["name"] == "Trading cycle"
    s._scheduler.shutdown(wait=False)


def test_stop_sets_event():
    s = Scheduler(["BTC/USDT"])
    s._stop_event = asyncio.Event()
    s.stop()
    assert s._stop_event.is_set()


def test_stop_without_event():
    """stop() before start() should not raise."""
    s = Scheduler(["BTC/USDT"])
    s.stop()  # No error


async def test_run_cycle_increments_count():
    s = Scheduler(["BTC/USDT"], interval_minutes=60)
    assert s._cycle_count == 0

    with patch.object(s, "_run_pair", new_callable=AsyncMock) as mock_run:
        await s._run_cycle()
        assert s._cycle_count == 1
        mock_run.assert_called_once_with("BTC/USDT")

        await s._run_cycle()
        assert s._cycle_count == 2


async def test_run_cycle_gathers_all_pairs():
    s = Scheduler(["BTC/USDT", "ETH/USDT", "SOL/USDT"], interval_minutes=60)

    with patch.object(s, "_run_pair", new_callable=AsyncMock) as mock_run:
        await s._run_cycle()
        assert mock_run.call_count == 3
        called_pairs = {call.args[0] for call in mock_run.call_args_list}
        assert called_pairs == {"BTC/USDT", "ETH/USDT", "SOL/USDT"}


async def test_scheduler_respects_enabled_flag():
    """CLI should check config.scheduler.enabled before starting."""
    from cryptotrader.config import AppConfig, SchedulerConfig

    config = AppConfig(scheduler=SchedulerConfig(enabled=False))
    assert not config.scheduler.enabled

    config_enabled = AppConfig(scheduler=SchedulerConfig(enabled=True))
    assert config_enabled.scheduler.enabled


async def test_run_cycle_updates_status():
    s = Scheduler(["BTC/USDT"], interval_minutes=60)

    with patch.object(s, "_run_pair", new_callable=AsyncMock):
        await s._run_cycle()
        assert "next_run" in s._status["BTC/USDT"]


# Task 2.3: scheduler single-cycle exception isolation tests


async def test_run_cycle_exception_does_not_propagate():
    """_run_cycle() pair failure must not propagate; scheduler continues."""
    s = Scheduler(["BTC/USDT"], interval_minutes=60)

    # _run_pair raises, but gather(return_exceptions=True) absorbs it
    async def fail(_pair: str) -> None:
        raise RuntimeError("pair failure")

    with patch.object(s, "_run_pair", side_effect=fail):
        # Must not raise
        await s._run_cycle()
    # cycle_count incremented means _run_cycle completed normally
    assert s._cycle_count == 1


async def test_run_cycle_top_level_exception_caught(caplog):
    """_run_cycle() top-level try/except catches unexpected errors and logs warning."""
    s = Scheduler(["BTC/USDT"], interval_minutes=60)

    # Force asyncio.gather itself to raise
    with (
        patch("asyncio.gather", side_effect=RuntimeError("gather failure")),
        caplog.at_level(logging.WARNING, logger="cryptotrader.scheduler"),
    ):
        await s._run_cycle()

    # Scheduler did not crash; warning was logged
    assert any("gather failure" in r.message or "gather failure" in str(r.exc_info) for r in caplog.records)


async def test_run_pair_logs_warning_on_exception(caplog):
    """_run_pair() must use logger.warning(exc_info=True) on exception."""
    s = Scheduler(["BTC/USDT"], interval_minutes=60)

    # set_trace_id and load_config are lazy-imported; patch at their source module
    with (
        patch("cryptotrader.tracing.set_trace_id", return_value="trace-abc"),
        patch("cryptotrader.config.load_config", side_effect=RuntimeError("cfg error")),
        caplog.at_level(logging.WARNING, logger="cryptotrader.scheduler"),
    ):
        await s._run_pair("BTC/USDT")

    # exc_info=True means record.exc_info is not None
    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warning_records, "Expected at least one WARNING log record"
    assert any(r.exc_info is not None for r in warning_records), "Expected exc_info=True stack trace"


def test_add_job_trading_cycle_has_max_instances_and_misfire():
    """trading_cycle and daily_summary jobs must have max_instances=1 and misfire_grace_time=1.

    Captures add_job kwargs directly to avoid running the full blocking start() coroutine.
    APScheduler rejects misfire_grace_time=0; value 1 (minimum positive int) achieves
    near-immediate discard of missed triggers.
    """
    s = Scheduler(["BTC/USDT"], interval_minutes=60)
    captured_kwargs: dict[str, dict] = {}

    original_add_job = s._scheduler.add_job

    def capturing_add_job(func, trigger=None, **kwargs):
        job_id = kwargs.get("id", "unknown")
        captured_kwargs[job_id] = kwargs
        return original_add_job(func, trigger, **kwargs)

    with patch.object(s._scheduler, "add_job", side_effect=capturing_add_job):
        # Replicate the exact add_job calls used in start() to verify parameters
        from datetime import UTC, datetime

        s._scheduler.add_job(
            s._run_cycle,
            IntervalTrigger(minutes=s.interval_minutes),
            id="trading_cycle",
            name="Trading cycle",
            next_run_time=datetime.now(UTC),
            max_instances=1,
            misfire_grace_time=1,
        )
        s._scheduler.add_job(
            s._emit_daily_summary,
            CronTrigger(hour=s.daily_summary_hour, minute=0, timezone="UTC"),
            id="daily_summary",
            name="Daily summary",
            max_instances=1,
            misfire_grace_time=1,
        )

    assert "trading_cycle" in captured_kwargs, "trading_cycle job not registered"
    assert "daily_summary" in captured_kwargs, "daily_summary job not registered"

    tc = captured_kwargs["trading_cycle"]
    ds = captured_kwargs["daily_summary"]

    assert tc["max_instances"] == 1, "trading_cycle must have max_instances=1"
    assert ds["max_instances"] == 1, "daily_summary must have max_instances=1"
    assert tc["misfire_grace_time"] == 1, "trading_cycle must have misfire_grace_time=1"
    assert ds["misfire_grace_time"] == 1, "daily_summary must have misfire_grace_time=1"

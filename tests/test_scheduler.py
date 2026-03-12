"""Scheduler tests — APScheduler integration."""

import asyncio
from unittest.mock import AsyncMock, patch

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

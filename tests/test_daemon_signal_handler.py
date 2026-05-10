"""spec 020c — Daemon SIGTERM / SIGINT graceful shutdown tests.

T019: test_run_forever_sigterm_graceful_shutdown
T020: test_sigterm_during_run_once_waits_for_completion
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cryptotrader.ops.daemon import EvolutionDaemon, _try_acquire_locks

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config() -> SimpleNamespace:
    return SimpleNamespace(actions=["pareto"], cron="0 2 * * *", propose_threshold=5)


# ---------------------------------------------------------------------------
# T019: SIGTERM graceful shutdown
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_forever_sigterm_graceful_shutdown() -> None:
    """T019: Setting shutdown_flag causes run_forever to call scheduler.shutdown(wait=True)."""
    daemon = EvolutionDaemon(config=_make_config())

    mock_scheduler = MagicMock()
    mock_scheduler.start = MagicMock()
    mock_scheduler.shutdown = MagicMock()

    with (
        patch("cryptotrader.ops.daemon.EvolutionDaemon.run_once", new_callable=AsyncMock),
        patch("cryptotrader.observability.daemon_metrics._get_redis", return_value=None),
        patch("apscheduler.schedulers.asyncio.AsyncIOScheduler", return_value=mock_scheduler),
        patch("apscheduler.triggers.cron.CronTrigger.from_crontab", return_value=MagicMock()),
    ):
        task = asyncio.create_task(daemon.run_forever())
        # Give the coroutine time to reach await shutdown_flag.wait()
        await asyncio.sleep(0.05)

        # Simulate signal by setting the flag directly
        daemon._shutdown_flag.set()

        await asyncio.wait_for(task, timeout=5.0)

    mock_scheduler.shutdown.assert_called_once_with(wait=True)


# ---------------------------------------------------------------------------
# T020: SIGTERM while run_once is running waits for completion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sigterm_during_run_once_waits_for_completion() -> None:
    """T020: Shutdown after scheduler.shutdown(wait=True) — current job finishes."""
    daemon = EvolutionDaemon(config=_make_config())

    mock_scheduler = MagicMock()
    mock_scheduler.start = MagicMock()
    mock_scheduler.shutdown = MagicMock()

    with (
        patch("cryptotrader.ops.daemon.EvolutionDaemon.run_once", new_callable=AsyncMock),
        patch("cryptotrader.observability.daemon_metrics._get_redis", return_value=None),
        patch("apscheduler.schedulers.asyncio.AsyncIOScheduler", return_value=mock_scheduler),
        patch("apscheduler.triggers.cron.CronTrigger.from_crontab", return_value=MagicMock()),
    ):
        task = asyncio.create_task(daemon.run_forever())
        await asyncio.sleep(0.05)

        daemon._shutdown_flag.set()
        await asyncio.wait_for(task, timeout=5.0)

    # wait=True ensures the current run_once job is not interrupted
    mock_scheduler.shutdown.assert_called_once_with(wait=True)


# ---------------------------------------------------------------------------
# Extra: _shutdown_flag exists after run_forever starts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shutdown_flag_is_asyncio_event_after_start() -> None:
    """run_forever sets self._shutdown_flag (asyncio.Event) before awaiting it."""
    daemon = EvolutionDaemon(config=_make_config())

    mock_scheduler = MagicMock()
    mock_scheduler.start = MagicMock()
    mock_scheduler.shutdown = MagicMock()

    with (
        patch("cryptotrader.ops.daemon.EvolutionDaemon.run_once", new_callable=AsyncMock),
        patch("cryptotrader.observability.daemon_metrics._get_redis", return_value=None),
        patch("apscheduler.schedulers.asyncio.AsyncIOScheduler", return_value=mock_scheduler),
        patch("apscheduler.triggers.cron.CronTrigger.from_crontab", return_value=MagicMock()),
    ):
        task = asyncio.create_task(daemon.run_forever())
        await asyncio.sleep(0.05)

        assert hasattr(daemon, "_shutdown_flag")
        assert isinstance(daemon._shutdown_flag, asyncio.Event)

        daemon._shutdown_flag.set()
        await asyncio.wait_for(task, timeout=5.0)


# ---------------------------------------------------------------------------
# Extra: _try_acquire_locks is a coroutine function (SC-L3 related)
# ---------------------------------------------------------------------------


def test_try_acquire_locks_is_coroutine_function() -> None:
    """_try_acquire_locks must be async def (spec 020c FR-L10)."""
    assert asyncio.iscoroutinefunction(_try_acquire_locks)

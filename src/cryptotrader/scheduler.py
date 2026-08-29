"""APScheduler-based trading scheduler with interval and cron triggers."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Protocol

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from cryptotrader._compat import UTC
from cryptotrader.execution_ownership import wait_for_owned
from cryptotrader.pair import Pair

if TYPE_CHECKING:
    from contextlib import AbstractAsyncContextManager

    from cryptotrader.runtime_config.models import RuntimeConfigSnapshot, SchedulerConfig
    from cryptotrader.trading_cycle import TradingCycle

logger = logging.getLogger(__name__)
_slog = structlog.get_logger(__name__)


class RuntimeCycleSource(Protocol):
    """Scheduler 所需的唯一可热重载 Runtime 边界。"""

    snapshot: RuntimeConfigSnapshot
    cycle: TradingCycle | None

    def cycle_lease(self) -> AbstractAsyncContextManager[TradingCycle]: ...

    def execution_lease(self, pair: str) -> AbstractAsyncContextManager[TradingCycle]: ...


class Scheduler:
    def __init__(
        self,
        config: SchedulerConfig,
        runtime: RuntimeCycleSource,
        trigger_engine: Any | None = None,
    ) -> None:
        self.config = config
        self.pairs = tuple(Pair.parse(pair) for pair in config.pairs)
        self.interval_minutes = config.interval_minutes
        self.daily_summary_hour = config.daily_summary_hour
        self._cycle_count = 0
        # Status dict keyed by canonical pair string for stable lookups across
        # the trading-cycle / daily-summary / API surface.
        self._status: dict[str, dict[str, Any]] = {p.canonical(): {} for p in self.pairs}
        self._scheduler = AsyncIOScheduler()
        self._stop_event: asyncio.Event | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._trigger_engine = trigger_engine
        self.runtime = runtime
        self.config_revision = runtime.snapshot.revision
        # Watchdog state — tracks last successful cycle completion so the
        # heartbeat task can detect IntervalTrigger silent-miss bug (observed
        # 5/18 18:57 + 19:57 + 5/19 18:36; APScheduler's next_fire_time gets
        # stuck and wakeup() alone doesn't recover). When > 1.5x interval has
        # elapsed without a success, watchdog force-reschedules the job.
        self._last_successful_cycle_at: datetime | None = None
        self._inflight_batches: set[asyncio.Task[Any]] = set()

        # FR-103: structured boot log so ops can grep `pair_init` for
        # spot/swap/future split at startup.
        spot = [p.canonical() for p in self.pairs if p.market_type == "spot"]
        swap = [p.canonical() for p in self.pairs if p.market_type == "swap"]
        future = [p.canonical() for p in self.pairs if p.market_type == "future"]
        _slog.info("pair_init", spot=spot, swap=swap, future=future)

    async def start(self) -> None:
        self._stop_event = asyncio.Event()
        primary_failure: BaseException | None = None
        try:
            await self._run_until_stopped()
        except BaseException as error:
            primary_failure = error
        try:
            await wait_for_owned(asyncio.create_task(self._shutdown_owned_tasks()))
        except BaseException as cleanup_error:
            if not isinstance(cleanup_error, Exception):
                raise
            if primary_failure is None:
                raise
        if primary_failure is not None:
            raise primary_failure

    async def _run_until_stopped(self) -> None:
        self._require_active_cycle()

        # Register trading cycle job. Delay the first run by 15s so that:
        #   (a) async HTTP clients (OKX / data providers) finish their TLS
        #       handshakes and prime connection pools — without this, the very
        #       first batch of 5 parallel snapshot calls regularly races with
        #       API startup and a few pairs die with asyncio.CancelledError;
        #   (b) APScheduler / aiohttp internals have a moment to bind signal
        #       handlers and event loop before being pelted with traffic.
        # max_instances=1: prevents overlap when previous cycle is still running
        # misfire_grace_time=300 (5 min): tolerate brief event-loop blockage
        # (frontend polling burst, OKX slow round-trips, GC pause) without
        # silently dropping the trigger. 1s was too strict — 2026-05-12 11:21
        # UTC cycle missed fire silently because an unidentified blip
        # delayed the trigger by >1s; cycle came back only after API restart.
        # 5-min grace ≪ 60-min interval so no risk of doubling up.
        _startup_delay_s = 15
        # Align the trading cycle to the candle close instead of an arbitrary
        # interval anchored at process start. For a 240m (4h) interval we fire
        # 2 min after each 4h close (00:02/04:02/.../20:02 UTC) so the just-closed
        # bar is available from the exchange — Kronos must predict on a COMPLETE
        # latest bar, not a forming one. Falls back to interval if the cadence
        # isn't a clean hour multiple.
        _candle_aligned = self.interval_minutes % 60 == 0 and self.interval_minutes <= 24 * 60
        if _candle_aligned:
            _hours_step = self.interval_minutes // 60
            _cycle_trigger: Any = CronTrigger(hour=f"*/{_hours_step}", minute=2, timezone="UTC")
            # No immediate startup fire: let the first cycle land on the next
            # aligned candle close so Kronos always predicts on a complete bar.
            _cycle_job_kwargs: dict[str, Any] = {}
            logger.info(
                "Trading cycle trigger: cron */%dh at minute=2 UTC (candle-aligned, first fire at next close)",
                _hours_step,
            )
        else:
            _cycle_trigger = IntervalTrigger(minutes=self.interval_minutes)
            _cycle_job_kwargs = {"next_run_time": datetime.now(UTC) + timedelta(seconds=_startup_delay_s)}
            logger.info("Trading cycle trigger: interval %dm (not hour-aligned)", self.interval_minutes)
        self._scheduler.add_job(
            self._run_cycle,
            _cycle_trigger,
            id="trading_cycle",
            name="Trading cycle",
            max_instances=1,
            misfire_grace_time=300,
            **_cycle_job_kwargs,
        )

        # Register daily summary job — cron at configured hour UTC
        self._scheduler.add_job(
            self._emit_daily_summary,
            CronTrigger(hour=self.daily_summary_hour, minute=0, timezone="UTC"),
            id="daily_summary",
            name="Daily summary",
            max_instances=1,
            misfire_grace_time=300,
        )

        # Start price trigger engine if configured
        if self._trigger_engine is not None:
            await self._trigger_engine.start()
            self._scheduler.add_job(
                self._cleanup_expired_rules,
                CronTrigger(minute=0, timezone="UTC"),
                id="cleanup_expired_rules",
                name="Cleanup expired trigger rules",
                max_instances=1,
                misfire_grace_time=300,
            )

        # Signal handlers for graceful shutdown
        import signal

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self.stop)

        self._scheduler.start()
        self._write_scheduler_heartbeat()
        logger.info(
            "Scheduler started: pairs=%s interval=%dm daily_summary_hour=%d",
            [p.canonical() for p in self.pairs],
            self.interval_minutes,
            self.daily_summary_hour,
        )

        # Block until stop() is called
        # Defensive heartbeat against APScheduler timer-state staleness.
        # Observed twice (2026-05-12 11:21 / 2026-05-13 00:53 UTC):
        # AsyncIOScheduler stops firing after ~10h uptime, with
        # next_run_time still showing a past timestamp. Increasing
        # misfire_grace_time alone (1s → 300s) did not prevent recurrence.
        # The heartbeat calls scheduler.wakeup() periodically, forcing the
        # scheduler to re-evaluate triggers and re-schedule its internal
        # asyncio timer — even when the timer would otherwise stay stale.
        self._heartbeat_task = asyncio.create_task(self._scheduler_heartbeat())

        await self._stop_event.wait()

    async def _shutdown_owned_tasks(self) -> None:
        """Pause fires, drain batches, and release scheduler-owned resources."""

        failures: list[BaseException] = []
        try:
            self._scheduler.pause()
        except BaseException as error:
            failures.append(error)
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            await asyncio.gather(self._heartbeat_task, return_exceptions=True)
            self._heartbeat_task = None
        current = asyncio.current_task()
        pending = tuple(task for task in self._inflight_batches if task is not current)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        if self._trigger_engine is not None:
            try:
                await self._trigger_engine.stop()
            except BaseException as error:
                failures.append(error)
        try:
            self._scheduler.shutdown(wait=False)
        except BaseException as error:
            failures.append(error)
        control_flow = next((error for error in failures if not isinstance(error, Exception)), None)
        if control_flow is not None:
            raise control_flow
        if failures:
            raise failures[0]
        logger.info("Scheduler stopped gracefully")

    async def _scheduler_heartbeat(self) -> None:
        """Periodic wakeup nudge + staleness watchdog.

        wakeup() alone proved insufficient against the IntervalTrigger silent-miss
        bug (5/18 18:57, 5/18 19:57, 5/19 18:36 — three confirmed misses). The
        AsyncIOScheduler's next_fire_time gets stuck in the past and wakeup()
        doesn't re-anchor it. So we also track last_successful_cycle_at and
        force-reschedule the trading_cycle job when staleness exceeds 1.5x interval.

        5-min cadence keeps the apscheduler internal timer fresh without
        meaningful overhead. Errors here are non-fatal; we log and keep looping.
        """
        # Staleness threshold = 1.5x interval; long enough to allow a slow
        # cycle (e.g., 5/18 09:19 took 40min during OKX cooldown) without
        # false-positive force-reschedule, short enough to catch the silent
        # miss before a second interval is lost.
        stale_threshold_s = int(self.interval_minutes * 60 * 1.5)
        while self._stop_event is not None and not self._stop_event.is_set():
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=300)
                return  # stop_event set; heartbeat exits cleanly
            except TimeoutError:
                pass  # 5 min elapsed; nudge the scheduler
            try:
                self._scheduler.wakeup()
                logger.debug("scheduler heartbeat: wakeup() nudged")
            except Exception as exc:
                logger.warning("scheduler heartbeat wakeup failed: %s", exc)
            # Staleness check — only if we have a baseline (skip until first
            # successful cycle to avoid bogus reschedule during startup delay).
            if self._last_successful_cycle_at is not None:
                elapsed_s = (datetime.now(UTC) - self._last_successful_cycle_at).total_seconds()
                if elapsed_s > stale_threshold_s:
                    logger.warning(
                        "scheduler watchdog: last successful cycle was %.0fs ago "
                        "(threshold %ds = 1.5x%dm) — IntervalTrigger likely silent-missed; "
                        "force-rescheduling trading_cycle job",
                        elapsed_s,
                        stale_threshold_s,
                        self.interval_minutes,
                    )
                    try:
                        # Force the job to fire ASAP by rewriting next_run_time.
                        # APScheduler will then re-anchor the IntervalTrigger
                        # against this new baseline.
                        self._scheduler.modify_job(
                            "trading_cycle",
                            next_run_time=datetime.now(UTC) + timedelta(seconds=10),
                        )
                        logger.warning(
                            "scheduler watchdog: trading_cycle rescheduled to fire in 10s",
                        )
                    except Exception as exc:
                        logger.error(
                            "scheduler watchdog: force-reschedule failed: %s",
                            exc,
                            exc_info=True,
                        )

    def stop(self) -> None:
        if self._stop_event:
            self._stop_event.set()

    @property
    def status(self) -> dict[str, dict[str, Any]]:
        return self._status

    @property
    def jobs(self) -> list[dict[str, Any]]:
        """Return list of registered jobs with their next run times."""
        return [
            {
                "id": job.id,
                "name": job.name,
                "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
                "trigger": str(job.trigger),
            }
            for job in self._scheduler.get_jobs()
        ]

    async def _run_cycle(self) -> None:
        """Single trading cycle — run all pairs concurrently.

        Wrapped in an outer asyncio.wait_for so any hanging await inside
        a pair's graph (ccxt connection, LLM call stuck after its own
        timeout window, downstream callback that never resolves) cannot
        block APScheduler past one interval. Observed 5/16-5/17: a single
        hung cycle leaked an asyncio Task that AsyncIOScheduler still
        treated as "running" under max_instances=1, silently dropping
        every subsequent fire until the process was restarted. The hard
        outer cap forces the asyncio.CancelledError back to APScheduler
        so the slot frees and the next interval fires normally.
        """
        owner = asyncio.current_task()
        if owner is not None:
            self._inflight_batches.add(owner)
        cycle_timeout_s = max(self.interval_minutes * 60 - 60, 60)
        try:
            await asyncio.wait_for(self.run_once(), timeout=cycle_timeout_s)
        except TimeoutError:
            logger.error(
                "Trading cycle exceeded outer timeout of %ds — cancelled to free "
                "the APScheduler slot for the next interval",
                cycle_timeout_s,
            )
        except Exception:
            logger.warning("Scheduled trading batch failed")
        finally:
            if owner is not None:
                self._inflight_batches.discard(owner)

    async def run_once(self) -> None:
        """Reload once, then run every configured pair against that exact graph."""
        cycle = self._require_active_cycle()
        self.config_revision = cycle.snapshot.revision
        await asyncio.gather(*(self._run_pair(pair.canonical()) for pair in self.pairs))
        self._cycle_count += 1
        for pair in self.pairs:
            next_run = datetime.now(UTC) + timedelta(minutes=self.interval_minutes)
            self._status[pair.canonical()]["next_run"] = next_run.isoformat()
        self._write_scheduler_heartbeat()
        self._last_successful_cycle_at = datetime.now(UTC)

    @staticmethod
    def _write_scheduler_heartbeat() -> None:
        """Record scheduler liveness without forcing an off-cycle trade."""
        try:
            from pathlib import Path

            hb_dir = Path.home() / ".cryptotrader"
            hb_dir.mkdir(parents=True, exist_ok=True)
            (hb_dir / "scheduler.heartbeat").write_text(datetime.now(UTC).isoformat())
        except Exception:
            logger.info("Failed to write scheduler heartbeat", exc_info=True)

    def _require_active_cycle(self) -> TradingCycle:
        if self.runtime.cycle is None:
            raise RuntimeError("runtime configuration is not active")
        return self.runtime.cycle

    async def _run_pair(self, pair: str) -> None:
        from cryptotrader.tracing import set_trace_id

        # Per-pair mutex prevents concurrent cycles on the same pair (e.g. a
        # manual ``trader run`` overlapping with a scheduler tick). The lock
        # holder writes its uuid; release is owner-checked so a TTL-expired
        # holder cannot wipe a fresh holder's key.
        trace_id = set_trace_id()
        self._status[pair]["last_run"] = datetime.now(UTC).isoformat()
        self._status[pair]["trace_id"] = trace_id
        try:
            async with self.runtime.execution_lease(pair) as cycle:
                await self._run_pair_locked(pair, cycle, trace_id)
        except Exception:
            # Config / Redis init failures must not propagate to gather() — the
            # cycle should continue with the remaining pairs. _run_pair_locked
            # has its own catch for in-cycle errors; this wrapper covers
            # everything before the lock is acquired.
            logger.warning("Scheduler setup failed for pair %s trace=%s", pair, trace_id)
            self._status[pair]["last_error"] = "cycle_failed"

    async def _run_pair_locked(self, pair: str, cycle: TradingCycle, trace_id: str) -> None:
        # Spec 013 FR-203 / T021: bind canonical pair so every log line in this
        # cycle is greppable by ccxt symbol regardless of which node logs.
        _slog.bind(pair=pair, trace_id=trace_id).info("cycle_pair_start")
        try:
            from cryptotrader.decision.models import CycleRequest

            cycle_timeout = 300
            try:
                outcome = await asyncio.wait_for(
                    cycle.run(CycleRequest(Pair.parse(pair))),
                    timeout=cycle_timeout,
                )
            except TimeoutError:
                logger.error("Scheduler timed out after %ds for pair %s", cycle_timeout, pair)
                self._status[pair]["last_error"] = "cycle_timeout"
                return
            self._status[pair]["last_error"] = None
            action = outcome.target_position.side if outcome.target_position is not None else "flat"
            risk_passed = outcome.status != "risk_rejected"
            self._status[pair]["last_action"] = action
            self._status[pair]["risk_passed"] = risk_passed
            self._status[pair]["last_status"] = outcome.status
            self._status[pair]["cycle_id"] = outcome.cycle_id
            logger.info(
                "Cycle complete [%s] trace=%s: status=%s target=%s risk=%s",
                pair,
                trace_id,
                outcome.status,
                action,
                risk_passed,
            )

        except Exception:
            logger.warning("Scheduler cycle failed for pair %s trace=%s", pair, trace_id)
            self._status[pair]["last_error"] = "cycle_failed"

    async def _emit_daily_summary(self) -> None:
        """Send a safe scheduler/book summary from the current Runtime snapshot."""
        try:
            from cryptotrader.notifications import Notifier

            config = self.runtime.snapshot.document
            notifier = Notifier(
                webhook_url=config.notifications.webhook_url,
                events=config.notifications.events,
                webhook_timeout=config.notifications.webhook_timeout,
                telegram_config=config.notifications.telegram,
            )
            summary = {
                "date": datetime.now(UTC).strftime("%Y-%m-%d"),
                "config_revision": self.runtime.snapshot.revision,
                "enabled_books": [book.id for book in config.execution.books if book.enabled],
                "pairs": {
                    p: {
                        "last_action": s.get("last_action", "none"),
                        "risk_passed": s.get("risk_passed"),
                        "last_error": s.get("last_error"),
                    }
                    for p, s in self._status.items()
                },
            }
            await notifier.notify("daily_summary", summary)
        except Exception:
            logger.warning("Failed to emit daily summary")

    async def _cleanup_expired_rules(self) -> None:
        """Hourly cleanup of expired agent-created trigger rules."""
        if self._trigger_engine is None:
            return
        try:
            store = self._trigger_engine._store
            count = await store.cleanup_expired_rules()
            if count > 0:
                await self._trigger_engine.reload_rules()
        except Exception:
            logger.warning("Failed to cleanup expired rules")

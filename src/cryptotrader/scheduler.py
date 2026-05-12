"""APScheduler-based trading scheduler with interval and cron triggers."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from cryptotrader._compat import UTC
from cryptotrader.pair import Pair

logger = logging.getLogger(__name__)
_slog = structlog.get_logger(__name__)


class Scheduler:
    def __init__(
        self,
        pairs: list,
        interval_minutes: int = 240,
        daily_summary_hour: int = 0,
        trigger_engine: Any | None = None,
    ):
        # Per spec 013-pair-value-object: scheduler holds list[Pair]; legacy
        # callers passing list[str] are auto-promoted to spot Pair instances
        # (matches D4 backwards-compat for ``[scheduler].pairs`` legacy form).
        normalized: list[Pair] = []
        for p in pairs:
            if isinstance(p, Pair):
                normalized.append(p)
            elif isinstance(p, str):
                normalized.append(Pair.parse(p))
            else:
                raise TypeError(f"Scheduler.pairs item must be Pair or str; got {type(p).__name__}")
        self.pairs: list[Pair] = normalized
        self.interval_minutes = interval_minutes
        self.daily_summary_hour = daily_summary_hour
        self._cycle_count = 0
        # Status dict keyed by canonical pair string for stable lookups across
        # the trading-cycle / daily-summary / API surface.
        self._status: dict[str, dict[str, Any]] = {p.canonical(): {} for p in self.pairs}
        self._scheduler = AsyncIOScheduler()
        self._stop_event: asyncio.Event | None = None
        self._trigger_engine = trigger_engine

        # FR-103: structured boot log so ops can grep `pair_init` for
        # spot/swap/future split at startup.
        spot = [p.canonical() for p in self.pairs if p.market_type == "spot"]
        swap = [p.canonical() for p in self.pairs if p.market_type == "swap"]
        future = [p.canonical() for p in self.pairs if p.market_type == "future"]
        _slog.info("pair_init", spot=spot, swap=swap, future=future)

    async def start(self) -> None:
        # Startup reconciliation for live mode
        await self._startup_reconcile()

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
        self._scheduler.add_job(
            self._run_cycle,
            IntervalTrigger(minutes=self.interval_minutes),
            id="trading_cycle",
            name="Trading cycle",
            next_run_time=datetime.now(UTC) + timedelta(seconds=_startup_delay_s),
            max_instances=1,
            misfire_grace_time=300,
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
            from cryptotrader.config import load_config as _lc

            _cfg = _lc()
            self._scheduler.add_job(
                self._trigger_engine.poll_funding_rates,
                IntervalTrigger(minutes=_cfg.triggers.funding_rate_poll_interval_minutes),
                id="funding_rate_poll",
                name="Funding rate poll",
                max_instances=1,
                misfire_grace_time=300,
            )
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
        logger.info(
            "Scheduler started: pairs=%s interval=%dm daily_summary_hour=%d",
            [p.canonical() for p in self.pairs],
            self.interval_minutes,
            self.daily_summary_hour,
        )

        # Block until stop() is called
        self._stop_event = asyncio.Event()
        await self._stop_event.wait()

        # Cleanup
        if self._trigger_engine is not None:
            await self._trigger_engine.stop()
        self._scheduler.shutdown(wait=False)
        await self._close_live_exchanges()
        logger.info("Scheduler stopped gracefully")

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
        """Single trading cycle — run all pairs concurrently."""
        try:
            tasks = [self._run_pair(p.canonical()) for p in self.pairs]
            await asyncio.gather(*tasks, return_exceptions=True)
            self._cycle_count += 1
            for p in self.pairs:
                next_run = datetime.now(UTC) + timedelta(minutes=self.interval_minutes)
                self._status[p.canonical()]["next_run"] = next_run.isoformat()
            # Persist a portfolio snapshot every cycle so risk gate's
            # get_daily_pnl has a real time series. nodes/execution.snapshot only
            # fires on actual trades; on quiet days that meant zero rows for
            # hours and the daily-loss check fell back to "unknown".
            await self._write_cycle_snapshot()
            # Heartbeat: write timestamp to disk after every cycle so docker
            # healthcheck ("arena scheduler healthcheck") can detect a hung
            # scheduler (file mtime older than 2x interval = unhealthy).
            try:
                from pathlib import Path

                hb_dir = Path.home() / ".cryptotrader"
                hb_dir.mkdir(parents=True, exist_ok=True)
                (hb_dir / "scheduler.heartbeat").write_text(
                    datetime.now(UTC).isoformat(),
                )
            except Exception:
                logger.info("Failed to write scheduler heartbeat", exc_info=True)
        except Exception:
            logger.warning("Unexpected error in trading cycle", exc_info=True)

    async def _write_cycle_snapshot(self) -> None:
        """Write a fresh portfolio_snapshots row for the cycle.

        Live mode: pull real exchange equity via read_portfolio_from_exchange.
        Paper / fallback: read whatever PortfolioManager already knows from DB.
        Either way, swallow errors — a missed snapshot is far less harmful than
        crashing the cycle.
        """
        from cryptotrader.config import load_config
        from cryptotrader.portfolio.manager import PortfolioManager, read_portfolio_from_exchange

        config = load_config()
        db_url = config.infrastructure.database_url
        pm = PortfolioManager(db_url)

        total = 0.0
        cash = 0.0
        try:
            if config.engine == "live" and self.pairs:
                # Build a minimal state for the exchange call; price=0 is fine —
                # exchange.get_balance/get_positions don't need it for the cash leg.
                state = {
                    "metadata": {
                        "engine": "live",
                        "exchange_id": config.scheduler.exchange_id,
                        "pair": self.pairs[0].canonical(),
                    },
                    "data": {"snapshot_summary": {"price": 0}},
                }
                ex_portfolio = await read_portfolio_from_exchange(state)
                if ex_portfolio:
                    total = float(ex_portfolio.get("total_value", 0.0) or 0.0)
                    cash = float(ex_portfolio.get("cash", 0.0) or 0.0)
            if total <= 0:
                # Fallback to DB-known portfolio (paper mode, or live read failed)
                pm_portfolio = await pm.get_portfolio()
                total = float(pm_portfolio.get("total_value", 0.0) or 0.0)
                cash = float(pm_portfolio.get("cash", 0.0) or 0.0)
        except Exception:
            logger.info("cycle snapshot: portfolio read failed", exc_info=True)
            return

        if total <= 0:
            logger.debug("cycle snapshot: total_value=0, skipping write")
            return

        try:
            await pm.snapshot("default", total, cash)
            logger.info("cycle snapshot written: total=%.2f cash=%.2f", total, cash)
        except Exception:
            logger.info("cycle snapshot: write failed", exc_info=True)

    async def _close_live_exchanges(self) -> None:
        from cryptotrader.nodes.execution import _live_exchanges

        for ex_id, exchange in list(_live_exchanges.items()):
            try:
                await exchange.close()
            except Exception:
                logger.info("Failed to close exchange %s", ex_id, exc_info=True)
        _live_exchanges.clear()

    async def _startup_reconcile(self) -> None:
        """Run startup reconciliation to detect orphaned orders (live mode only)."""
        from cryptotrader.config import load_config

        config = load_config()
        if config.engine != "live":
            return

        try:
            from cryptotrader.execution.reconcile import Reconciler
            from cryptotrader.nodes.execution import _get_exchange

            dummy_state = {"metadata": {"engine": "live", "exchange_id": config.scheduler.exchange_id}, "data": {}}
            exchange, _ = await _get_exchange(dummy_state, self.pairs[0].canonical())
            reconciler = Reconciler(exchange)
            orphans = await reconciler.detect_orphans(set())
            if orphans:
                logger.warning("Startup reconciliation found %d orphaned orders", len(orphans))
                notifier = self._get_notifier(config)
                await notifier.notify(
                    "reconcile_mismatch",
                    {"orphan_count": len(orphans), "orphan_ids": [o.get("id") for o in orphans]},
                )
            else:
                logger.info("Startup reconciliation: no orphaned orders")
        except Exception:
            logger.warning("Startup reconciliation failed", exc_info=True)

    @staticmethod
    def _get_notifier(config):
        from cryptotrader.notifications import Notifier

        return Notifier(
            webhook_url=config.notifications.webhook_url,
            enabled=config.notifications.enabled,
            events=config.notifications.events,
            webhook_timeout=config.notifications.webhook_timeout,
            telegram_config=config.notifications.telegram,
        )

    async def _run_pair(self, pair: str, trigger_meta: dict[str, Any] | None = None) -> None:
        from cryptotrader.config import load_config
        from cryptotrader.cycle_lock import cycle_lock
        from cryptotrader.risk.state import RedisStateManager

        # Per-pair mutex prevents concurrent cycles on the same pair (e.g. a
        # manual ``arena run`` overlapping with a scheduler tick). The lock
        # holder writes its uuid; release is owner-checked so a TTL-expired
        # holder cannot wipe a fresh holder's key.
        try:
            redis_state = RedisStateManager(load_config().infrastructure.redis_url)
            async with cycle_lock(redis_state, pair) as acquired:
                if not acquired:
                    logger.warning("cycle_lock held for %s — skipping this scheduler tick", pair)
                    return
                await self._run_pair_locked(pair, trigger_meta)
        except Exception as e:
            # Config / Redis init failures must not propagate to gather() — the
            # cycle should continue with the remaining pairs. _run_pair_locked
            # has its own catch for in-cycle errors; this wrapper covers
            # everything before the lock is acquired.
            logger.warning("Scheduler setup failed for pair %s", pair, exc_info=True)
            self._status[pair]["last_error"] = str(e)

    async def _run_pair_locked(self, pair: str, trigger_meta: dict[str, Any] | None = None) -> None:
        from cryptotrader.tracing import set_trace_id

        trace_id = set_trace_id()
        # Spec 013 FR-203 / T021: bind canonical pair so every log line in this
        # cycle is greppable by ccxt symbol regardless of which node logs.
        _slog.bind(pair=pair, trace_id=trace_id).info("cycle_pair_start")
        self._status[pair]["last_run"] = datetime.now(UTC).isoformat()
        self._status[pair]["trace_id"] = trace_id
        try:
            from cryptotrader.config import load_config
            from cryptotrader.graph import build_trading_graph

            config = load_config()
            graph = build_trading_graph()
            from cryptotrader.state import build_initial_state

            extra_meta: dict[str, Any] = {"cycle_count": self._cycle_count}
            if trigger_meta:
                extra_meta["schedule_depth"] = trigger_meta.get("schedule_depth", 0)
                extra_meta["trigger_event_id"] = trigger_meta.get("trigger_event_id")
            initial = build_initial_state(
                pair,
                engine=config.engine,
                exchange_id=config.scheduler.exchange_id,
                config=config,
                extra_metadata=extra_meta,
            )
            from cryptotrader.tracing import add_timing_to_trace, run_graph_traced

            graph_timeout = config.execution.graph_timeout_s
            import time as _time

            from cryptotrader.metrics import get_metrics_collector

            pipeline_t0 = _time.monotonic()
            try:
                result, node_trace = await asyncio.wait_for(run_graph_traced(graph, initial), timeout=graph_timeout)
                # Histogram observation populates pipeline_p50_ms / p95_ms in
                # /api/metrics/summary. Also fired in chat analysis_runner.
                get_metrics_collector().observe_pipeline_duration(ms=(_time.monotonic() - pipeline_t0) * 1000.0)
                add_timing_to_trace(node_trace)
                for t in node_trace:
                    logger.info("Node %s [%dms]: %s", t["node"], t["duration_ms"], t["summary"][:120])
            except TimeoutError:
                logger.error("Scheduler timed out after %ds for pair %s", graph_timeout, pair)
                self._status[pair]["last_error"] = f"timeout after {graph_timeout}s"
                return
            self._status[pair]["last_error"] = None

            data = result.get("data", {})
            verdict = data.get("verdict", {})
            risk_gate = data.get("risk_gate", {})
            action = verdict.get("action", "unknown")
            risk_passed = risk_gate.get("passed", False)
            self._status[pair]["last_action"] = action
            self._status[pair]["risk_passed"] = risk_passed
            logger.info(
                "Cycle complete [%s] trace=%s: action=%s confidence=%.2f risk=%s",
                pair,
                trace_id,
                action,
                verdict.get("confidence", 0),
                "PASS" if risk_passed else f"REJECT({risk_gate.get('rejected_by', '?')})",
            )

        except Exception as e:
            logger.warning("Scheduler error for pair %s", pair, exc_info=True)
            self._status[pair]["last_error"] = str(e)

    async def _emit_daily_summary(self) -> None:
        """Send daily summary notification with portfolio and trading stats."""
        try:
            from cryptotrader.config import load_config
            from cryptotrader.notifications import Notifier
            from cryptotrader.portfolio.manager import PortfolioManager

            config = load_config()
            notifier = Notifier(
                webhook_url=config.notifications.webhook_url,
                events=config.notifications.events,
                webhook_timeout=config.notifications.webhook_timeout,
                telegram_config=config.notifications.telegram,
            )
            pm = PortfolioManager(config.infrastructure.database_url)
            portfolio = await pm.get_portfolio()
            # get_daily_pnl returns None when no snapshot exists in today's UTC window
            daily_pnl_raw = await pm.get_daily_pnl()
            drawdown = await pm.get_drawdown()

            summary = {
                "date": datetime.now(UTC).strftime("%Y-%m-%d"),
                "portfolio_value": portfolio.get("total_value", 0),
                "daily_pnl": daily_pnl_raw,  # may be null in the notification payload
                "drawdown": drawdown,
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
            logger.warning("Failed to emit daily summary", exc_info=True)

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
            logger.warning("Failed to cleanup expired rules", exc_info=True)

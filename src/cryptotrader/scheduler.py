"""APScheduler-based trading scheduler with interval and cron triggers."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

logger = logging.getLogger(__name__)


class Scheduler:
    def __init__(self, pairs: list[str], interval_minutes: int = 240, daily_summary_hour: int = 0):
        self.pairs = pairs
        self.interval_minutes = interval_minutes
        self.daily_summary_hour = daily_summary_hour
        self._cycle_count = 0
        self._status: dict[str, dict[str, Any]] = {p: {} for p in pairs}
        self._scheduler = AsyncIOScheduler()
        self._stop_event: asyncio.Event | None = None

    async def start(self) -> None:
        # Startup reconciliation for live mode
        await self._startup_reconcile()

        # Register trading cycle job -- runs immediately, then every interval_minutes
        # max_instances=1: prevents overlap when previous cycle is still running
        # misfire_grace_time=1: discard missed triggers after 1s instead of catching up
        self._scheduler.add_job(
            self._run_cycle,
            IntervalTrigger(minutes=self.interval_minutes),
            id="trading_cycle",
            name="Trading cycle",
            next_run_time=datetime.now(UTC),
            max_instances=1,
            misfire_grace_time=1,
        )

        # Register daily summary job — cron at configured hour UTC
        self._scheduler.add_job(
            self._emit_daily_summary,
            CronTrigger(hour=self.daily_summary_hour, minute=0, timezone="UTC"),
            id="daily_summary",
            name="Daily summary",
            max_instances=1,
            misfire_grace_time=1,
        )

        # Signal handlers for graceful shutdown
        import signal

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self.stop)

        self._scheduler.start()
        logger.info(
            "Scheduler started: pairs=%s interval=%dm daily_summary_hour=%d",
            self.pairs,
            self.interval_minutes,
            self.daily_summary_hour,
        )

        # Block until stop() is called
        self._stop_event = asyncio.Event()
        await self._stop_event.wait()

        # Cleanup
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
            tasks = [self._run_pair(p) for p in self.pairs]
            await asyncio.gather(*tasks, return_exceptions=True)
            self._cycle_count += 1
            for p in self.pairs:
                next_run = datetime.now(UTC) + timedelta(minutes=self.interval_minutes)
                self._status[p]["next_run"] = next_run.isoformat()
        except Exception:
            logger.warning("Unexpected error in trading cycle", exc_info=True)

    async def _close_live_exchanges(self) -> None:
        from cryptotrader.nodes.execution import _live_exchanges

        for ex_id, exchange in list(_live_exchanges.items()):
            try:
                await exchange.close()
            except Exception:
                logger.debug("Failed to close exchange %s", ex_id, exc_info=True)
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
            exchange, _ = await _get_exchange(dummy_state, self.pairs[0])
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
        )

    async def _run_pair(self, pair: str) -> None:
        from cryptotrader.tracing import set_trace_id

        trace_id = set_trace_id()
        self._status[pair]["last_run"] = datetime.now(UTC).isoformat()
        self._status[pair]["trace_id"] = trace_id
        try:
            from cryptotrader.config import load_config
            from cryptotrader.graph import build_trading_graph

            config = load_config()
            graph = build_trading_graph()
            from cryptotrader.state import build_initial_state

            initial = build_initial_state(
                pair,
                engine=config.engine,
                exchange_id=config.scheduler.exchange_id,
                config=config,
                extra_metadata={"cycle_count": self._cycle_count},
            )
            from cryptotrader.tracing import add_timing_to_trace, run_graph_traced

            graph_timeout = config.execution.graph_timeout_s
            try:
                result, node_trace = await asyncio.wait_for(run_graph_traced(graph, initial), timeout=graph_timeout)
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
            )
            pm = PortfolioManager(config.infrastructure.database_url)
            portfolio = await pm.get_portfolio()
            daily_pnl = await pm.get_daily_pnl()
            drawdown = await pm.get_drawdown()

            summary = {
                "date": datetime.now(UTC).strftime("%Y-%m-%d"),
                "portfolio_value": portfolio.get("total_value", 0),
                "daily_pnl": daily_pnl,
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

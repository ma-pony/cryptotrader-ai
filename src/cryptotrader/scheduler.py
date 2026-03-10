"""Simple asyncio-based scheduler for periodic trading cycles."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)


class Scheduler:
    def __init__(self, pairs: list[str], interval_minutes: int = 240):
        self.pairs = pairs
        self.interval = interval_minutes * 60
        self._running = False
        self._sleep_task: asyncio.Task | None = None
        self._status: dict[str, dict[str, Any]] = {p: {} for p in pairs}

    async def start(self) -> None:
        self._running = True
        self._cycle_count = 0

        # Register signal handlers for graceful shutdown
        import signal

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self.stop)

        # Startup reconciliation for live mode
        await self._startup_reconcile()

        logger.info("Scheduler started: pairs=%s interval=%dm", self.pairs, self.interval // 60)
        while self._running:
            tasks = [self._run_pair(p) for p in self.pairs]
            await asyncio.gather(*tasks, return_exceptions=True)
            self._cycle_count += 1
            for p in self.pairs:
                self._status[p]["next_run"] = (datetime.now(UTC) + timedelta(seconds=self.interval)).isoformat()

            # Emit daily summary once per day (every 24h / interval cycles)
            cycles_per_day = max(1, 86400 // self.interval)
            if self._cycle_count % cycles_per_day == 0:
                await self._emit_daily_summary()

            try:
                self._sleep_task = asyncio.ensure_future(asyncio.sleep(self.interval))
                await self._sleep_task
            except asyncio.CancelledError:
                break

        logger.info("Scheduler stopped gracefully")

    def stop(self) -> None:
        self._running = False
        if self._sleep_task and not self._sleep_task.done():
            self._sleep_task.cancel()
        # Schedule cleanup — may not run if loop is shutting down
        try:
            loop = asyncio.get_running_loop()
            self._cleanup_task = loop.create_task(self._close_live_exchanges())
        except RuntimeError:
            pass  # No running loop — exchanges will be GC'd

    async def _close_live_exchanges(self) -> None:
        from cryptotrader.nodes.execution import _live_exchanges

        for ex_id, exchange in list(_live_exchanges.items()):
            try:
                await exchange.close()
            except Exception:
                logger.debug("Failed to close exchange %s", ex_id, exc_info=True)
        _live_exchanges.clear()

    @property
    def status(self) -> dict[str, dict[str, Any]]:
        return self._status

    async def _startup_reconcile(self) -> None:
        """Run startup reconciliation to detect orphaned orders (live mode only)."""
        from cryptotrader.config import load_config

        config = load_config()
        if config.engine != "live":
            return

        try:
            from cryptotrader.execution.reconcile import Reconciler
            from cryptotrader.nodes.execution import _get_exchange

            # Build a minimal state to get the exchange
            dummy_state = {"metadata": {"engine": "live", "exchange_id": config.scheduler.exchange_id}, "data": {}}
            exchange, _ = _get_exchange(dummy_state, self.pairs[0])
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
            try:
                result = await asyncio.wait_for(graph.ainvoke(initial), timeout=300)
            except TimeoutError:
                logger.error("Scheduler timed out after 300s for pair %s", pair)
                self._status[pair]["last_error"] = "timeout after 300s"
                return
            self._status[pair]["last_error"] = None

            # Log outcome to status (notifications are already sent by the graph nodes)
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
            logger.error("Scheduler error for %s: %s", pair, e)
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

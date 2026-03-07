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

    @property
    def status(self) -> dict[str, dict[str, Any]]:
        return self._status

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
            initial = {
                "messages": [],
                "data": {},
                "metadata": {
                    "pair": pair,
                    "engine": config.engine,
                    "exchange_id": config.scheduler.exchange_id,
                    "timeframe": config.data.default_timeframe,
                    "ohlcv_limit": config.data.ohlcv_limit,
                    "analysis_model": config.models.analysis,
                    "debate_model": config.models.debate,
                    "verdict_model": config.models.verdict,
                    "models": {
                        "tech_agent": config.models.tech_agent,
                        "chain_agent": config.models.chain_agent,
                        "news_agent": config.models.news_agent,
                        "macro_agent": config.models.macro_agent,
                    },
                    "database_url": config.infrastructure.database_url,
                    "redis_url": config.infrastructure.redis_url,
                    "convergence_threshold": config.debate.convergence_threshold,
                    "max_single_pct": config.risk.position.max_single_pct,
                },
                "debate_round": 0,
                "max_debate_rounds": config.debate.max_rounds,
                "divergence_scores": [],
            }
            result = await graph.ainvoke(initial)
            self._status[pair]["last_error"] = None

            # Log outcome to status (notifications are already sent by the graph nodes)
            data = result.get("data", {})
            verdict = data.get("verdict", {})
            risk_gate = data.get("risk_gate", {})
            self._status[pair]["last_action"] = verdict.get("action", "unknown")
            self._status[pair]["risk_passed"] = risk_gate.get("passed", False)

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

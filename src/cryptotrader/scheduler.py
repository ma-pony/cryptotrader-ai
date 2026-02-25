"""Simple asyncio-based scheduler for periodic trading cycles."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import UTC, datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)


class Scheduler:
    def __init__(self, pairs: list[str], interval_minutes: int = 240):
        self.pairs = pairs
        self.interval = interval_minutes * 60
        self._running = False
        self._status: dict[str, dict[str, Any]] = {p: {} for p in pairs}

    async def start(self) -> None:
        self._running = True
        logger.info("Scheduler started: pairs=%s interval=%dm", self.pairs, self.interval // 60)
        while self._running:
            tasks = [self._run_pair(p) for p in self.pairs]
            await asyncio.gather(*tasks, return_exceptions=True)
            for p in self.pairs:
                self._status[p]["next_run"] = (
                    datetime.now(UTC) + timedelta(seconds=self.interval)
                ).isoformat()
            await asyncio.sleep(self.interval)

    def stop(self) -> None:
        self._running = False

    @property
    def status(self) -> dict[str, dict[str, Any]]:
        return self._status

    async def _run_pair(self, pair: str) -> None:
        self._status[pair]["last_run"] = datetime.now(UTC).isoformat()
        try:
            from cryptotrader.graph import build_trading_graph
            from cryptotrader.config import load_config

            config = load_config()
            graph = build_trading_graph()
            initial = {
                "messages": [], "data": {}, "metadata": {
                    "pair": pair,
                    "engine": config.engine,
                    "exchange_id": config.scheduler.exchange_id,
                    "timeframe": config.data.default_timeframe,
                    "ohlcv_limit": config.data.ohlcv_limit,
                    "analysis_model": config.models.analysis,
                    "debate_model": config.models.debate,
                    "verdict_model": config.models.verdict,
                    "models": config.models.agents,
                    "database_url": os.environ.get("DATABASE_URL"),
                    "redis_url": os.environ.get("REDIS_URL"),
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

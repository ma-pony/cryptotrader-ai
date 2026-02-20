"""Simple asyncio-based scheduler for periodic trading cycles."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
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

            graph = build_trading_graph()
            initial = {
                "messages": [], "data": {}, "metadata": {
                    "pair": pair, "engine": "paper", "exchange_id": "binance",
                    "timeframe": "1h", "ohlcv_limit": 100,
                },
                "debate_round": 0, "max_debate_rounds": 3, "divergence_scores": [],
            }
            await graph.ainvoke(initial)
            self._status[pair]["last_error"] = None
        except Exception as e:
            logger.error("Scheduler error for %s: %s", pair, e)
            self._status[pair]["last_error"] = str(e)
        self._status[pair]["next_run"] = datetime.now(UTC).isoformat()

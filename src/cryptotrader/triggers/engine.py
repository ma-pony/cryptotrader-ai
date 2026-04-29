"""Price trigger engine — WebSocket listener + condition matcher."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

if TYPE_CHECKING:
    from cryptotrader.config import TriggersConfig
    from cryptotrader.risk.state import RedisStateManager
    from cryptotrader.triggers.store import TriggerRuleStore

logger = logging.getLogger(__name__)

BINANCE_WS_BASE = "wss://stream.binance.com/stream"


class PriceTriggerEngine:
    """Event-driven price trigger engine using Binance WebSocket streams."""

    def __init__(
        self,
        store: TriggerRuleStore,
        redis_state: RedisStateManager,
        run_pair_callback: Callable[[str, dict[str, Any]], Coroutine[Any, Any, None]],
        config: TriggersConfig,
    ) -> None:
        self._store = store
        self._redis = redis_state
        self._run_pair = run_pair_callback
        self._config = config
        self._rules: list[Any] = []
        self._running = False
        self._ws_task: asyncio.Task[None] | None = None
        self._funding_rates: dict[str, float] = {}
        self._last_prices: dict[str, float] = {}
        self._price_history: dict[str, list[dict[str, float]]] = {}
        self._reconnect_delay = 1.0

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        await self.reload_rules()
        self._ws_task = asyncio.create_task(self._ws_connect())
        logger.info("PriceTriggerEngine started with %d rules", len(self._rules))

    async def stop(self) -> None:
        self._running = False
        if self._ws_task and not self._ws_task.done():
            self._ws_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._ws_task
        self._ws_task = None
        logger.info("PriceTriggerEngine stopped")

    async def reload_rules(self) -> None:
        self._rules = await self._store.list_rules(enabled_only=True)
        pairs = {r.pair.replace("/", "").lower() for r in self._rules}
        logger.info("Loaded %d enabled rules covering %d pairs", len(self._rules), len(pairs))

    def _subscribed_streams(self) -> list[str]:
        pairs = {r.pair.replace("/", "").lower() for r in self._rules}
        return [f"{p}@ticker" for p in sorted(pairs)]

    async def _ws_connect(self) -> None:
        import websockets

        while self._running:
            streams = self._subscribed_streams()
            if not streams:
                await asyncio.sleep(5)
                await self.reload_rules()
                continue

            stream_param = "/".join(streams)
            url = f"{BINANCE_WS_BASE}?streams={stream_param}"
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                    self._reconnect_delay = 1.0
                    logger.info("WebSocket connected to %d streams", len(streams))
                    await self._ws_listen_loop(ws)
            except asyncio.CancelledError:
                return
            except Exception:
                if not self._running:
                    return
                logger.warning("WebSocket disconnected, reconnecting in %.1fs", self._reconnect_delay, exc_info=True)
                await self._reconnect_with_backoff()
                await self.reload_rules()

    async def _ws_listen_loop(self, ws: Any) -> None:
        async for raw_msg in ws:
            if not self._running:
                return
            try:
                msg = json.loads(raw_msg)
                data = msg.get("data", {})
                if data.get("e") == "24hrTicker":
                    await self._handle_ticker(data)
            except Exception:
                logger.debug("Failed to parse WS message", exc_info=True)

    async def _handle_ticker(self, data: dict[str, Any]) -> None:
        symbol = data.get("s", "").upper()
        price = float(data.get("c", 0))
        if not symbol or price <= 0:
            return

        pair = self._symbol_to_pair(symbol)
        prev_price = self._last_prices.get(pair, price)
        self._last_prices[pair] = price

        for rule in self._rules:
            if rule.pair != pair or not rule.enabled:
                continue
            try:
                triggered = self._evaluate_rule(rule, price, prev_price)
                if triggered:
                    await self._dispatch(rule, {"pair": pair, "price": price, "ts": time.time()})
            except Exception:
                logger.debug("Error evaluating rule %s", rule.id, exc_info=True)

    def _evaluate_rule(self, rule: Any, current_price: float, reference_price: float) -> bool:
        from cryptotrader.triggers.conditions import (
            check_candle_pattern,
            check_funding_rate,
            check_pct_change,
            check_price_threshold,
        )

        tt = rule.trigger_type
        params = rule.parameters or {}
        if tt == "price_threshold":
            return check_price_threshold(current_price, params)
        if tt == "pct_change":
            return check_pct_change(current_price, reference_price, params)
        if tt == "candle_pattern":
            candles = self._price_history.get(rule.pair, [])
            return check_candle_pattern(candles, params)
        if tt == "funding_rate":
            rate = self._funding_rates.get(rule.pair, 0.0)
            return check_funding_rate(rate, params)
        return False

    async def _dispatch(self, rule: Any, price_snapshot: dict[str, Any]) -> None:
        cooldown_key = f"trigger:cooldown:{rule.id}"
        is_cooling = await self._check_cooldown(cooldown_key)

        reason = self._build_trigger_reason(rule, price_snapshot)

        if is_cooling:
            await self._store.record_event(
                {
                    "rule_id": rule.id,
                    "trigger_reason": reason,
                    "price_snapshot": price_snapshot,
                    "schedule_depth": rule.schedule_depth,
                    "cooldown_skipped": True,
                }
            )
            logger.debug("Rule %s skipped (cooldown active)", rule.id)
            return

        await self._set_cooldown(cooldown_key, rule.cooldown_minutes)

        event = await self._store.record_event(
            {
                "rule_id": rule.id,
                "trigger_reason": reason,
                "price_snapshot": price_snapshot,
                "schedule_depth": rule.schedule_depth,
                "cooldown_skipped": False,
            }
        )

        logger.info("Rule %s triggered: %s", rule.name, reason)

        try:
            await self._run_pair(rule.pair, {"trigger_event_id": event.id, "schedule_depth": rule.schedule_depth})
        except Exception:
            logger.warning("Trigger callback failed for rule %s", rule.id, exc_info=True)

    async def _check_cooldown(self, key: str) -> bool:
        val = await self._redis.get(key)
        return val is not None

    async def _set_cooldown(self, key: str, minutes: int) -> None:
        await self._redis.set(key, "1", ex=minutes * 60)

    async def _reconnect_with_backoff(self) -> None:
        await asyncio.sleep(self._reconnect_delay)
        self._reconnect_delay = min(self._reconnect_delay * 2, self._config.ws_reconnect_max_s)

    async def poll_funding_rates(self) -> dict[str, float]:
        """Poll Binance REST API for funding rates of tracked pairs."""
        import httpx

        pairs = {r.pair for r in self._rules}
        rates: dict[str, float] = {}
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get("https://fapi.binance.com/fapi/v1/premiumIndex")
                resp.raise_for_status()
                for item in resp.json():
                    symbol = item.get("symbol", "")
                    pair = self._symbol_to_pair(symbol)
                    if pair in pairs:
                        rates[pair] = float(item.get("lastFundingRate", 0))
        except Exception:
            logger.warning("Failed to poll funding rates", exc_info=True)
        self._funding_rates.update(rates)
        return rates

    @staticmethod
    def _symbol_to_pair(symbol: str) -> str:
        s = symbol.upper()
        if s.endswith("USDT"):
            return f"{s[:-4]}/USDT"
        if s.endswith("BUSD"):
            return f"{s[:-4]}/BUSD"
        return s

    @staticmethod
    def _build_trigger_reason(rule: Any, snapshot: dict[str, Any]) -> str:
        pair = snapshot.get("pair", "")
        price = snapshot.get("price", 0)
        params = rule.parameters or {}
        tt = rule.trigger_type

        if tt == "price_threshold":
            direction = "跌至" if params.get("direction") == "below" else "涨至"
            threshold = params.get("price", 0)
            return f"{pair} {direction} {price:,.2f} (阈值: {threshold:,.0f})"
        if tt == "pct_change":
            window = params.get("window_minutes", 0)
            pct = params.get("threshold_pct", 0)
            return f"{pair} {window}min 波动 >= {pct}%"
        if tt == "candle_pattern":
            direction = "阴线" if params.get("direction") == "bearish" else "阳线"
            count = params.get("candle_count", 0)
            tf = params.get("timeframe", "")
            return f"{pair} 连续 {count} 根{direction} ({tf})"
        if tt == "funding_rate":
            return f"{pair} funding rate >= {params.get('threshold_pct', 0)}%"
        return f"{pair} 触发规则 {rule.name}"

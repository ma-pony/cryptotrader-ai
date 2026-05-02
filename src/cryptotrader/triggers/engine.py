"""Price trigger engine — WebSocket listener + condition matcher."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections import deque
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

if TYPE_CHECKING:
    from cryptotrader.config import TriggersConfig
    from cryptotrader.risk.state import RedisStateManager
    from cryptotrader.triggers.store import TriggerRuleStore

logger = logging.getLogger(__name__)

BINANCE_WS_BASE = "wss://stream.binance.com/stream"

# Klines REST poll cadence — Binance allows 1200 weight/min and each klines
# call is weight 1. With 2 pairs and 1 interval that's 120 calls/h; trivial.
_KLINE_POLL_SECONDS = 60

# Rolling window we keep for pct_change reference lookup. Templates currently
# use 15 min; oversize so longer windows still work without reconfig.
_PCT_CHANGE_BUFFER_MIN = 60


def _to_swap_symbol(pair: str) -> str:
    """Map a spot pair (``BTC/USDT``) to its linear perp (``BTC/USDT:USDT``).

    ccxt's ``fetch_funding_rates`` is a perp-only API. Funding-rate trigger
    rules are typically configured against the spot symbol they reference,
    so we coerce here. Already-perp inputs are returned unchanged.
    """
    if ":" in pair:
        return pair
    base, _, quote = pair.partition("/")
    if not base or not quote:
        return pair
    return f"{pair}:{quote}"


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
        self._funding_task: asyncio.Task[None] | None = None
        self._kline_task: asyncio.Task[None] | None = None
        # Public-only ccxt client for kline + funding-rate REST. Lazy-init on
        # first use so test runs that never reach a poll don't pay the import.
        self._market_client: Any = None
        self._funding_rates: dict[str, float] = {}
        self._last_prices: dict[str, float] = {}
        # Rolling (timestamp_seconds, price) buffer per pair, used by pct_change
        # to compute "% move over last N minutes" — the previous-tick lookup
        # was effectively a tick-to-tick diff, never matching a 3% threshold.
        self._price_buffer: dict[str, deque[tuple[float, float]]] = {}
        # OHLC history per (pair, interval) populated by REST poll. The legacy
        # ``_price_history`` field was empty and unreferenced; use a clearer
        # 2-tuple key so multiple intervals can coexist on the same pair.
        self._klines: dict[tuple[str, str], list[dict[str, float]]] = {}
        self._reconnect_delay = 1.0

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        await self.reload_rules()
        self._ws_task = asyncio.create_task(self._ws_connect())
        # Kline + funding-rate REST polls. They were defined but never
        # scheduled — without them the candle_pattern and funding_rate
        # templates evaluated against empty data and could never fire.
        self._kline_task = asyncio.create_task(self._kline_poll_loop())
        self._funding_task = asyncio.create_task(self._funding_poll_loop())
        logger.info("PriceTriggerEngine started with %d rules", len(self._rules))

    async def stop(self) -> None:
        self._running = False
        for task_attr in ("_ws_task", "_kline_task", "_funding_task"):
            task = getattr(self, task_attr, None)
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            setattr(self, task_attr, None)
        # ccxt async clients hold an aiohttp connector; closing here avoids
        # the "Unclosed connector" warning ``arena run --mode live`` already
        # carefully avoids in cli/main._run.
        if self._market_client is not None:
            with contextlib.suppress(Exception):
                await self._market_client.close()
            self._market_client = None
        logger.info("PriceTriggerEngine stopped")

    async def _get_market_client(self) -> Any:
        """Return a lazy-initialized public ccxt async client.

        The trigger engine only needs read-only public data (klines, funding
        rates, ticker) so no API key is required. Using ccxt instead of raw
        ``httpx`` calls keeps a single source of truth for symbol mapping
        and rate-limit handling, and matches ``LiveExchange`` for the
        order-side path.
        """
        if self._market_client is None:
            import ccxt.async_support as ccxt_async

            # Binance public endpoints are the only source these polls
            # support today; can be made configurable later by reading
            # ``config.scheduler.exchange_id`` if other exchanges grow
            # equivalent ``fetch_ohlcv`` / ``fetch_funding_rates`` coverage.
            self._market_client = ccxt_async.binance({"enableRateLimit": True})
        return self._market_client

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
        now = time.time()
        self._last_prices[pair] = price
        self._append_price_buffer(pair, now, price)

        for rule in self._rules:
            if rule.pair != pair or not rule.enabled:
                continue
            try:
                triggered = self._evaluate_rule(rule, price, now)
                if triggered:
                    await self._dispatch(rule, {"pair": pair, "price": price, "ts": now})
            except Exception:
                logger.debug("Error evaluating rule %s", rule.id, exc_info=True)

    def _append_price_buffer(self, pair: str, ts: float, price: float) -> None:
        """Push (ts, price) to the rolling buffer and prune entries older than
        ``_PCT_CHANGE_BUFFER_MIN`` minutes."""
        buf = self._price_buffer.setdefault(pair, deque())
        buf.append((ts, price))
        cutoff = ts - _PCT_CHANGE_BUFFER_MIN * 60
        while buf and buf[0][0] < cutoff:
            buf.popleft()

    def _reference_price_for_window(self, pair: str, now: float, window_minutes: float) -> float:
        """Return the price from ~``window_minutes`` ago. Falls back to the
        oldest available sample if the buffer is shorter than the window
        (so a freshly-attached engine still emits *some* signal)."""
        buf = self._price_buffer.get(pair)
        if not buf:
            return 0.0
        target = now - window_minutes * 60
        for ts, price in buf:
            if ts >= target:
                return price
        # Buffer exists but every entry is past the cutoff (impossible after
        # the prune in _append_price_buffer, but defensive).
        return buf[0][1]

    def _evaluate_rule(self, rule: Any, current_price: float, now: float) -> bool:
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
            window = float(params.get("window_minutes", 0) or 0)
            if window <= 0:
                return False
            ref = self._reference_price_for_window(rule.pair, now, window)
            return check_pct_change(current_price, ref, params)
        if tt == "candle_pattern":
            interval = str(params.get("interval", ""))
            if not interval:
                return False
            candles = self._klines.get((rule.pair, interval), [])
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

    async def _funding_poll_loop(self) -> None:
        """Periodically refresh funding rates so funding_rate rules can fire.

        Cadence comes from ``triggers.funding_rate_poll_interval_minutes`` (5).
        Funding rates update every ~8h on Binance, so a 5-min poll is plenty.
        """
        # Initial poll so the first cycle has data even before the interval.
        await self.poll_funding_rates()
        interval_s = max(60, self._config.funding_rate_poll_interval_minutes * 60)
        while self._running:
            try:
                await asyncio.sleep(interval_s)
            except asyncio.CancelledError:
                return
            if not self._running:
                return
            try:
                await self.poll_funding_rates()
            except Exception:
                logger.warning("funding rate poll failed; will retry", exc_info=True)

    async def _kline_poll_loop(self) -> None:
        """Periodically refresh OHLC history for every (pair, interval) needed
        by a candle_pattern rule. Without this loop ``self._klines`` stayed
        empty and the candle_pattern template silently never fired."""
        while self._running:
            specs = self._kline_specs()
            for pair, interval, count in specs:
                try:
                    candles = await self._fetch_klines(pair, interval, count)
                    if candles:
                        self._klines[(pair, interval)] = candles
                except Exception:
                    logger.debug("kline fetch failed for %s %s", pair, interval, exc_info=True)
            try:
                await asyncio.sleep(_KLINE_POLL_SECONDS)
            except asyncio.CancelledError:
                return

    def _kline_specs(self) -> list[tuple[str, str, int]]:
        """Return (pair, interval, candles_needed) for every active candle_pattern rule.

        ``candles_needed`` is ``consecutive_count + 1`` so a rule that needs 3
        consecutive bears has a 4th candle for context if we ever want it.
        """
        specs: dict[tuple[str, str], int] = {}
        for rule in self._rules:
            if rule.trigger_type != "candle_pattern" or not rule.enabled:
                continue
            params = rule.parameters or {}
            interval = str(params.get("interval", ""))
            if not interval:
                continue
            count = int(params.get("consecutive_count", params.get("candle_count", 3)))
            key = (rule.pair, interval)
            specs[key] = max(specs.get(key, 0), count + 1)
        return [(p, i, c) for (p, i), c in specs.items()]

    async def _fetch_klines(self, pair: str, interval: str, limit: int) -> list[dict[str, float]]:
        """Fetch the last ``limit`` klines via ccxt ``fetch_ohlcv``.

        ccxt returns ``[[ts, open, high, low, close, volume], ...]`` ascending
        — already the order ``check_candle_pattern`` expects (last N entries =
        most recent N candles).
        """
        client = await self._get_market_client()
        raw = await client.fetch_ohlcv(pair, timeframe=interval, limit=limit)
        return [
            {
                "open_time": float(row[0]),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
            }
            for row in raw or []
        ]

    async def poll_funding_rates(self) -> dict[str, float]:
        """Refresh funding rates for every active rule's pair via ccxt.

        ``fetch_funding_rates`` operates on swap (perp) symbols. Spot pairs
        configured on a rule (e.g. ``BTC/USDT``) are mapped to their linear
        perp equivalent (``BTC/USDT:USDT``). Pairs without a perp listing
        (or any other ccxt error) are skipped — the rate stays absent and
        the rule simply doesn't fire that cycle.
        """
        rates: dict[str, float] = {}
        if not self._rules:
            return rates
        client = await self._get_market_client()
        # Build a unique perp-symbol list to ask ccxt for in one call.
        perp_for: dict[str, str] = {}
        for rule in self._rules:
            perp_for.setdefault(rule.pair, _to_swap_symbol(rule.pair))
        try:
            payload = await client.fetch_funding_rates(list(perp_for.values()))
        except Exception:
            logger.warning("Failed to poll funding rates", exc_info=True)
            return rates
        for pair, perp in perp_for.items():
            entry = (payload or {}).get(perp)
            if not entry:
                continue
            rate = entry.get("fundingRate")
            if rate is None:
                continue
            try:
                rates[pair] = float(rate)
            except (TypeError, ValueError):
                continue
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
            # Frontend persists ``consecutive_count``; the legacy ``candle_count``
            # name kept the original code reading 0 from the saved rule.
            count = params.get("consecutive_count", params.get("candle_count", 0))
            tf = params.get("interval", params.get("timeframe", ""))
            return f"{pair} 连续 {count} 根{direction} ({tf})"
        if tt == "funding_rate":
            return f"{pair} funding rate >= {params.get('threshold_pct', 0)}%"
        return f"{pair} 触发规则 {rule.name}"

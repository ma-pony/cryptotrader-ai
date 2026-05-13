"""Exchange adapter protocol and hardened live implementation."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from cryptotrader.models import Order

logger = logging.getLogger(__name__)

# Max attempts for set_leverage before giving up and caching the symbol.
# A typical reason for failure is "position already open" — once that
# position closes, the next order will retry. 3 attempts is enough to
# distinguish transient errors from permanent locks without spamming.
_LEVERAGE_RETRY_LIMIT = 3

# How long an exchange-side rejection (e.g. OKX 51202 "Market order amount
# exceeds the maximum amount") keeps the (pair, side) signature in a
# fallback-to-limit cache. 30 minutes covers most retry windows without
# permanently masking a transient venue blip.
_ORDER_FAILURE_CACHE_TTL_S = 30 * 60

# Conservative slippage band when synthesizing a limit order to replace a
# rejected/oversized market order. 0.3% is wide enough to fill in the same
# minute on liquid pairs but tight enough to behave like a market fill.
_MARKET_TO_LIMIT_SLIPPAGE = 0.003

# Module-level failure ledger. Key: (exchange_id, pair, side). Value: last
# rejection timestamp. Survives across LiveExchange instances within a
# process so a brief reconnection does not lose the learning.
_oversized_market_failures: dict[tuple[str, str, str], float] = {}

# spec 021 H3 (option 2): exchange trade-endpoint unavailability cooldown.
# When _retry exhausts attempts against OKX sCode=50013 / ExchangeNotAvailable,
# stamp a `wall-clock` deadline. The next risk-gate evaluation in the same
# cycle string consults this map and short-circuits actionable verdicts to
# hold, sparing each downstream pair the ~24s slow-backoff round-trip.
# 5-minute TTL is comfortably longer than one 5-pair cycle (~3 min) while
# short enough that the next hourly cycle will re-probe instead of stalling.
_trade_unavailable_until: dict[str, float] = {}
_TRADE_UNAVAIL_TTL_S = 300


def _mark_trade_unavailable(exchange_id: str) -> None:
    """Record that the venue's trade endpoint is currently rejecting orders."""
    _trade_unavailable_until[exchange_id] = time.time() + _TRADE_UNAVAIL_TTL_S


def trade_unavailable_remaining_s(exchange_id: str) -> float:
    """Seconds remaining on the trade-endpoint cooldown, 0 if available."""
    deadline = _trade_unavailable_until.get(exchange_id, 0.0)
    remaining = deadline - time.time()
    return remaining if remaining > 0 else 0.0


def _is_venue_unavailable(exc: BaseException) -> bool:
    """Whether the exception looks like a persistent venue unavailability
    (50013 throttle, 50001/50002/50026 maintenance, ccxt OnMaintenance /
    ExchangeNotAvailable). Any of these justify the cooldown mark — we
    want to back off all retries and let downstream pairs short-circuit
    instead of hammering an already-broken endpoint.

    String-based codes cover the cases where ccxt wraps the OKX response
    in a generic ExchangeError instead of OnMaintenance. The isinstance
    branch covers the cases where ccxt classified it correctly.
    """
    msg = str(exc)
    for code in ('"50013"', '"50001"', '"50002"', '"50026"'):
        if code in msg:
            return True
    if "Systems are busy" in msg or "Service temporarily unavailable" in msg:
        return True
    try:
        import ccxt

        return isinstance(exc, (ccxt.OnMaintenance, ccxt.ExchangeNotAvailable, ccxt.DDoSProtection))
    except ImportError:
        return False


@runtime_checkable
class ExchangeAdapter(Protocol):
    async def place_order(self, order: Order) -> dict[str, Any]: ...

    async def cancel_order(self, order_id: str, symbol: str | None = None) -> dict[str, Any]: ...

    async def get_order(self, order_id: str, symbol: str | None = None) -> dict[str, Any]: ...

    async def get_balance(self) -> dict[str, float]: ...

    async def get_positions(self) -> dict[str, dict[str, Any]]: ...

    async def fetch_open_orders(self) -> list[dict[str, Any]]: ...

    async def close(self) -> None: ...


class LiveExchange:
    def __init__(
        self,
        exchange_id: str,
        api_key: str,
        secret: str,
        *,
        sandbox: bool,
        passphrase: str = "",
        leverage: int = 1,
        margin_mode: str = "isolated",
    ) -> None:
        try:
            import ccxt.async_support as ccxt_async
        except ImportError:
            raise ImportError(
                "ccxt.async_support is required for LiveExchange. Install with: pip install ccxt"
            ) from None
        exchange_cls = getattr(ccxt_async, exchange_id)
        config = {
            "apiKey": api_key,
            "secret": secret,
            "sandbox": sandbox,
            "enableRateLimit": True,
            # We only trade spot and linear perp (swap). Excluding future/option
            # avoids `load_markets` failing wholesale when one of those instType
            # endpoints is briefly unavailable upstream — the surfaced
            # ExchangeNotAvailable cascades to portfolio_unknown rejections.
            "options": {"fetchMarkets": ["spot", "swap"]},
        }
        if passphrase:
            config["password"] = passphrase
        self._exchange = exchange_cls(config)
        self._exchange_id = exchange_id
        self._markets_loaded = False
        self._leverage = leverage
        self._margin_mode = margin_mode
        # Symbols where leverage has been confirmed (or given up after
        # _LEVERAGE_RETRY_LIMIT failed attempts).
        self._leveraged_symbols: set[str] = set()
        # Per-symbol failure counter: failures don't immediately cache so a
        # later retry (e.g. after the user closes the open position that was
        # locking leverage) can still apply the new value.
        self._leverage_attempts: dict[str, int] = {}

    async def _ensure_markets(self) -> None:
        if not self._markets_loaded:
            await self._exchange.load_markets()
            self._markets_loaded = True

    async def _ensure_leverage(self, symbol: str) -> None:
        """Apply configured leverage to a perp ``symbol``.

        - ``leverage <= 1``: no-op (matches OKX default; saves an API call).
        - Spot symbol: no-op (cached so we don't re-parse).
        - Long-short position mode: set both posSides (OKX requires this).
        - On full success (both posSides) → cache and never retry.
        - On failure (e.g. position already open → OKX locks leverage) → log
          warning, increment a per-symbol fail counter; once it hits
          ``_LEVERAGE_RETRY_LIMIT``, give up and cache. This lets users change
          ``leverage`` config without restarting the process: the next order
          after they close the position triggers a successful retry.
        """
        if self._leverage <= 1 or symbol in self._leveraged_symbols:
            return
        from cryptotrader.pair import Pair

        try:
            if Pair.parse(symbol).market_type == "spot":
                self._leveraged_symbols.add(symbol)
                return
        except (ValueError, NotImplementedError):
            return
        all_ok = True
        for pos_side in ("long", "short"):
            try:
                await self._exchange.set_leverage(
                    self._leverage,
                    symbol,
                    {"mgnMode": self._margin_mode, "posSide": pos_side},
                )
                logger.info("set_leverage %dx %s posSide=%s ok", self._leverage, symbol, pos_side)
            except Exception as e:
                all_ok = False
                logger.warning(
                    "set_leverage %dx failed for %s posSide=%s: %s: %s",
                    self._leverage,
                    symbol,
                    pos_side,
                    type(e).__name__,
                    e,
                )
        if all_ok:
            self._leveraged_symbols.add(symbol)
            return
        attempts = self._leverage_attempts.get(symbol, 0) + 1
        self._leverage_attempts[symbol] = attempts
        if attempts >= _LEVERAGE_RETRY_LIMIT:
            logger.warning(
                "set_leverage giving up on %s after %d attempts; orders will use exchange-side leverage",
                symbol,
                attempts,
            )
            self._leveraged_symbols.add(symbol)

    async def _retry(self, coro_fn, *args, attempts: int | None = None):
        import ccxt

        if attempts is None:
            from cryptotrader.config import load_config

            attempts = load_config().execution.retry_attempts

        _fatal = (ccxt.AuthenticationError, ccxt.PermissionDenied, ccxt.BadSymbol, ccxt.InsufficientFunds)
        # spec 021 H3: OKX sandbox throttles burst swap orders with
        # sCode=50013 "Systems are busy" → ExchangeNotAvailable. Default
        # 2^i backoff (max 16s) was too aggressive — every retry hit the
        # same throttle window. Scale these classes with 5·3^i (5/15/45/60s)
        # so the throttle has time to clear. RateLimitExceeded gets the
        # same treatment since OKX uses both interchangeably for the same
        # condition.
        _slow_backoff = (ccxt.ExchangeNotAvailable, ccxt.RateLimitExceeded, ccxt.DDoSProtection)
        for i in range(attempts):
            # Intra-retry cooldown check: a sibling pair in this cycle may
            # have already exhausted its retries and stamped the venue as
            # unavailable. Abort early instead of burning another wait
            # cycle against a known-down endpoint. Only applies after the
            # first attempt (so we don't break callers that need to probe
            # before any mark exists).
            if i > 0 and trade_unavailable_remaining_s(self._exchange_id) > 0:
                raise ccxt.ExchangeNotAvailable(f"{self._exchange_id} venue cooldown active — short-circuiting retry")
            try:
                return await coro_fn(*args)
            except _fatal:
                raise  # Fatal errors — don't retry
            except _slow_backoff as e:
                # Mark cooldown on EVERY persistent-unavailability error
                # (50013 throttle, 50001/50002/50026 maintenance, etc.),
                # not only the final raise. Concurrent pairs in the same
                # cycle string short-circuit as soon as the first one
                # starts failing — both read (get_positions) and write
                # (place_order) paths consult the same marker.
                if _is_venue_unavailable(e):
                    _mark_trade_unavailable(self._exchange_id)
                if i == attempts - 1:
                    if _is_venue_unavailable(e):
                        logger.warning(
                            "venue marked unavailable for %ds (%s)",
                            _TRADE_UNAVAIL_TTL_S,
                            type(e).__name__,
                        )
                    raise
                wait = min(60, 5 * (3**i))
                logger.warning("Slow-retry %d/%d after %ss (throttle): %s", i + 1, attempts, wait, e)
                await asyncio.sleep(wait)
            except Exception as e:
                if i == attempts - 1:
                    raise
                wait = 2**i
                logger.warning("Retry %d/%d after %ss: %s", i + 1, attempts, wait, e)
                await asyncio.sleep(wait)
        return None

    async def place_order(self, order: Order) -> dict[str, Any]:
        # spec 021 H3 (option 2): venue trade-endpoint cooldown short-circuit.
        # If a sibling pair in this cycle string already exhausted retries
        # on 50013 (or one of its retries marked the cooldown mid-flight),
        # raise immediately so we don't spend another ~24s walking the
        # slow-retry ladder against a known-down endpoint.
        cooldown = trade_unavailable_remaining_s(self._exchange_id)
        if cooldown > 0:
            import ccxt

            raise ccxt.ExchangeNotAvailable(
                f"{self._exchange_id} trade endpoint in cooldown ({cooldown:.0f}s remaining) — 50013 fail-fast"
            )

        await self._ensure_markets()
        await self._ensure_leverage(order.pair)

        from cryptotrader.pair import Pair

        pair = Pair.parse(order.pair)

        # Balance pre-check. Spot is asset-denominated (need quote on buy,
        # base on sell). Derivatives are margin-denominated in the settle
        # currency; we only verify margin currency is non-zero and let the
        # exchange enforce leverage/maintenance margin.
        bal = await self.get_balance()
        if pair.market_type == "spot":
            if order.side == "buy":
                needed = order.amount * order.price
                if bal.get(pair.quote, 0) < needed:
                    raise ValueError(f"Insufficient {pair.quote}: need {needed}, have {bal.get(pair.quote, 0)}")
            else:
                if bal.get(pair.base, 0) < order.amount:
                    raise ValueError(f"Insufficient {pair.base}: need {order.amount}, have {bal.get(pair.base, 0)}")
        else:
            margin_ccy = pair.settle or pair.quote
            if bal.get(margin_ccy, 0) <= 0:
                raise ValueError(f"No {margin_ccy} margin available for {pair.canonical()}")

        # Precision
        market = self._exchange.markets.get(order.pair, {})
        # spec 021 H1 真因: OKX swap orders quote `amount` as COUNT OF
        # CONTRACTS, not base-currency units. contractSize varies per pair:
        #   DOGE=1000, ETH=0.1, BTC=0.01, LINK/SOL=1.
        # Sending 60418 DOGE base units gets interpreted as 60418 contracts
        # × 1000 ctVal = 60M DOGE notional → sCode=51008 "Insufficient
        # USDT margin" even with healthy cash balance. Convert base→contracts
        # before handing to ccxt; spot keeps base-unit semantics.
        contract_size = 1.0
        if pair.market_type != "spot":
            cs = market.get("contractSize")
            if cs is not None:
                try:
                    cs_f = float(cs)
                    if cs_f > 0:
                        contract_size = cs_f
                except (TypeError, ValueError):
                    contract_size = 1.0
        raw_amount = order.amount / contract_size if contract_size != 1.0 else order.amount
        amount = self._exchange.amount_to_precision(order.pair, raw_amount) if market else raw_amount
        price = (
            self._exchange.price_to_precision(order.pair, order.price)
            if market and order.order_type == "limit"
            else order.price
        )

        # Min order size check (units match contracts for swap, base for spot)
        min_amount = market.get("limits", {}).get("amount", {}).get("min", 0)
        if min_amount and float(amount) < min_amount:
            raise ValueError(f"Order amount {amount} below minimum {min_amount}")

        # Max market-order amount precheck + failure-learning cache.
        # Some venues (notably OKX) have a *separate* maximum for market
        # orders that is much smaller than the limit-order maximum. When
        # exceeded the venue returns 51202 ("Market order amount exceeds the
        # maximum amount") and the order silently fails. Two defenses:
        #   1. If we know the venue cap (`maxMktSz` on OKX, or unified
        #      `limits.amount.max`) and the amount exceeds it, downgrade to a
        #      limit order at price ± slippage band proactively.
        #   2. If a previous attempt for the same (pair, side) hit the cap in
        #      the last 30 min, downgrade preemptively even if the cap value
        #      isn't visible — we already know market orders are unsafe.
        order_type, amount, price = self._maybe_downgrade_to_limit(
            order,
            market,
            amount,
            price,
            pair.market_type,
        )

        # OKX perp/swap (and similar derivative venues) require ``posSide`` in
        # long_short position mode. Without it OKX returns sCode=51000
        # "Parameter posSide error". Derive it from the existing position when
        # one exists (we're closing or adjusting), else infer from order.side.
        # Spot trades skip this (params remains empty).
        params: dict[str, Any] = {}
        if pair.market_type != "spot":
            params["posSide"] = await self._derive_pos_side(order)
            # spec 021 H1: pass tdMode so OKX uses the configured margin mode
            # (isolated / cross) on this order. Without it, OKX defaults to
            # the account-default tdMode which may NOT match the mgnMode we
            # set in _ensure_leverage above — manifesting as sCode=51008
            # "Insufficient USDT margin" even with $95k free cash, because
            # isolated mode wants the per-pair subaccount which is empty
            # until first funding (cross mode uses the main account pool).
            # Mirrors mgnMode arg in set_leverage call (line 134) so the
            # order-side and leverage-side modes stay aligned.
            params["tdMode"] = self._margin_mode

        try:
            result = await self._retry(
                self._exchange.create_order,
                order.pair,
                order_type,
                order.side,
                float(amount),
                float(price) if order_type == "limit" else None,
                params,
            )
        except Exception as e:
            # Detect OKX 51202 / Binance "MAX_NUM_ALGO_ORDERS" / generic
            # "exceeds maximum amount" rejection and remember it so the next
            # call for this (pair, side) auto-downgrades.
            if self._is_oversized_market_rejection(e):
                self._record_oversized_failure(order.pair, order.side)
                logger.warning(
                    "Oversized market-order rejection cached for %s %s — next order will use limit fallback",
                    order.pair,
                    order.side,
                )
            raise

        # Order timeout: cancel if not filled
        if result.get("status") != "closed":
            order_id = result.get("id")
            if order_id:
                from cryptotrader.config import load_config

                wait_s = load_config().execution.order_wait_seconds
                result = await self._wait_or_cancel(order_id, order.pair, wait_seconds=wait_s)

        return result

    # ── oversized-market handling ────────────────────────────────────────

    def _read_max_market_amount(self, market: dict[str, Any]) -> float | None:
        """Best-effort read of the per-venue max market-order size.

        Sources, in priority order:
          1. ``market['limits']['amount']['max']`` — ccxt unified field.
          2. OKX raw ``info.maxMktSz`` — string of contract count for swaps.
        Returns None if no usable cap is found.
        """
        cap = market.get("limits", {}).get("amount", {}).get("max")
        if cap is not None:
            try:
                v = float(cap)
                if v > 0:
                    return v
            except (TypeError, ValueError):
                pass
        info = market.get("info", {}) or {}
        raw = info.get("maxMktSz")
        if raw:
            try:
                v = float(raw)
                if v > 0:
                    return v
            except (TypeError, ValueError):
                pass
        return None

    def _is_recent_oversized_failure(self, pair_str: str, side: str) -> bool:
        key = (self._exchange.id, pair_str, side)
        ts = _oversized_market_failures.get(key)
        if ts is None:
            return False
        if time.time() - ts > _ORDER_FAILURE_CACHE_TTL_S:
            _oversized_market_failures.pop(key, None)
            return False
        return True

    def _record_oversized_failure(self, pair_str: str, side: str) -> None:
        _oversized_market_failures[(self._exchange.id, pair_str, side)] = time.time()

    @staticmethod
    def _is_oversized_market_rejection(exc: BaseException) -> bool:
        """Heuristically classify an exchange exception as oversized-market.

        OKX 51202 message:  "Market order amount exceeds the maximum amount"
        We match on substrings rather than ccxt error class because ccxt
        wraps everything in ``InvalidOrder`` regardless of subcode.
        """
        msg = str(exc).lower()
        markers = (
            "exceeds the maximum amount",  # OKX 51202 wording
            "exceeds maximum amount",
            "exceeds the maximum order",
            "exceeds maximum order",
            "51202",  # OKX subcode
            "max_num_orders",  # Binance variants
            "max amount",
        )
        return any(m in msg for m in markers)

    def _maybe_downgrade_to_limit(
        self,
        order: Order,
        market: dict[str, Any],
        amount: Any,
        price: float,
        market_type: str,
    ) -> tuple[str, Any, float]:
        """Decide whether to convert a market order to a limit order.

        Triggers a downgrade when EITHER:
          - we know the venue's market-order cap and ``amount`` exceeds it, OR
          - this (pair, side) had an oversized-market rejection in the last
            ``_ORDER_FAILURE_CACHE_TTL_S`` seconds (failure-learning cache).

        For sells we set the limit price slightly *below* current to ensure a
        cross; for buys, slightly *above*. Returns ``(order_type, amount, price)``
        ready for ``create_order``.
        """
        if order.order_type != "market":
            return order.order_type, amount, price

        oversized = False
        cap = self._read_max_market_amount(market)
        if cap is not None:
            try:
                oversized = float(amount) > cap
            except (TypeError, ValueError):
                oversized = False

        recent_fail = self._is_recent_oversized_failure(order.pair, order.side)
        if not (oversized or recent_fail):
            return "market", amount, price

        # Compute crossing limit price. Sell crosses below mid; buy crosses
        # above mid. The 0.3% band is wide enough to fill on liquid OKX perps
        # without becoming a true resting order.
        slip = _MARKET_TO_LIMIT_SLIPPAGE
        if order.side == "sell":
            limit_price = price * (1.0 - slip)
        else:
            limit_price = price * (1.0 + slip)

        # If the cap is known and amount still exceeds the limit-order cap (rare
        # for OKX where maxLmtSz >> maxMktSz), clip — the caller's strategy is
        # responsible for splitting; we don't silently break a position into
        # multiple LLM-unaware orders.
        if cap is not None and oversized:
            try:
                clip_to = self._exchange.amount_to_precision(order.pair, cap)
                amount = clip_to
                logger.warning(
                    "Order amount > venue maxMktSz (%s); clipped to %s and downgraded to limit",
                    cap,
                    clip_to,
                )
            except Exception:
                logger.warning("amount_to_precision failed; using raw cap %s", cap, exc_info=True)
                amount = cap
        else:
            logger.info(
                "Recent oversized-market failure on %s %s — preemptively using limit order",
                order.pair,
                order.side,
            )

        # ccxt expects price to go through price_to_precision for limit orders.
        try:
            limit_price = float(self._exchange.price_to_precision(order.pair, limit_price))
        except Exception:
            pass

        # Track the original spot/perp split — caller already passed this in.
        _ = market_type
        return "limit", amount, limit_price

    async def _derive_pos_side(self, order: Order) -> str:
        """Return ``posSide`` for an OKX-style perp order.

        - If a position already exists for this pair, use its side
          (closing long → posSide=long, closing short → posSide=short).
          Same rule covers "add to existing" since posSide stays the same.
        - If no position exists (new entry), infer from ``order.side``:
          buy → long, sell → short.

        Falls back to side-inference if the position lookup raises so that
        a transient venue error does not block order placement entirely.
        """
        try:
            positions = await self.get_positions()
        except Exception:
            logger.warning("derive_pos_side: get_positions failed, using side-inference", exc_info=True)
            return "long" if order.side == "buy" else "short"

        existing = positions.get(order.pair) or {}
        amount = existing.get("amount", 0)
        if amount > 0:
            return "long"
        if amount < 0:
            return "short"
        return "long" if order.side == "buy" else "short"

    async def _wait_or_cancel(self, order_id: str, pair: str, wait_seconds: int = 30) -> dict:
        for _ in range(wait_seconds // 2):
            await asyncio.sleep(2)
            info = await self._retry(self._exchange.fetch_order, order_id, pair)
            if info.get("status") in ("closed", "canceled", "cancelled"):
                return info
        # Timeout — cancel
        logger.warning("Order %s timed out, cancelling", order_id)
        with contextlib.suppress(Exception):
            await self._retry(self._exchange.cancel_order, order_id, pair)
        return await self._retry(self._exchange.fetch_order, order_id, pair)

    async def cancel_order(self, order_id: str, symbol: str | None = None) -> dict[str, Any]:
        return await self._retry(self._exchange.cancel_order, order_id, symbol)

    async def get_order(self, order_id: str, symbol: str | None = None) -> dict[str, Any]:
        return await self._retry(self._exchange.fetch_order, order_id, symbol)

    async def get_balance(self) -> dict[str, float]:
        bal = await self._retry(self._exchange.fetch_balance)
        return {k: float(v) for k, v in bal.get("total", {}).items() if float(v) > 0}

    async def get_free_balance(self) -> dict[str, float]:
        """Return *free* (unlocked, openable) balance per asset.

        ``get_balance`` returns ``total`` which includes margin locked by open
        positions — using it for pre-trade margin sizing causes false "have
        enough" decisions that OKX then rejects with ``sCode=51008
        Insufficient USDT margin`` once the order hits the matching engine
        (observed 2026-05-11 DOGE short rejection).
        """
        bal = await self._retry(self._exchange.fetch_balance)
        return {k: float(v) for k, v in bal.get("free", {}).items() if float(v) > 0}

    async def fetch_ticker(self, symbol: str) -> dict[str, Any]:
        """Live market ticker. Used by portfolio sync to price non-traded
        balances when no historical cost basis exists (avoids inheriting the
        traded pair's price — spec ledger 2026-04-30 sync bug).
        """
        await self._ensure_markets()
        return await self._retry(self._exchange.fetch_ticker, symbol)

    async def get_positions(self) -> dict[str, dict[str, Any]]:
        """Fetch open positions, keyed by ccxt unified symbol.

        Per spec 013, ccxt's unified symbol IS the project canonical (spot
        ``BTC/USDT``, linear perp ``BTC/USDT:USDT``, inverse perp
        ``BTC/USD:BTC``). No translation; downstream callers use the same
        key as ``state["metadata"]["pair"]`` and ``Pair.canonical()``.

        Returns: {pair: {"amount": float, "side": str, "avg_price": float,
                         "unrealized_pnl": float, "liquidation_price": float | None}}
        """
        # spec 021 H3 (option 2 extension): when the venue read endpoint is
        # in cooldown (set by a prior get_positions / place_order that hit
        # 50001 OnMaintenance / 50013 / RateLimitExceeded), bail immediately.
        # The existing portfolio_unknown risk check converts this to a
        # conservative REJECT, sparing 4 trailing pairs ~20s slow-retry
        # each (~80s/cycle when sandbox is fully down).
        cooldown = trade_unavailable_remaining_s(self._exchange_id)
        if cooldown > 0:
            import ccxt

            raise ccxt.ExchangeNotAvailable(
                f"{self._exchange_id} venue in cooldown ({cooldown:.0f}s remaining) — fetch_positions fail-fast"
            )

        await self._ensure_markets()
        positions: dict[str, dict[str, Any]] = {}
        try:
            raw = await self._retry(self._exchange.fetch_positions)
            for p in raw or []:
                contracts = float(p.get("contracts", 0) or 0)
                if contracts == 0:
                    continue
                symbol = p.get("symbol", "")
                if not symbol:
                    continue
                # spec 021 H1: keep internal "amount" in base-currency units
                # so it composes with `target_amount = notional / price` in
                # nodes/execution. ccxt exposes contractSize per market;
                # multiply contracts → base units. Spot markets have no
                # contractSize and the field stays 1.0 by convention.
                market = (self._exchange.markets or {}).get(symbol, {}) or {}
                cs_raw = market.get("contractSize")
                contract_size = 1.0
                if cs_raw is not None:
                    try:
                        cs_f = float(cs_raw)
                        if cs_f > 0:
                            contract_size = cs_f
                    except (TypeError, ValueError):
                        contract_size = 1.0
                base_amount = contracts * contract_size
                side = p.get("side", "long")
                amount = base_amount if side == "long" else -base_amount
                entry_px = float(p.get("entryPrice", 0) or 0)
                upnl = float(p.get("unrealizedPnl", 0) or 0)
                liq = float(p["liquidationPrice"]) if p.get("liquidationPrice") else None
                # spec 021 H3 (B1): same symbol may appear multiple times when
                # OKX has parallel isolated + cross positions on the same
                # contract (long_short_mode allows it). Aggregate by signed
                # amount and weight avg_price by |size| so the downstream
                # model sees the true net exposure instead of just the last
                # row that won the dict overwrite.
                if symbol in positions:
                    prev = positions[symbol]
                    new_amount = prev["amount"] + amount
                    prev_abs, cur_abs = abs(prev["amount"]), abs(amount)
                    total_abs = prev_abs + cur_abs
                    if total_abs > 0:
                        avg_px = (prev["avg_price"] * prev_abs + entry_px * cur_abs) / total_abs
                    else:
                        avg_px = entry_px
                    positions[symbol] = {
                        "amount": new_amount,
                        "side": "long" if new_amount > 0 else ("short" if new_amount < 0 else prev["side"]),
                        "avg_price": avg_px,
                        "unrealized_pnl": prev["unrealized_pnl"] + upnl,
                        # liquidation_price is per-position; aggregated number
                        # is meaningless — drop to None when ambiguous.
                        "liquidation_price": None if prev["liquidation_price"] != liq else liq,
                    }
                else:
                    positions[symbol] = {
                        "amount": amount,
                        "side": side,
                        "avg_price": entry_px,
                        "unrealized_pnl": upnl,
                        "liquidation_price": liq,
                    }
        except Exception as exc:
            # Spec 013 deep-review production FINDING-2: narrow the catch.
            # Transient ccxt errors (network blip / rate limit) MUST propagate so
            # the caller knows positions are unknown, not silently fall back to
            # balance-derived spot positions (which masks open perp exposure and
            # re-introduces the close-on-flat bug via a different path).
            import ccxt

            transient = (
                getattr(ccxt, "NetworkError", Exception),
                getattr(ccxt, "RateLimitExceeded", Exception),
                getattr(ccxt, "ExchangeNotAvailable", Exception),
                getattr(ccxt, "RequestTimeout", Exception),
            )
            if isinstance(exc, transient):
                logger.warning("fetchPositions transient failure (%s) — re-raising", type(exc).__name__)
                raise
            # Spot exchanges that don't support fetchPositions raise ccxt.NotSupported
            # or BadSymbol. Fall through and derive from balance.
            logger.warning("fetchPositions not available, deriving from balance", exc_info=True)
            bal = await self.get_balance()
            for asset, amount in bal.items():
                if asset == "USDT" or amount == 0:
                    continue
                pair = f"{asset}/USDT"
                positions[pair] = {
                    "amount": amount,
                    "side": "long" if amount > 0 else "short",
                    "avg_price": 0.0,  # Not available from spot balance
                    "unrealized_pnl": 0.0,
                    "liquidation_price": None,
                }
        return positions

    async def fetch_open_orders(self) -> list[dict[str, Any]]:
        await self._ensure_markets()
        return await self._retry(self._exchange.fetch_open_orders)

    # ── OKX algo OCO (Phase 2A) ──────────────────────────────────────────
    #
    # Server-side stop-loss + take-profit via OKX algo orders. ccxt's unified
    # ``create_order`` doesn't model OCO algos, so we hit the raw OKX endpoint
    # directly. Other venues (Binance / Bybit) are not supported here — the
    # methods raise NotImplementedError for non-OKX exchange_ids so callers
    # don't silently send malformed params elsewhere.

    def _require_okx(self, op: str) -> None:
        if self._exchange_id != "okx":
            raise NotImplementedError(f"{op} is OKX-only; current exchange_id={self._exchange_id!r}")

    def _to_okx_inst_id(self, pair: str) -> str:
        """Convert ccxt unified symbol → OKX instId (e.g. 'BTC/USDT:USDT' → 'BTC-USDT-SWAP')."""
        market = self._exchange.markets.get(pair, {})
        inst_id = market.get("id")
        if not inst_id:
            raise ValueError(f"Cannot resolve OKX instId for pair={pair!r} (market not loaded)")
        return str(inst_id)

    async def place_algo_oco(
        self,
        pair: str,
        *,
        side: str,
        amount: float,
        sl_trigger_px: float,
        tp_trigger_px: float,
        pos_side: str,
    ) -> str:
        """Submit an OCO algo (stop-loss + take-profit) with reduceOnly.

        Args:
            pair: ccxt unified symbol (e.g. ``"BTC/USDT:USDT"``).
            side: Closing direction: ``"buy"`` to close a short, ``"sell"`` to
                close a long. Caller is responsible for picking the inverse of
                the open position.
            amount: Size in **contracts** (already converted from base units;
                same convention as ``place_order``).
            sl_trigger_px: Stop-loss trigger price (last-price reference).
            tp_trigger_px: Take-profit trigger price (last-price reference).
            pos_side: ``"long"`` or ``"short"`` — the position being protected.

        Returns:
            ``algoId`` string, or raises on rejection.

        OKX behaviour: the two child orders sit pending until either triggers,
        then the other is auto-cancelled (OCO semantics). Both legs execute at
        market on trigger (``slOrdPx="-1"`` / ``tpOrdPx="-1"``).
        """
        self._require_okx("place_algo_oco")
        await self._ensure_markets()

        inst_id = self._to_okx_inst_id(pair)
        params = {
            "instId": inst_id,
            "tdMode": self._margin_mode,
            "side": side,
            "posSide": pos_side,
            "ordType": "oco",
            "sz": str(amount),
            "reduceOnly": "true",
            "slTriggerPx": str(sl_trigger_px),
            "slTriggerPxType": "last",
            "slOrdPx": "-1",
            "tpTriggerPx": str(tp_trigger_px),
            "tpTriggerPxType": "last",
            "tpOrdPx": "-1",
        }

        resp = await self._retry(self._exchange.private_post_trade_order_algo, params)
        # OKX response shape: {"code": "0", "data": [{"algoId": "...", "sCode": "0", "sMsg": "..."}], ...}
        if str(resp.get("code", "")) != "0":
            raise RuntimeError(f"OKX algo OCO rejected: code={resp.get('code')} msg={resp.get('msg')}")
        data = resp.get("data") or []
        if not data:
            raise RuntimeError(f"OKX algo OCO returned no data: {resp!r}")
        leg = data[0]
        if str(leg.get("sCode", "")) != "0":
            raise RuntimeError(f"OKX algo OCO leg rejected: sCode={leg.get('sCode')} sMsg={leg.get('sMsg')}")
        algo_id = leg.get("algoId")
        if not algo_id:
            raise RuntimeError(f"OKX algo OCO succeeded but no algoId: {leg!r}")
        logger.info(
            "place_algo_oco %s side=%s posSide=%s sl=%s tp=%s sz=%s → algoId=%s",
            inst_id,
            side,
            pos_side,
            sl_trigger_px,
            tp_trigger_px,
            amount,
            algo_id,
        )
        return str(algo_id)

    async def cancel_algo(self, algo_id: str, pair: str) -> None:
        """Cancel a pending algo by ``algoId``. Swallows "not found" errors."""
        self._require_okx("cancel_algo")
        await self._ensure_markets()

        inst_id = self._to_okx_inst_id(pair)
        params = [{"algoId": algo_id, "instId": inst_id}]
        try:
            resp = await self._retry(self._exchange.private_post_trade_cancel_algos, params)
        except Exception as exc:
            # OKX returns 51400 / 51401 for already-triggered or unknown algos.
            # Treat as success — caller's intent is "ensure it's gone".
            msg = str(exc)
            if "51400" in msg or "51401" in msg or "not exist" in msg.lower():
                logger.info("cancel_algo %s: already gone (%s)", algo_id, msg[:120])
                return
            raise

        # OKX returns code=0 even when individual leg failed; check sCode per leg.
        for leg in resp.get("data", []) or []:
            sc = str(leg.get("sCode", ""))
            if sc not in ("0", "51400", "51401"):
                logger.warning(
                    "cancel_algo %s leg returned sCode=%s sMsg=%s",
                    algo_id,
                    sc,
                    leg.get("sMsg"),
                )
        logger.info("cancel_algo %s ok (instId=%s)", algo_id, inst_id)

    async def list_pending_algos(self, pair: str | None = None) -> list[dict[str, Any]]:
        """List currently pending algos. ``pair`` filters by instId when given."""
        self._require_okx("list_pending_algos")
        await self._ensure_markets()

        params: dict[str, Any] = {"ordType": "oco"}
        if pair is not None:
            params["instId"] = self._to_okx_inst_id(pair)

        resp = await self._retry(self._exchange.private_get_trade_orders_algo_pending, params)
        if str(resp.get("code", "")) != "0":
            raise RuntimeError(f"OKX list algos failed: code={resp.get('code')} msg={resp.get('msg')}")
        return list(resp.get("data") or [])

    async def close(self) -> None:
        try:
            await self._exchange.close()
        except Exception as e:
            logger.warning("Exchange close failed: %s", e)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()

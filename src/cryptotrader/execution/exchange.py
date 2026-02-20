"""Exchange adapter protocol and hardened live implementation."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Protocol, runtime_checkable

from cryptotrader.models import Order

logger = logging.getLogger(__name__)


@runtime_checkable
class ExchangeAdapter(Protocol):
    async def place_order(self, order: Order) -> dict[str, Any]: ...
    async def cancel_order(self, order_id: str) -> dict[str, Any]: ...
    async def get_order(self, order_id: str) -> dict[str, Any]: ...
    async def get_balance(self) -> dict[str, float]: ...


class LiveExchange:
    def __init__(self, exchange_id: str, api_key: str, secret: str, sandbox: bool = True) -> None:
        try:
            import ccxt.async_support as ccxt_async
        except ImportError:
            import ccxt as ccxt_async
        exchange_cls = getattr(ccxt_async, exchange_id)
        self._exchange = exchange_cls({
            "apiKey": api_key, "secret": secret,
            "sandbox": sandbox, "enableRateLimit": True,
        })
        self._markets_loaded = False

    async def _ensure_markets(self) -> None:
        if not self._markets_loaded:
            await self._exchange.load_markets()
            self._markets_loaded = True

    async def _retry(self, coro_fn, *args, attempts: int = 3):
        for i in range(attempts):
            try:
                return await coro_fn(*args)
            except Exception as e:
                if i == attempts - 1:
                    raise
                wait = 2 ** i
                logger.warning("Retry %d/%d after %s: %s", i + 1, attempts, wait, e)
                await asyncio.sleep(wait)

    async def place_order(self, order: Order) -> dict[str, Any]:
        await self._ensure_markets()
        # Balance pre-check
        bal = await self.get_balance()
        if order.side == "buy":
            quote = order.pair.split("/")[1]
            needed = order.amount * order.price
            if bal.get(quote, 0) < needed:
                raise ValueError(f"Insufficient {quote}: need {needed}, have {bal.get(quote, 0)}")
        else:
            base = order.pair.split("/")[0]
            if bal.get(base, 0) < order.amount:
                raise ValueError(f"Insufficient {base}: need {order.amount}, have {bal.get(base, 0)}")

        # Precision
        market = self._exchange.markets.get(order.pair, {})
        amount = self._exchange.amount_to_precision(order.pair, order.amount) if market else order.amount
        price = self._exchange.price_to_precision(order.pair, order.price) if market and order.order_type == "limit" else order.price

        # Min order size check
        min_amount = market.get("limits", {}).get("amount", {}).get("min", 0)
        if min_amount and float(amount) < min_amount:
            raise ValueError(f"Order amount {amount} below minimum {min_amount}")

        result = await self._retry(
            self._exchange.create_order,
            order.pair, order.order_type, order.side,
            float(amount), float(price) if order.order_type == "limit" else None,
        )

        # Order timeout: cancel after 30s if not filled
        if result.get("status") != "closed":
            order_id = result.get("id")
            if order_id:
                result = await self._wait_or_cancel(order_id, timeout=30)

        return result

    async def _wait_or_cancel(self, order_id: str, timeout: int = 30) -> dict:
        for _ in range(timeout // 2):
            await asyncio.sleep(2)
            info = await self._retry(self._exchange.fetch_order, order_id)
            if info.get("status") in ("closed", "canceled", "cancelled"):
                return info
        # Timeout — cancel
        logger.warning("Order %s timed out, cancelling", order_id)
        try:
            await self._retry(self._exchange.cancel_order, order_id)
        except Exception:
            pass
        return await self._retry(self._exchange.fetch_order, order_id)

    async def cancel_order(self, order_id: str) -> dict[str, Any]:
        return await self._retry(self._exchange.cancel_order, order_id)

    async def get_order(self, order_id: str) -> dict[str, Any]:
        return await self._retry(self._exchange.fetch_order, order_id)

    async def get_balance(self) -> dict[str, float]:
        bal = await self._retry(self._exchange.fetch_balance)
        return {k: float(v) for k, v in bal.get("total", {}).items() if float(v) > 0}

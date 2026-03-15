"""Exchange adapter protocol and hardened live implementation."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from cryptotrader.models import Order

logger = logging.getLogger(__name__)


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
    def __init__(self, exchange_id: str, api_key: str, secret: str, *, sandbox: bool, passphrase: str = "") -> None:
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
        }
        if passphrase:
            config["password"] = passphrase
        self._exchange = exchange_cls(config)
        self._markets_loaded = False

    async def _ensure_markets(self) -> None:
        if not self._markets_loaded:
            await self._exchange.load_markets()
            self._markets_loaded = True

    async def _retry(self, coro_fn, *args, attempts: int | None = None):
        import ccxt

        if attempts is None:
            from cryptotrader.config import load_config

            attempts = load_config().execution.retry_attempts

        _fatal = (ccxt.AuthenticationError, ccxt.PermissionDenied, ccxt.BadSymbol, ccxt.InsufficientFunds)
        for i in range(attempts):
            try:
                return await coro_fn(*args)
            except _fatal:
                raise  # Fatal errors — don't retry
            except Exception as e:
                if i == attempts - 1:
                    raise
                wait = 2**i
                logger.warning("Retry %d/%d after %ss: %s", i + 1, attempts, wait, e)
                await asyncio.sleep(wait)
        return None

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
        price = (
            self._exchange.price_to_precision(order.pair, order.price)
            if market and order.order_type == "limit"
            else order.price
        )

        # Min order size check
        min_amount = market.get("limits", {}).get("amount", {}).get("min", 0)
        if min_amount and float(amount) < min_amount:
            raise ValueError(f"Order amount {amount} below minimum {min_amount}")

        result = await self._retry(
            self._exchange.create_order,
            order.pair,
            order.order_type,
            order.side,
            float(amount),
            float(price) if order.order_type == "limit" else None,
        )

        # Order timeout: cancel if not filled
        if result.get("status") != "closed":
            order_id = result.get("id")
            if order_id:
                from cryptotrader.config import load_config

                wait_s = load_config().execution.order_wait_seconds
                result = await self._wait_or_cancel(order_id, order.pair, wait_seconds=wait_s)

        return result

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

    async def get_positions(self) -> dict[str, dict[str, Any]]:
        """Fetch open positions from exchange.

        Returns: {pair: {"amount": float, "side": str, "avg_price": float,
                         "unrealized_pnl": float, "liquidation_price": float | None}}
        """
        await self._ensure_markets()
        positions: dict[str, dict[str, Any]] = {}
        try:
            raw = await self._retry(self._exchange.fetch_positions)
            for p in raw:
                contracts = float(p.get("contracts", 0) or 0)
                if contracts == 0:
                    continue
                symbol = p.get("symbol", "")
                side = p.get("side", "long")
                amount = contracts if side == "long" else -contracts
                positions[symbol] = {
                    "amount": amount,
                    "side": side,
                    "avg_price": float(p.get("entryPrice", 0) or 0),
                    "unrealized_pnl": float(p.get("unrealizedPnl", 0) or 0),
                    "liquidation_price": float(p["liquidationPrice"]) if p.get("liquidationPrice") else None,
                }
        except Exception:
            # Spot exchanges don't support fetchPositions — derive from balance
            logger.debug("fetchPositions not available, deriving from balance", exc_info=True)
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

    async def close(self) -> None:
        try:
            await self._exchange.close()
        except Exception as e:
            logger.warning("Exchange close failed: %s", e)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()

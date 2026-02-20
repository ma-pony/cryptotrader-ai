"""Exchange adapter protocol and live implementation."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from cryptotrader.models import Order


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
            import ccxt as ccxt_async  # type: ignore[no-redef]
        exchange_cls = getattr(ccxt_async, exchange_id)
        self._exchange = exchange_cls({
            "apiKey": api_key,
            "secret": secret,
            "sandbox": sandbox,
        })

    async def place_order(self, order: Order) -> dict[str, Any]:
        return await self._exchange.create_order(
            symbol=order.pair,
            type=order.order_type,
            side=order.side,
            amount=order.amount,
            price=order.price if order.order_type == "limit" else None,
        )

    async def cancel_order(self, order_id: str) -> dict[str, Any]:
        return await self._exchange.cancel_order(order_id)

    async def get_order(self, order_id: str) -> dict[str, Any]:
        return await self._exchange.fetch_order(order_id)

    async def get_balance(self) -> dict[str, float]:
        bal = await self._exchange.fetch_balance()
        return {k: float(v) for k, v in bal.get("total", {}).items() if float(v) > 0}

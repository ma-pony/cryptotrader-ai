"""Exchange-backed portfolio input for the shared trading cycle."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cryptotrader.decision.models import CycleRequest

logger = logging.getLogger(__name__)

_DUST_AMOUNT_THRESHOLD = 1e-6


def _is_dust(amount: float) -> bool:
    return abs(float(amount or 0.0)) < _DUST_AMOUNT_THRESHOLD


async def _market_price(exchange: Any, pair: str) -> float:
    fetcher = getattr(exchange, "fetch_ticker", None)
    if fetcher is None:
        return 0.0
    try:
        ticker = await fetcher(pair)
    except Exception:
        logger.info("fetch_ticker failed for %s", pair, exc_info=True)
        return 0.0
    if not isinstance(ticker, dict):
        return 0.0
    raw = ticker.get("last") or ticker.get("close")
    try:
        return float(raw) if raw is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


class ExchangePortfolioReader:
    """Read balances and positions from the exchange used by execution."""

    def __init__(self, exchange: Any, database_url: str | None = None) -> None:
        self.exchange = exchange
        self.database_url = database_url

    async def read(
        self,
        request: CycleRequest,
        current_price: float,
        *,
        refresh_price: bool = False,
    ) -> dict[str, Any]:
        pair = request.pair.canonical()
        latest_price = await _market_price(self.exchange, pair) if refresh_price else 0.0
        execution_price = latest_price if latest_price > 0.0 else current_price
        balances = await self.exchange.get_balance()
        try:
            free_balances = await self.exchange.get_free_balance()
        except AttributeError:
            free_balances = balances

        try:
            raw_positions = await self.exchange.get_positions(
                current_prices={pair: execution_price},
            )
        except TypeError:
            raw_positions = await self.exchange.get_positions()
        positions = {pair: dict(position) for pair, position in raw_positions.items()}

        for asset, raw_amount in balances.items():
            amount = float(raw_amount or 0.0)
            if asset == request.pair.quote or _is_dust(amount):
                continue
            spot_pair = f"{asset}/{request.pair.quote}"
            if spot_pair in positions:
                continue
            mark = (
                execution_price
                if refresh_price and spot_pair == pair
                else await _market_price(self.exchange, spot_pair)
            )
            positions[spot_pair] = {
                "amount": amount,
                "side": "long" if amount > 0 else "short",
                "avg_price": mark,
                "current_price": mark,
                "unrealized_pnl": 0.0,
                "liquidation_price": None,
            }

        cash = float(balances.get(request.pair.quote, 0.0) or 0.0)
        free_cash = float(free_balances.get(request.pair.quote, cash) or 0.0)
        total_value = cash
        for pair, position in positions.items():
            amount = float(position.get("amount", 0.0) or 0.0)
            if ":" in pair:
                total_value += float(position.get("unrealized_pnl", 0.0) or 0.0)
                continue
            mark = float(position.get("current_price", 0.0) or position.get("avg_price", 0.0) or 0.0)
            if pair == request.pair.canonical() and execution_price > 0 and not position.get("current_price"):
                mark = execution_price
            total_value += amount * mark

        portfolio: dict[str, Any] = {
            "cash": cash,
            "free_cash": free_cash,
            "positions": positions,
            "total_value": total_value,
            "current_price": execution_price,
        }
        if self.database_url:
            from cryptotrader.portfolio.manager import PortfolioManager

            manager = PortfolioManager(self.database_url)
            portfolio["daily_pnl"] = await manager.get_daily_pnl()
            portfolio["drawdown"] = await manager.get_drawdown()
        return portfolio

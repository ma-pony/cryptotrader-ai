"""Materialize immutable, point-in-time contexts shared by signal components."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Protocol

from cryptotrader.agents._indicators import atr
from cryptotrader.data.market import clip_ohlcv_at
from cryptotrader.signals.models import PositionSnapshot, SignalContext

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from datetime import datetime

    from cryptotrader.decision.models import CycleRequest
    from cryptotrader.models import DataSnapshot, MarketData
    from cryptotrader.signals.models import DataRequirements


class SignalContextProvider(Protocol):
    async def collect(self, request: CycleRequest, requirements: DataRequirements) -> SignalContext: ...

    async def refresh_execution_state(self, context: SignalContext) -> SignalContext: ...


class PortfolioReader(Protocol):
    async def read(self, request: CycleRequest, current_price: float) -> dict: ...


def _current_price(market: MarketData) -> float:
    ticker_price = float(market.ticker.get("last", 0.0) or 0.0)
    if ticker_price > 0.0:
        return ticker_price
    if market.ohlcv.empty:
        raise ValueError("market snapshot has no current price")
    return float(market.ohlcv["close"].iloc[-1])


def _atr_value(snapshot: DataSnapshot) -> float:
    frame = snapshot.market.ohlcv
    values = atr(frame["high"], frame["low"], frame["close"], length=14).dropna()
    return float(values.iloc[-1]) if not values.empty else 0.0


def _position_snapshot(
    portfolio: dict,
    pair: str,
    current_price: float,
    equity: float,
    max_single_pct: float,
) -> PositionSnapshot:
    raw = (portfolio.get("positions") or {}).get(pair)
    if raw is None:
        return PositionSnapshot("flat", 0.0, 0.0)
    if isinstance(raw, int | float):
        signed_amount = float(raw)
        avg_price = None
        unrealized_pnl = 0.0
        declared_side = None
    else:
        amount = float(raw.get("amount", 0.0) or 0.0)
        declared_side = raw.get("side")
        signed_amount = -abs(amount) if declared_side == "short" else amount
        avg_price = raw.get("avg_price")
        unrealized_pnl = float(raw.get("unrealized_pnl", 0.0) or 0.0)
    if signed_amount == 0.0:
        return PositionSnapshot("flat", 0.0, 0.0, avg_price, unrealized_pnl)
    side = "short" if signed_amount < 0.0 or declared_side == "short" else "long"
    amount = abs(signed_amount)
    maximum_notional = equity * max_single_pct
    size_ratio = amount * current_price / maximum_notional if maximum_notional > 0.0 else 0.0
    return PositionSnapshot(side, amount, size_ratio, avg_price, unrealized_pnl)


def _materialize_snapshot(base: DataSnapshot, market: MarketData, as_of, limit: int) -> DataSnapshot:
    clipped_market = replace(market, ohlcv=clip_ohlcv_at(market.ohlcv, as_of, limit))
    return replace(base, timestamp=as_of, market=clipped_market)


class LiveSignalContextProvider:
    def __init__(
        self,
        aggregator,
        market,
        portfolio: PortfolioReader,
        *,
        default_timeframe: str,
        max_single_pct: float = 1.0,
        kronos_aux_symbol: str = "BTCUSDT",
    ) -> None:
        self.aggregator = aggregator
        self.market = market
        self.portfolio = portfolio
        self.default_timeframe = default_timeframe
        self.max_single_pct = max_single_pct
        self.kronos_aux_symbol = kronos_aux_symbol

    async def collect(self, request: CycleRequest, requirements: DataRequirements) -> SignalContext:
        if not requirements.candles:
            raise ValueError("signal context requires at least one candle timeframe")
        primary = requirements.candles[0]
        base = await self.aggregator.collect(
            pair=request.pair.canonical(),
            exchange_id=request.exchange_id,
            timeframe=primary.timeframe,
            limit=primary.limit,
            backtest_mode=False,
            kronos_aux=requirements.kronos_aux,
            kronos_aux_symbol=self.kronos_aux_symbol,
        )
        as_of = base.timestamp
        snapshots = {
            primary.timeframe: _materialize_snapshot(base, base.market, as_of, primary.limit),
        }
        for requirement in requirements.candles[1:]:
            market = await self.market.collect(
                request.pair.canonical(),
                request.exchange_id,
                requirement.timeframe,
                requirement.limit,
            )
            snapshots[requirement.timeframe] = _materialize_snapshot(base, market, as_of, requirement.limit)

        if self.default_timeframe not in snapshots:
            raise ValueError(f"default timeframe {self.default_timeframe!r} is missing from requirements")
        price = _current_price(snapshots[primary.timeframe].market)
        portfolio = dict(await self.portfolio.read(request, price))
        primary_market = snapshots[primary.timeframe].market
        portfolio["recent_prices"] = [float(value) for value in primary_market.ohlcv["close"].dropna().tolist()]
        portfolio["funding_rate"] = float(primary_market.funding_rate or 0.0)
        portfolio["symbol"] = request.pair.base
        equity = float(portfolio.get("total_value", 0.0) or 0.0)
        current_position = _position_snapshot(
            portfolio,
            request.pair.canonical(),
            price,
            equity,
            self.max_single_pct,
        )
        return SignalContext(
            pair=request.pair,
            as_of=as_of,
            mode=request.mode,
            exchange_id=request.exchange_id,
            market_type=request.pair.market_type,
            equity=equity,
            current_price=price,
            atr=_atr_value(snapshots[self.default_timeframe]),
            current_position=current_position,
            snapshots=snapshots,
            portfolio=portfolio,
        )

    async def refresh_execution_state(self, context: SignalContext) -> SignalContext:
        from cryptotrader.decision.models import CycleRequest

        request = CycleRequest(context.pair, context.mode, context.exchange_id, context.as_of)
        portfolio = dict(await self.portfolio.read(request, context.current_price))
        portfolio.setdefault("recent_prices", context.portfolio.get("recent_prices", []))
        portfolio.setdefault("funding_rate", context.portfolio.get("funding_rate", 0.0))
        portfolio.setdefault("symbol", context.pair.base)
        price = float(portfolio.get("current_price", context.current_price) or context.current_price)
        equity = float(portfolio.get("total_value", 0.0) or 0.0)
        current_position = _position_snapshot(
            portfolio,
            context.pair.canonical(),
            price,
            equity,
            self.max_single_pct,
        )
        return replace(
            context,
            current_price=price,
            equity=equity,
            current_position=current_position,
            portfolio=portfolio,
        )


class HistoricalSignalContextProvider:
    def __init__(
        self,
        history: Mapping[str, DataSnapshot] | Callable[[str, datetime], DataSnapshot],
        *,
        default_timeframe: str,
        equity: float = 10_000.0,
        current_position: PositionSnapshot | None = None,
        max_single_pct: float = 1.0,
    ) -> None:
        self.history = history
        self.default_timeframe = default_timeframe
        self.equity = equity
        self.cash = equity
        self.current_position = current_position or PositionSnapshot("flat", 0.0, 0.0)
        self.max_single_pct = max_single_pct
        self.daily_pnl = 0.0
        self.drawdown = 0.0

    def set_execution_state(
        self,
        *,
        equity: float,
        cash: float,
        current_position: PositionSnapshot,
        daily_pnl: float = 0.0,
        drawdown: float = 0.0,
    ) -> None:
        self.equity = equity
        self.cash = cash
        self.current_position = current_position
        self.daily_pnl = daily_pnl
        self.drawdown = drawdown

    def _snapshot(self, timeframe: str, as_of: datetime) -> DataSnapshot:
        if callable(self.history):
            return self.history(timeframe, as_of)
        return self.history[timeframe]

    async def collect(self, request: CycleRequest, requirements: DataRequirements) -> SignalContext:
        if request.as_of is None:
            raise ValueError("historical context requires request.as_of")
        snapshots = {
            requirement.timeframe: _materialize_snapshot(
                snapshot,
                snapshot.market,
                request.as_of,
                requirement.limit,
            )
            for requirement in requirements.candles
            for snapshot in (self._snapshot(requirement.timeframe, request.as_of),)
        }
        if self.default_timeframe not in snapshots:
            raise ValueError(f"default timeframe {self.default_timeframe!r} is missing from requirements")
        market = snapshots[self.default_timeframe].market
        price = float(market.ohlcv["close"].iloc[-1])
        position = replace(
            self.current_position,
            size_ratio=(
                self.current_position.amount * price / (self.equity * self.max_single_pct)
                if self.current_position.side != "flat" and self.equity * self.max_single_pct > 0.0
                else 0.0
            ),
        )
        recent_prices = [float(value) for value in market.ohlcv["close"].dropna().tolist()]
        positions = (
            {
                request.pair.canonical(): {
                    "amount": position.signed_amount,
                    "side": position.side,
                    "avg_price": position.avg_price or price,
                    "unrealized_pnl": position.unrealized_pnl,
                }
            }
            if position.side != "flat"
            else {}
        )
        return SignalContext(
            pair=request.pair,
            as_of=request.as_of,
            mode=request.mode,
            exchange_id=request.exchange_id,
            market_type=request.pair.market_type,
            equity=self.equity,
            current_price=price,
            atr=_atr_value(snapshots[self.default_timeframe]),
            current_position=position,
            snapshots=snapshots,
            portfolio={
                "total_value": self.equity,
                "cash": self.cash,
                "free_cash": self.cash,
                "positions": positions,
                "daily_pnl": self.daily_pnl,
                "drawdown": self.drawdown,
                "recent_prices": recent_prices,
                "funding_rate": float(market.funding_rate or 0.0),
                "api_latency_ms": 0.0,
                "pair": request.pair.canonical(),
                "symbol": request.pair.base,
            },
        )

    async def refresh_execution_state(self, context: SignalContext) -> SignalContext:
        return context

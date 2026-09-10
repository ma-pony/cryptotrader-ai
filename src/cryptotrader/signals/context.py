"""Market-only context materializers retained for explicit source implementations."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Protocol

from cryptotrader.agents._indicators import atr
from cryptotrader.data.market import clip_ohlcv_at
from cryptotrader.signals.models import DataRequirements, SignalContext

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from datetime import datetime

    from cryptotrader.models import DataSnapshot, MarketData
    from cryptotrader.pair import Pair


class SignalContextProvider(Protocol):
    id: str

    async def collect(
        self,
        pair: Pair,
        as_of: datetime,
        requirements: DataRequirements,
    ) -> SignalContext: ...


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


def _materialize_snapshot(base: DataSnapshot, market: MarketData, as_of, limit: int) -> DataSnapshot:
    clipped_market = replace(market, ohlcv=clip_ohlcv_at(market.ohlcv, as_of, limit))
    return replace(base, timestamp=as_of, market=clipped_market)


class LiveSignalContextProvider:
    id = "default"

    def __init__(
        self,
        aggregator,
        market,
        *,
        market_adapter_id: str,
        default_timeframe: str,
        kronos_aux_symbol: str = "BTCUSDT",
    ) -> None:
        self.aggregator = aggregator
        self.market = market
        self.market_adapter_id = market_adapter_id
        self.default_timeframe = default_timeframe
        self.kronos_aux_symbol = kronos_aux_symbol

    async def collect(self, pair, as_of, requirements) -> SignalContext:
        if not requirements.candles:
            raise ValueError("signal context requires at least one candle timeframe")
        primary = requirements.candles[0]
        base = await self.aggregator.collect(
            pair=pair.canonical(),
            market_adapter_id=self.market_adapter_id,
            timeframe=primary.timeframe,
            limit=primary.limit,
            backtest_mode=False,
            kronos_aux=requirements.kronos_aux,
            kronos_aux_symbol=self.kronos_aux_symbol,
        )
        snapshots = {primary.timeframe: _materialize_snapshot(base, base.market, as_of, primary.limit)}
        for requirement in requirements.candles[1:]:
            market = await self.market.collect(
                pair.canonical(),
                self.market_adapter_id,
                requirement.timeframe,
                requirement.limit,
            )
            snapshots[requirement.timeframe] = _materialize_snapshot(base, market, as_of, requirement.limit)
        if self.default_timeframe not in snapshots:
            raise ValueError(f"default timeframe {self.default_timeframe!r} is missing from requirements")
        primary_snapshot = snapshots[primary.timeframe]
        return SignalContext(
            pair,
            as_of,
            self.id,
            pair.market_type,
            _current_price(primary_snapshot.market),
            _atr_value(snapshots[self.default_timeframe]),
            snapshots,
        )


class HistoricalSignalContextProvider:
    id = "historical"

    def __init__(
        self,
        history: Mapping[str, DataSnapshot] | Callable[[str, datetime], DataSnapshot],
        *,
        default_timeframe: str,
        atr_timeframe: str,
        source_id: str = "historical",
    ) -> None:
        self.history = history
        self.default_timeframe = default_timeframe
        self.atr_timeframe = atr_timeframe
        self.id = source_id

    def _snapshot(self, timeframe: str, as_of: datetime) -> DataSnapshot:
        if callable(self.history):
            return self.history(timeframe, as_of)
        return self.history[timeframe]

    def requirements(self) -> DataRequirements:
        return DataRequirements()

    async def read_candles(self, pair, timeframe, start, end, as_of):
        """Read only the supplied historical evidence, never a live fallback."""
        from decimal import Decimal

        import pandas as pd

        from cryptotrader.market_sources.protocol import HistoricalCandle
        from cryptotrader.signals.presentation import interval_delta

        if any(value.utcoffset() is None for value in (start, end, as_of)) or start >= end:
            raise ValueError("historical window requires increasing aware timestamps")
        try:
            snapshot = self._snapshot(timeframe, as_of)
        except KeyError:
            return ()
        if snapshot.market.pair != pair.canonical():
            return ()
        frame = snapshot.market.ohlcv
        fields = ("open", "high", "low", "close", "volume")
        if not all(name in frame for name in fields) or frame.empty:
            return ()
        if "timestamp" in frame:
            raw = frame["timestamp"]
            opened = pd.to_datetime(raw, unit="ms" if pd.api.types.is_numeric_dtype(raw) else None, utc=True)
        elif isinstance(frame.index, pd.DatetimeIndex):
            opened = pd.to_datetime(frame.index, utc=True)
        else:
            return ()
        delta = interval_delta(timeframe)
        return tuple(
            HistoricalCandle(
                open_time=time.to_pydatetime(),
                **{name: Decimal(str(value)) for name, value in zip(fields, values, strict=True)},
            )
            for time, values in zip(opened, frame[list(fields)].itertuples(index=False, name=None), strict=True)
            if start <= time < end and time + delta <= as_of
        )

    async def collect(self, pair, as_of, requirements) -> SignalContext:
        snapshots = {
            requirement.timeframe: _materialize_snapshot(snapshot, snapshot.market, as_of, requirement.limit)
            for requirement in requirements.candles
            for snapshot in (self._snapshot(requirement.timeframe, as_of),)
        }
        if self.default_timeframe not in snapshots:
            raise ValueError(f"default timeframe {self.default_timeframe!r} is missing from requirements")
        market = snapshots[self.default_timeframe].market
        return SignalContext(
            pair,
            as_of,
            self.id,
            pair.market_type,
            _current_price(market),
            _atr_value(snapshots[self.atr_timeframe]),
            snapshots,
        )

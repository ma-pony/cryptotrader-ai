"""Protocol implemented by installed market evidence sources."""

from __future__ import annotations

from decimal import Decimal  # noqa: TC003 - Pydantic resolves this field at runtime.
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator


class HistoricalCandle(BaseModel):
    """Public OHLCV evidence. Timestamp identifies the bar's OPEN, never its close."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    open_time: AwareDatetime
    open: Decimal = Field(gt=0)
    high: Decimal = Field(gt=0)
    low: Decimal = Field(gt=0)
    close: Decimal = Field(gt=0)
    volume: Decimal = Field(ge=0)

    @model_validator(mode="after")
    def valid_range(self):
        if not self.low <= min(self.open, self.close) <= max(self.open, self.close) <= self.high:
            raise ValueError("OHLC prices must lie inside the actual bar range")
        return self


if TYPE_CHECKING:
    from datetime import datetime

    from cryptotrader.pair import Pair
    from cryptotrader.signals.models import DataRequirements, SignalContext


@runtime_checkable
class MarketDataSource(Protocol):
    id: str

    def requirements(self) -> DataRequirements: ...

    async def collect(
        self,
        pair: Pair,
        as_of: datetime,
        requirements: DataRequirements,
    ) -> SignalContext: ...

    async def read_candles(
        self,
        pair: Pair,
        timeframe: str,
        start: datetime,
        end: datetime,
        as_of: datetime,
    ) -> tuple[HistoricalCandle, ...]:
        """Read exact public-source bars with opens in [start, end), closed by as_of.

        No account, ticker, news, inference, alternate source or nearest-price fallback.
        Missing/unsupported bars return an empty tuple. Transport errors may raise.
        """
        ...

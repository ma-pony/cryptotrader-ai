"""Protocol implemented by installed market evidence sources."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

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

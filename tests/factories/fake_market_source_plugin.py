"""Installed market-source entry point fixture without production I/O."""

from __future__ import annotations

from cryptotrader.signals.models import DataRequirements, SignalContext


class FixtureMarketSource:
    id = "fixture-market"

    def __init__(self, config) -> None:
        self.config = config

    def requirements(self) -> DataRequirements:
        return DataRequirements()

    async def collect(self, pair, as_of, requirements) -> SignalContext:
        del requirements
        return SignalContext(
            pair=pair,
            as_of=as_of,
            market_data_source_id=self.id,
            market_type=pair.market_type,
            current_price=100.0,
            atr=5.0,
            snapshots={},
        )


def create_source(config) -> FixtureMarketSource:
    return FixtureMarketSource(config)


def create_conflicting_default(config) -> FixtureMarketSource:
    source = FixtureMarketSource(config)
    source.id = "default"
    return source

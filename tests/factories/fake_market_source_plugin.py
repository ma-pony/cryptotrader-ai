"""Installed market-source entry point fixture without production I/O."""

from __future__ import annotations

from cryptotrader.configuration.catalog import PluginConfiguration, configured_factory
from cryptotrader.configuration.fields import LocalizedText
from cryptotrader.configuration.parameters import DefaultMarketSourceParameters
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


create_source = configured_factory(
    PluginConfiguration(
        id="fixture-market",
        label=LocalizedText(zh_CN="测试市场", en_US="Fixture market"),
        description=LocalizedText(zh_CN="测试数据源。", en_US="Test source."),
        parameter_model=DefaultMarketSourceParameters,
    )
)(create_source)
create_conflicting_default = configured_factory(
    PluginConfiguration(
        id="default",
        label=LocalizedText(zh_CN="冲突市场", en_US="Conflicting market"),
        description=LocalizedText(zh_CN="测试数据源。", en_US="Test source."),
        parameter_model=DefaultMarketSourceParameters,
    )
)(create_conflicting_default)

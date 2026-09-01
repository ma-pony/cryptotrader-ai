"""Offline public-source/component fixtures for full replay execution."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from cryptotrader.market_sources.protocol import HistoricalCandle
from cryptotrader.market_sources.registry import MarketSourceRegistry
from cryptotrader.runtime_config.models import (
    MarketDataConfig,
    RuntimeConfigSnapshot,
    SignalComponentConfig,
    SignalConfig,
)
from cryptotrader.signals.models import CandleRequirement, ComponentSignal, DataRequirements
from cryptotrader.signals.registry import SignalComponentRegistry
from tests.factories.runtime_config import runtime_document

START = datetime(2024, 1, 1, tzinfo=UTC)


class HistoricalSource:
    id = "default"

    def __init__(self):
        self.requests = []

    def requirements(self):
        return DataRequirements()

    async def collect(self, *args):
        raise AssertionError("replay must not collect current market or news")

    async def read_candles(self, pair, timeframe, start, end, as_of):
        self.requests.append((pair, timeframe, start, end, as_of))
        assert timeframe == "1h"
        rows = []
        opened = start
        while opened < end and opened + timedelta(hours=1) <= as_of:
            price = Decimal("100") + Decimal(int((opened - START).total_seconds() // 3600))
            rows.append(
                HistoricalCandle(
                    open_time=opened, open=price, high=price + 3, low=price - 3, close=price, volume=Decimal("10")
                )
            )
            opened += timedelta(hours=1)
        return tuple(rows)


class Component:
    id = "fixture"
    display_name = "Fixture"
    description = "deterministic historical component"

    def __init__(self):
        self.contexts = []

    def requirements(self):
        return DataRequirements(candles=(CandleRequirement("1h", 20),))

    async def evaluate(self, context):
        self.contexts.append(context)
        return ComponentSignal(self.id, "long", 0.1, context.pair.canonical())


def replay_config():
    document = runtime_document(
        market_data=MarketDataConfig(
            source_id="default", timeframe="1h", parameters={"market_adapter_id": "bybit", "limit": 20}
        ),
        signals=SignalConfig(
            components=(SignalComponentConfig(component_id="fixture", enabled=True, weight=1.0),),
            neutral_threshold=0.01,
            max_target_ratio=1.0,
            atr_stop_multiplier=2.0,
            reward_ratio=2.0,
        ),
    )
    return RuntimeConfigSnapshot(7, document, START)


def registries():
    source, component = HistoricalSource(), Component()
    return MarketSourceRegistry((source,)), SignalComponentRegistry((component,)), source, component

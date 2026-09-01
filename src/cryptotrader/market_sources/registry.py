"""Resolve code-registered market sources without database-provided code paths."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cryptotrader.configuration import registry as extension_registry
from cryptotrader.market_sources.protocol import MarketDataSource

if TYPE_CHECKING:
    from collections.abc import Iterable

    from cryptotrader.runtime_config.models import MarketDataConfig


class MarketSourceRegistry:
    def __init__(self, sources: Iterable[MarketDataSource] = ()) -> None:
        self._sources: dict[str, MarketDataSource] = {}
        self._registered_ids: frozenset[str] = frozenset()
        for source in sources:
            self.register(source)

    @classmethod
    def discover(
        cls,
        config: MarketDataConfig,
        *,
        news_provider_key: str = "",
    ) -> MarketSourceRegistry:
        registrations = extension_registry.get_extension_registry().market_sources
        factories = {key: item.factory for key, item in registrations.items()}

        if config.source_id not in factories:
            raise ValueError(f"unregistered market source: {config.source_id}")
        source = factories[config.source_id](config, news_provider_key=news_provider_key)
        if not isinstance(source, MarketDataSource):
            raise TypeError(f"factory for {config.source_id} did not return MarketDataSource")
        if source.id != config.source_id:
            raise ValueError(f"market source factory id mismatch: registration {config.source_id}, source {source.id}")

        registry = cls((source,))
        registry._registered_ids = frozenset(factories)
        return registry

    def register(self, source: MarketDataSource) -> None:
        if not isinstance(source, MarketDataSource):
            raise TypeError("source does not implement MarketDataSource")
        if source.id in self._sources:
            raise ValueError(f"duplicate market source id: {source.id}")
        self._sources[source.id] = source
        self._registered_ids = self._registered_ids | {source.id}

    def require(self, source_id: str) -> MarketDataSource:
        try:
            return self._sources[source_id]
        except KeyError as error:
            raise KeyError(f"market source is not configured: {source_id}") from error

    def ids(self) -> tuple[str, ...]:
        return tuple(self._sources)

    def registered_ids(self) -> frozenset[str]:
        return self._registered_ids

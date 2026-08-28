"""Discover installed market sources without database-provided code paths."""

from __future__ import annotations

from importlib import metadata
from typing import TYPE_CHECKING, Any

from cryptotrader.market_sources.protocol import MarketDataSource

if TYPE_CHECKING:
    from collections.abc import Iterable

    from cryptotrader.runtime_config.models import MarketDataConfig

_ENTRY_POINT_GROUP = "cryptotrader.market_sources"


class MarketSourceRegistry:
    def __init__(self, sources: Iterable[MarketDataSource] = ()) -> None:
        self._sources: dict[str, MarketDataSource] = {}
        self._installed_ids: frozenset[str] = frozenset()
        for source in sources:
            self.register(source)

    @classmethod
    def discover(
        cls,
        config: MarketDataConfig,
        *,
        entry_points=None,
    ) -> MarketSourceRegistry:
        from cryptotrader.market_sources.default import create_source as create_default

        factories: dict[str, Any] = {"default": create_default}
        discovered = metadata.entry_points(group=_ENTRY_POINT_GROUP) if entry_points is None else entry_points
        for entry_point in discovered:
            factory = entry_point.load()
            installed = factories.get(entry_point.name)
            if installed is not None:
                if entry_point.name == "default" and installed is factory:
                    continue
                raise ValueError(f"duplicate market source id: {entry_point.name}")
            factories[entry_point.name] = factory

        if config.source_id not in factories:
            raise ValueError(f"uninstalled market source: {config.source_id}")
        source = factories[config.source_id](config)
        if not isinstance(source, MarketDataSource):
            raise TypeError(f"factory for {config.source_id} did not return MarketDataSource")
        if source.id != config.source_id:
            raise ValueError(f"market source factory id mismatch: entry point {config.source_id}, source {source.id}")

        registry = cls((source,))
        registry._installed_ids = frozenset(factories)
        return registry

    def register(self, source: MarketDataSource) -> None:
        if not isinstance(source, MarketDataSource):
            raise TypeError("source does not implement MarketDataSource")
        if source.id in self._sources:
            raise ValueError(f"duplicate market source id: {source.id}")
        self._sources[source.id] = source
        self._installed_ids = self._installed_ids | {source.id}

    def require(self, source_id: str) -> MarketDataSource:
        try:
            return self._sources[source_id]
        except KeyError as error:
            raise KeyError(f"market source is not configured: {source_id}") from error

    def ids(self) -> tuple[str, ...]:
        return tuple(self._sources)

    def installed_ids(self) -> frozenset[str]:
        return self._installed_ids

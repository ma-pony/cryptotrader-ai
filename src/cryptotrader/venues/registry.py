"""Discover code-owned venue adapters from Python entry points."""

from __future__ import annotations

from importlib import metadata
from inspect import iscoroutinefunction
from typing import TYPE_CHECKING, Any

from cryptotrader.venues.protocol import VenueAdapter

if TYPE_CHECKING:
    from collections.abc import Iterable

_ENTRY_POINT_GROUP = "cryptotrader.venue_adapters"


class VenueAdapterRegistry:
    """Immutable-at-runtime lookup of installed venue adapter instances."""

    def __init__(self, adapters: Iterable[VenueAdapter] = ()) -> None:
        self._adapters: dict[str, VenueAdapter] = {}
        for adapter in adapters:
            self._register(adapter)
        self._installed_ids = frozenset(self._adapters)

    @classmethod
    def discover(
        cls,
        configured_adapter_ids: Iterable[str] = (),
        *,
        entry_points: Iterable[Any] | None = None,
    ) -> VenueAdapterRegistry:
        """Load installed factories and fail if configured adapters cannot resolve."""
        configured_ids = frozenset(configured_adapter_ids)
        if not all(type(adapter_id) is str and bool(adapter_id.strip()) for adapter_id in configured_ids):
            raise ValueError("configured adapter ids must be non-empty strings")

        discovered = metadata.entry_points(group=_ENTRY_POINT_GROUP) if entry_points is None else entry_points
        factories: dict[str, Any] = {}
        for entry_point in discovered:
            if entry_point.name in factories:
                raise ValueError(f"duplicate venue adapter id: {entry_point.name}")
            factories[entry_point.name] = entry_point.load()

        missing = sorted(configured_ids - set(factories))
        if missing:
            raise ValueError(f"uninstalled venue adapter ids: {', '.join(missing)}")

        adapters = []
        for adapter_id, factory in factories.items():
            if not callable(factory):
                raise TypeError(f"factory for {adapter_id} is not callable")
            adapter = factory()
            if not isinstance(adapter, VenueAdapter):
                raise TypeError(f"factory for {adapter_id} did not return VenueAdapter")
            if adapter.adapter_id != adapter_id:
                raise ValueError(
                    f"venue adapter factory id mismatch: entry point {adapter_id}, adapter {adapter.adapter_id}"
                )
            adapters.append(adapter)

        registry = cls(adapters)
        registry._installed_ids = frozenset(factories)
        return registry

    def _register(self, adapter: VenueAdapter) -> None:
        if not isinstance(adapter, VenueAdapter):
            raise TypeError("adapter does not implement VenueAdapter")
        if not iscoroutinefunction(adapter.connect):
            raise TypeError(f"venue adapter {adapter.adapter_id} connect must be async")
        if type(adapter.adapter_id) is not str or not adapter.adapter_id.strip():
            raise ValueError("venue adapter id must be a non-empty string")
        if adapter.adapter_id in self._adapters:
            raise ValueError(f"duplicate venue adapter id: {adapter.adapter_id}")
        self._adapters[adapter.adapter_id] = adapter

    def require(self, adapter_id: str) -> VenueAdapter:
        try:
            return self._adapters[adapter_id]
        except KeyError as error:
            raise KeyError(f"venue adapter is not configured: {adapter_id}") from error

    def ids(self) -> tuple[str, ...]:
        return tuple(self._adapters)

    def installed_ids(self) -> frozenset[str]:
        return self._installed_ids

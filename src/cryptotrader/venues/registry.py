"""Resolve code-owned venue adapters from the application registry."""

from __future__ import annotations

from inspect import iscoroutinefunction
from typing import TYPE_CHECKING

from cryptotrader.configuration import registry as extension_registry
from cryptotrader.venues.protocol import VenueAdapter

if TYPE_CHECKING:
    from collections.abc import Iterable


class VenueAdapterRegistry:
    """Immutable-at-runtime lookup of code-registered venue adapters."""

    def __init__(self, adapters: Iterable[VenueAdapter] = ()) -> None:
        self._adapters: dict[str, VenueAdapter] = {}
        for adapter in adapters:
            self._register(adapter)
        self._registered_ids = frozenset(self._adapters)
        self._factories = {}
        self._account_store = None

    def bind_account_store(self, store) -> None:
        """Bind local Paper state without instantiating any adapter."""
        self._account_store = store

    @classmethod
    def discover(
        cls,
        configured_adapter_ids: Iterable[str] = (),
    ) -> VenueAdapterRegistry:
        """Load code-registered factories and fail if configured adapters cannot resolve."""
        configured_ids = frozenset(configured_adapter_ids)
        if not all(type(adapter_id) is str and bool(adapter_id.strip()) for adapter_id in configured_ids):
            raise ValueError("configured adapter ids must be non-empty strings")

        registrations = extension_registry.get_extension_registry().venues
        factories = {key: item.factory for key, item in registrations.items()}

        missing = sorted(configured_ids - set(factories))
        if missing:
            raise ValueError(f"unregistered venue adapter ids: {', '.join(missing)}")

        for adapter_id, factory in factories.items():
            if not callable(factory):
                raise TypeError(f"factory for {adapter_id} is not callable")
        registry = cls()
        registry._factories = factories
        registry._registered_ids = frozenset(factories)
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
        if adapter_id not in self._adapters and adapter_id in self._factories:
            adapter = self._factories[adapter_id]()
            if not isinstance(adapter, VenueAdapter):
                raise TypeError(f"factory for {adapter_id} did not return VenueAdapter")
            if adapter.adapter_id != adapter_id:
                raise ValueError(
                    f"venue adapter factory id mismatch: registration {adapter_id}, adapter {adapter.adapter_id}"
                )
            self._register(adapter)
        try:
            adapter = self._adapters[adapter_id]
            from cryptotrader.venues.paper import PaperVenueAdapter

            if isinstance(adapter, PaperVenueAdapter) and self._account_store is not None:
                adapter.account_store = self._account_store
            return adapter
        except KeyError as error:
            raise KeyError(f"venue adapter is not configured: {adapter_id}") from error

    def ids(self) -> tuple[str, ...]:
        return tuple(self._adapters)

    def registered_ids(self) -> frozenset[str]:
        return self._registered_ids

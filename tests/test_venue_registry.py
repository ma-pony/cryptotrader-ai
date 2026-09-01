"""Discovery and lookup for installed venue adapters."""

from __future__ import annotations

import pytest


class FakeAdapter:
    def __init__(self, adapter_id: str) -> None:
        self.adapter_id = adapter_id

    def capabilities(self, environment):
        from cryptotrader.venues.models import VenueCapabilities

        return VenueCapabilities(frozenset({"swap"}), True, False, True, frozenset({"market"}))

    async def connect(self, connection, credentials):
        raise NotImplementedError


class SyncConnectAdapter(FakeAdapter):
    def connect(self, connection, credentials):
        raise NotImplementedError


def test_registry_resolves_by_adapter_id_without_brand_conditionals():
    from cryptotrader.venues.registry import VenueAdapterRegistry

    registry = VenueAdapterRegistry((FakeAdapter("alpha"), FakeAdapter("beta")))

    assert registry.require("beta").adapter_id == "beta"
    with pytest.raises(KeyError, match="missing"):
        registry.require("missing")


def test_registry_rejects_duplicate_adapter_ids():
    from cryptotrader.venues.registry import VenueAdapterRegistry

    with pytest.raises(ValueError, match="duplicate venue adapter id: alpha"):
        VenueAdapterRegistry((FakeAdapter("alpha"), FakeAdapter("alpha")))


def test_registry_rejects_adapter_with_synchronous_connect_before_runtime_use():
    from cryptotrader.venues.registry import VenueAdapterRegistry

    with pytest.raises(TypeError, match="connect must be async"):
        VenueAdapterRegistry((SyncConnectAdapter("sync"),))


def test_registry_discovers_code_registered_factories(monkeypatch):
    from cryptotrader.configuration import registry
    from cryptotrader.venues.registry import VenueAdapterRegistry
    from tests.factories.workbench_extensions import sample_registry

    extensions, calls = sample_registry()
    monkeypatch.setattr(registry, "get_extension_registry", lambda: extensions)
    discovered = VenueAdapterRegistry.discover({"sample_venue"})
    assert discovered.require("sample_venue").adapter_id == "sample_venue"
    assert "sample_venue" in discovered.registered_ids()
    assert calls == ["venue"]


def test_registry_rejects_factory_id_mismatch(monkeypatch):
    from dataclasses import replace

    from cryptotrader.configuration import registry
    from cryptotrader.venues.registry import VenueAdapterRegistry
    from tests.factories.workbench_extensions import sample_registry

    extensions, _ = sample_registry()
    extensions.venues["sample_venue"] = replace(extensions.venues["sample_venue"], factory=lambda: FakeAdapter("other"))
    monkeypatch.setattr(registry, "get_extension_registry", lambda: extensions)
    with pytest.raises(ValueError, match="registration sample_venue, adapter other"):
        VenueAdapterRegistry.discover({"sample_venue"}).require("sample_venue")


def test_registry_rejects_configured_unregistered_adapter():
    from cryptotrader.venues.registry import VenueAdapterRegistry

    with pytest.raises(ValueError, match="unregistered venue adapter ids: missing"):
        VenueAdapterRegistry.discover({"missing"})

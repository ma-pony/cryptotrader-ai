"""Discovery and lookup for installed venue adapters."""

from __future__ import annotations

from importlib import metadata

import pytest


class FakeAdapter:
    def __init__(self, adapter_id: str) -> None:
        self.adapter_id = adapter_id

    def capabilities(self, environment):
        from cryptotrader.venues.models import VenueCapabilities

        return VenueCapabilities(frozenset({"swap"}), True, False, True, frozenset({"market"}))

    async def connect(self, connection, credentials):
        raise NotImplementedError


def _entry_point(name: str, factory_name: str) -> metadata.EntryPoint:
    return metadata.EntryPoint(
        name=name,
        value=f"tests.test_venue_registry:{factory_name}",
        group="cryptotrader.venue_adapters",
    )


def create_alpha_adapter():
    return FakeAdapter("alpha")


def create_mismatched_adapter():
    return FakeAdapter("other")


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


def test_registry_discovers_entry_point_factories(monkeypatch):
    from cryptotrader.venues.registry import VenueAdapterRegistry

    monkeypatch.setattr(
        metadata,
        "entry_points",
        lambda *, group: [_entry_point("alpha", "create_alpha_adapter")],
    )

    registry = VenueAdapterRegistry.discover({"alpha"})

    assert registry.ids() == ("alpha",)
    assert registry.installed_ids() == frozenset({"alpha"})


def test_registry_rejects_entry_point_factory_id_mismatch(monkeypatch):
    from cryptotrader.venues.registry import VenueAdapterRegistry

    monkeypatch.setattr(
        metadata,
        "entry_points",
        lambda *, group: [_entry_point("alpha", "create_mismatched_adapter")],
    )

    with pytest.raises(ValueError, match="entry point alpha, adapter other"):
        VenueAdapterRegistry.discover({"alpha"})


def test_registry_rejects_duplicate_entry_point_ids(monkeypatch):
    from cryptotrader.venues.registry import VenueAdapterRegistry

    monkeypatch.setattr(
        metadata,
        "entry_points",
        lambda *, group: [
            _entry_point("alpha", "create_alpha_adapter"),
            _entry_point("alpha", "create_alpha_adapter"),
        ],
    )

    with pytest.raises(ValueError, match="duplicate venue adapter id: alpha"):
        VenueAdapterRegistry.discover({"alpha"})


def test_registry_rejects_configured_uninstalled_adapter(monkeypatch):
    from cryptotrader.venues.registry import VenueAdapterRegistry

    monkeypatch.setattr(metadata, "entry_points", lambda *, group: [])

    with pytest.raises(ValueError, match="uninstalled venue adapter ids: missing"):
        VenueAdapterRegistry.discover({"missing"})

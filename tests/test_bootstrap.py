"""Bootstrap 只接受两个外部参数并保持首次启动安全。"""

from __future__ import annotations

from importlib import import_module
from unittest.mock import AsyncMock

import pytest

MASTER_KEY = "A" * 43 + "="


def _settings_type():
    return import_module("cryptotrader.bootstrap").BootstrapSettings


def test_bootstrap_settings_read_exactly_two_environment_variables(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///test.db")
    monkeypatch.setenv("CONFIG_MASTER_KEY", MASTER_KEY)
    monkeypatch.setenv("CRYPTOTRADER_EXCHANGE_ID", "must-be-ignored")

    settings = _settings_type().from_environment()

    assert settings == _settings_type()("sqlite+aiosqlite:///test.db", MASTER_KEY)
    assert "must-be-ignored" not in repr(settings)


def test_bootstrap_settings_hide_master_key_and_are_frozen():
    settings = _settings_type()("sqlite+aiosqlite:///test.db", MASTER_KEY)

    assert MASTER_KEY not in repr(settings)
    with pytest.raises((AttributeError, TypeError)):
        settings.database_url = "sqlite+aiosqlite:///other.db"


@pytest.mark.parametrize(
    ("missing", "message"),
    [
        ("DATABASE_URL", "DATABASE_URL is required"),
        ("CONFIG_MASTER_KEY", "CONFIG_MASTER_KEY is required"),
    ],
)
def test_bootstrap_settings_missing_values_use_fixed_safe_errors(monkeypatch, missing, message):
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///test.db")
    monkeypatch.setenv("CONFIG_MASTER_KEY", MASTER_KEY)
    monkeypatch.delenv(missing)

    with pytest.raises(RuntimeError, match=f"^{message}$"):
        _settings_type().from_environment()


@pytest.mark.asyncio
async def test_first_start_seeds_setup_document_without_opening_venue_session(tmp_path):
    from cryptotrader.cycle_events import NullCycleEventSink
    from cryptotrader.runtime import build_runtime

    registry = _RecordingVenueRegistry()
    event_sink = NullCycleEventSink()
    settings = _settings_type()(f"sqlite+aiosqlite:///{tmp_path / 'runtime.db'}", MASTER_KEY)

    runtime = await build_runtime(
        settings,
        event_sink=event_sink,
        signal_registry=_InstalledRegistry({"kronos", "llm_committee"}),
        venue_registry=registry,
        market_registry=_InstalledRegistry({"default"}),
    )

    assert runtime.snapshot.setup_required is True
    assert runtime.events.base is event_sink
    assert registry.connect_calls == []
    await runtime.close()


@pytest.mark.asyncio
async def test_setup_discovers_all_metadata_without_resolving_or_opening_runtime_resources(monkeypatch, tmp_path):
    from cryptotrader.market_sources.registry import MarketSourceRegistry
    from cryptotrader.runtime import build_runtime
    from cryptotrader.runtime_config.repository import RuntimeConfigRepository
    from cryptotrader.signals.registry import SignalComponentRegistry
    from cryptotrader.venues.registry import VenueAdapterRegistry

    discovery_calls: list[str] = []
    signals = _MetadataRegistry({"kronos", "llm_committee"})
    venues = _MetadataRegistry({"paper", "okx", "bybit"})
    markets = _MetadataRegistry({"default"})

    def discover_signals(_cls, _document, _event_sink):
        discovery_calls.append("signals")
        return signals

    def discover_venues(_cls, configured_adapter_ids):
        assert tuple(configured_adapter_ids) == ()
        discovery_calls.append("venues")
        return venues

    def discover_markets(_cls, _market_data):
        discovery_calls.append("markets")
        return markets

    monkeypatch.setattr(SignalComponentRegistry, "discover", classmethod(discover_signals))
    monkeypatch.setattr(VenueAdapterRegistry, "discover", classmethod(discover_venues))
    monkeypatch.setattr(MarketSourceRegistry, "discover", classmethod(discover_markets))
    reveal = AsyncMock(side_effect=AssertionError("setup must not reveal sensitive material"))
    monkeypatch.setattr(RuntimeConfigRepository, "reveal_credentials", reveal)
    settings = _settings_type()(f"sqlite+aiosqlite:///{tmp_path / 'setup.db'}", MASTER_KEY)

    runtime = await build_runtime(settings)

    assert discovery_calls == ["signals", "venues", "markets"]
    assert signals.installed_calls == 1
    assert venues.installed_calls == 1
    assert markets.installed_calls == 1
    assert runtime.signal_registry is signals
    assert runtime.venue_registry is venues
    assert runtime.market_registry is markets
    assert signals.require_calls == []
    assert venues.require_calls == []
    assert markets.require_calls == []
    reveal.assert_not_awaited()
    assert runtime.sessions == {}
    assert runtime.cycle is None
    await runtime.close()


class _InstalledRegistry:
    def __init__(self, installed: set[str]) -> None:
        self._installed = frozenset(installed)

    def installed_ids(self):
        return self._installed


class _RecordingVenueRegistry(_InstalledRegistry):
    def __init__(self) -> None:
        super().__init__({"paper", "okx", "bybit"})
        self.connect_calls: list[str] = []

    def require(self, adapter_id):
        self.connect_calls.append(adapter_id)
        raise AssertionError("setup runtime must not resolve or connect a venue adapter")


class _MetadataRegistry(_InstalledRegistry):
    def __init__(self, installed: set[str]) -> None:
        super().__init__(installed)
        self.installed_calls = 0
        self.require_calls: list[str] = []

    def installed_ids(self):
        self.installed_calls += 1
        return super().installed_ids()

    def require(self, resource_id):
        self.require_calls.append(resource_id)
        raise AssertionError("setup runtime must not resolve executable resources")

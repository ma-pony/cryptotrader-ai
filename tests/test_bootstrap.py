"""Bootstrap 只接受两个外部参数并保持首次启动安全。"""

from __future__ import annotations

from importlib import import_module, metadata
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
    import cryptotrader.runtime as runtime_module
    from cryptotrader.market_sources.registry import MarketSourceRegistry
    from cryptotrader.runtime_config.defaults import minimal_runtime_document
    from cryptotrader.runtime_config.models import validate_runtime_document
    from cryptotrader.runtime_config.repository import RuntimeConfigRepository
    from cryptotrader.signals.registry import SignalComponentRegistry
    from cryptotrader.venues.registry import VenueAdapterRegistry

    downstream_calls: list[str] = []

    def fail_if_called(stage):
        def fail(*_args, **_kwargs):
            downstream_calls.append(stage)
            raise AssertionError(f"setup must not enter {stage}")

        return fail

    monkeypatch.setattr(SignalComponentRegistry, "enabled", fail_if_called("signal execution resolution"))
    monkeypatch.setattr(VenueAdapterRegistry, "require", fail_if_called("venue adapter resolution"))
    monkeypatch.setattr(MarketSourceRegistry, "require", fail_if_called("market source resolution"))
    monkeypatch.setattr(runtime_module, "_assemble_cycle", fail_if_called("cycle assembly"))
    reveal = AsyncMock(side_effect=fail_if_called("credential reveal"))
    monkeypatch.setattr(RuntimeConfigRepository, "reveal_credentials", reveal)
    settings = _settings_type()(f"sqlite+aiosqlite:///{tmp_path / 'setup.db'}", MASTER_KEY)

    runtime = await runtime_module.build_runtime(settings)
    document = runtime.snapshot.document
    configured_signal_ids = {component.component_id for component in document.signals.components}
    expected_signal_ids = {"kronos", "llm_committee"} | {
        entry_point.name for entry_point in metadata.entry_points(group="cryptotrader.signal_components")
    }
    expected_venue_ids = {
        entry_point.name for entry_point in metadata.entry_points(group="cryptotrader.venue_adapters")
    }
    expected_market_ids = {"default"} | {
        entry_point.name for entry_point in metadata.entry_points(group="cryptotrader.market_sources")
    }

    assert type(runtime.signal_registry) is SignalComponentRegistry
    assert type(runtime.venue_registry) is VenueAdapterRegistry
    assert type(runtime.market_registry) is MarketSourceRegistry
    assert document == minimal_runtime_document()
    assert runtime.signal_registry.installed_ids() == frozenset(expected_signal_ids)
    assert runtime.venue_registry.installed_ids() == frozenset(expected_venue_ids)
    assert runtime.market_registry.installed_ids() == frozenset(expected_market_ids)
    assert set(runtime.signal_registry.ids()) == configured_signal_ids
    assert set(runtime.venue_registry.ids()) == expected_venue_ids
    assert set(runtime.market_registry.ids()) == {document.market_data.source_id}
    validate_runtime_document(
        document,
        set(runtime.signal_registry.installed_ids()),
        set(runtime.venue_registry.installed_ids()),
        set(runtime.market_registry.installed_ids()),
    )
    assert runtime.snapshot.setup_required is True
    reveal.assert_not_awaited()
    assert downstream_calls == []
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

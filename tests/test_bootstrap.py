"""Bootstrap 只接受两个外部参数并保持首次启动安全。"""

from __future__ import annotations

from importlib import import_module

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

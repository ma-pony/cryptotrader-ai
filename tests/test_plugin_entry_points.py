"""Signal and market plugins are selected only from installed entry points."""

from __future__ import annotations

from datetime import UTC, datetime
from importlib import metadata

import pytest

from cryptotrader.cycle_events import NullCycleEventSink
from cryptotrader.pair import Pair
from cryptotrader.runtime_config.models import SignalComponentConfig
from tests.factories.runtime_config import market_config, runtime_document, signal_config

SIGNAL_GROUP = "cryptotrader.signal_components"
MARKET_GROUP = "cryptotrader.market_sources"


def _entry_point(name: str, value: str, group: str) -> metadata.EntryPoint:
    return metadata.EntryPoint(name=name, value=value, group=group)


def _patch_entry_points(monkeypatch, *entry_points: metadata.EntryPoint) -> None:
    def selected(*, group: str):
        return [entry_point for entry_point in entry_points if entry_point.group == group]

    monkeypatch.setattr(metadata, "entry_points", selected)


def _document_with(*components: SignalComponentConfig):
    return runtime_document(signals=signal_config(components=components))


def test_registry_discovers_only_configured_installed_component_entry_points(monkeypatch):
    from cryptotrader.signals.registry import SignalComponentRegistry

    _patch_entry_points(
        monkeypatch,
        _entry_point(
            "fixture_signal",
            "tests.factories.fake_signal_plugin:create_fixture_signal",
            SIGNAL_GROUP,
        ),
    )
    document = _document_with(
        SignalComponentConfig(component_id="kronos", enabled=True, weight=0.4),
        SignalComponentConfig(component_id="llm_committee", enabled=True, weight=0.4),
        SignalComponentConfig(component_id="fixture_signal", enabled=False, weight=0.2),
    )

    registry = SignalComponentRegistry.discover(document, sink=NullCycleEventSink())

    assert registry.ids() == ("kronos", "llm_committee", "fixture_signal")
    assert len(registry.get("llm_committee").agents) == 4


def test_signal_registry_rejects_configured_uninstalled_component(monkeypatch):
    from cryptotrader.signals.registry import SignalComponentRegistry

    _patch_entry_points(monkeypatch)
    document = _document_with(SignalComponentConfig(component_id="missing", enabled=True, weight=1.0))

    with pytest.raises(ValueError, match="uninstalled signal component ids: missing"):
        SignalComponentRegistry.discover(document, sink=NullCycleEventSink())


def test_signal_registry_rejects_conflicting_builtin_plugin_id(monkeypatch):
    from cryptotrader.signals.registry import SignalComponentRegistry

    _patch_entry_points(
        monkeypatch,
        _entry_point(
            "kronos",
            "tests.factories.fake_signal_plugin:create_conflicting_kronos",
            SIGNAL_GROUP,
        ),
    )

    with pytest.raises(ValueError, match="duplicate signal component id: kronos"):
        SignalComponentRegistry.discover(runtime_document(), sink=NullCycleEventSink())


def test_signal_registry_deduplicates_exact_editable_builtin_entry_point(monkeypatch):
    from cryptotrader.signals.registry import SignalComponentRegistry

    _patch_entry_points(
        monkeypatch,
        _entry_point(
            "kronos",
            "cryptotrader.signals.components.kronos:create_component",
            SIGNAL_GROUP,
        ),
        _entry_point(
            "llm_committee",
            "cryptotrader.signals.components.llm_committee:create_component",
            SIGNAL_GROUP,
        ),
    )

    registry = SignalComponentRegistry.discover(runtime_document(), sink=NullCycleEventSink())

    assert registry.ids() == ("kronos", "llm_committee")


def test_signal_registry_treats_database_python_path_as_opaque_parameters(monkeypatch):
    from cryptotrader.signals.registry import SignalComponentRegistry

    _patch_entry_points(
        monkeypatch,
        _entry_point(
            "fixture_signal",
            "tests.factories.fake_signal_plugin:create_fixture_signal",
            SIGNAL_GROUP,
        ),
    )
    document = _document_with(
        SignalComponentConfig(
            component_id="fixture_signal",
            enabled=True,
            weight=1.0,
            parameters={"python_path": "missing.module:create_component"},
        )
    )

    registry = SignalComponentRegistry.discover(document, sink=NullCycleEventSink())

    assert registry.get("fixture_signal").parameters["python_path"] == "missing.module:create_component"


def test_market_source_entry_point_returns_configured_source():
    from cryptotrader.market_sources.registry import MarketSourceRegistry

    registry = MarketSourceRegistry.discover(
        market_config(source_id="fixture-market", parameters={"exchange_id": "binance"}),
        entry_points=(
            _entry_point(
                "fixture-market",
                "tests.factories.fake_market_source_plugin:create_source",
                MARKET_GROUP,
            ),
        ),
    )

    assert registry.ids() == ("fixture-market",)
    assert registry.require("fixture-market").config.parameters["exchange_id"] == "binance"


def test_market_source_registry_rejects_configured_uninstalled_source():
    from cryptotrader.market_sources.registry import MarketSourceRegistry

    with pytest.raises(ValueError, match="uninstalled market source: missing"):
        MarketSourceRegistry.discover(market_config(source_id="missing"), entry_points=())


def test_market_source_registry_rejects_conflicting_builtin_plugin_id():
    from cryptotrader.market_sources.registry import MarketSourceRegistry

    entry_points = (
        _entry_point(
            "default",
            "tests.factories.fake_market_source_plugin:create_conflicting_default",
            MARKET_GROUP,
        ),
    )

    with pytest.raises(ValueError, match="duplicate market source id: default"):
        MarketSourceRegistry.discover(market_config(), entry_points=entry_points)


def test_market_source_registry_deduplicates_exact_editable_builtin_entry_point():
    from cryptotrader.market_sources.registry import MarketSourceRegistry

    registry = MarketSourceRegistry.discover(
        market_config(),
        entry_points=(
            _entry_point(
                "default",
                "cryptotrader.market_sources.default:create_source",
                MARKET_GROUP,
            ),
        ),
    )

    assert registry.ids() == ("default",)


@pytest.mark.asyncio
async def test_market_source_context_carries_source_id():
    from cryptotrader.market_sources.registry import MarketSourceRegistry

    registry = MarketSourceRegistry.discover(
        market_config(source_id="fixture-market"),
        entry_points=(
            _entry_point(
                "fixture-market",
                "tests.factories.fake_market_source_plugin:create_source",
                MARKET_GROUP,
            ),
        ),
    )
    source = registry.require("fixture-market")

    context = await source.collect(
        Pair.parse("BTC/USDT:USDT"),
        datetime(2026, 1, 1, tzinfo=UTC),
        source.requirements(),
    )

    assert context.market_data_source_id == "fixture-market"

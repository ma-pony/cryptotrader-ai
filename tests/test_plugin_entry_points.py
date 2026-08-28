"""Signal and market plugins are selected only from installed entry points."""

from __future__ import annotations

from datetime import UTC, datetime
from importlib import metadata

import pytest
from langchain_core.messages import AIMessage

from cryptotrader.cycle_events import NullCycleEventSink
from cryptotrader.pair import Pair
from cryptotrader.runtime_config.models import (
    LlmConfig,
    LlmModelCostConfig,
    LlmModelsConfig,
    LlmRetryConfig,
    SignalComponentConfig,
)
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


def test_runtime_llm_factory_builds_from_database_settings_without_legacy_config(monkeypatch):
    from langchain_core.outputs import ChatGeneration, LLMResult

    from cryptotrader.agents.base import create_runtime_llm_factory
    from cryptotrader.llm.token_tracker import set_ledger, start_ledger

    built = []
    retries = []

    class FakeChatModel:
        def __init__(self, **kwargs) -> None:
            built.append(kwargs)

    monkeypatch.setattr("cryptotrader.agents.base.ChatOpenAI", FakeChatModel)
    monkeypatch.setattr(
        "cryptotrader.llm.factory._wrap_with_retry",
        lambda llm, retry: retries.append(retry) or llm,
    )
    settings = LlmConfig(
        base_url="https://db-gateway.example/v1",
        streaming_models=("analysis-db",),
        default_temperature=0.17,
        timeout=47,
        prompt_caching=False,
        retry=LlmRetryConfig(max_attempts=7, retry_base_delay_s=0.25),
        model_costs=(LlmModelCostConfig(name="analysis-db", input_usd_per_mtok=1.2, output_usd_per_mtok=3.4),),
        models=LlmModelsConfig(analysis="analysis-db", fallback="analysis-db"),
    )

    llm = create_runtime_llm_factory(settings)()

    assert isinstance(llm, FakeChatModel)
    assert built == [
        {
            "model": "analysis-db",
            "temperature": 0.17,
            "timeout": 47,
            "api_key": "",
            "base_url": "https://db-gateway.example/v1",
            "streaming": True,
            "callbacks": built[0]["callbacks"],
        }
    ]
    assert built[0]["callbacks"][0].model_costs == {"analysis-db": (1.2, 3.4)}
    assert retries == [settings.retry]

    def reject_legacy_config():
        raise AssertionError("runtime token costs read legacy config")

    monkeypatch.setattr("cryptotrader.config.load_config", reject_legacy_config)
    ledger = start_ledger()
    try:
        message = AIMessage(
            content="answer",
            usage_metadata={"input_tokens": 1_000_000, "output_tokens": 1_000_000, "total_tokens": 2_000_000},
            response_metadata={"model_name": "analysis-db"},
        )
        built[0]["callbacks"][0].on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]))
        assert ledger.cost_usd == pytest.approx(4.6)
    finally:
        set_ledger(None)


@pytest.mark.asyncio
async def test_llm_committee_factory_resolves_empty_role_from_database_without_legacy_config(monkeypatch):
    from cryptotrader.signals.components.llm_committee import create_component
    from tests.test_llm_committee_component import RecordingSink, _context

    models = LlmModelsConfig(
        analysis="analysis-db",
        debate="debate-db",
        committee_summary="summary-db",
        tech_agent="",
        chain_agent="chain-db",
        news_agent="news-db",
        macro_agent="macro-db",
        fallback="fallback-db",
        timeout_seconds=11,
    )
    settings = LlmConfig(
        base_url="https://db-gateway.example/v1",
        default_temperature=0.19,
        timeout=43,
        prompt_caching=False,
        retry=LlmRetryConfig(max_attempts=5),
        model_costs=(LlmModelCostConfig(name="analysis-db", input_usd_per_mtok=1.0),),
        models=models,
    )
    document = runtime_document(
        llm=settings,
        signals=signal_config(
            components=(
                SignalComponentConfig(
                    component_id="llm_committee",
                    enabled=True,
                    weight=1.0,
                    parameters={"debate": {"skip_debate": False, "max_rounds": 1}},
                ),
            )
        ),
    )
    calls = []

    class FakeLLM:
        async def ainvoke(self, messages):
            system = str(messages[0].content)
            if "Summarize a four-domain market debate" in system:
                content = '{"direction":"long","confidence":0.72,"reasoning":"db summary"}'
            elif "specialist in a multi-agent trading debate" in system:
                content = (
                    '{"direction":"bullish","confidence":0.7,"reasoning":"debated",'
                    '"key_factors":[],"risk_flags":[],"new_findings":""}'
                )
            else:
                content = (
                    '{"direction":"bullish","confidence":0.7,"reasoning":"analysis",'
                    '"key_factors":[],"risk_flags":[],"data_sufficiency":"high"}'
                )
            return AIMessage(content=content)

    fake_llm = FakeLLM()

    def fake_llm_factory(**kwargs):
        calls.append(kwargs)
        return fake_llm

    captured_settings = []

    def fake_builder(config):
        captured_settings.append(config)
        return fake_llm_factory

    def reject_legacy_config():
        raise AssertionError("new discovery path read legacy load_config")

    monkeypatch.setattr("cryptotrader.config.load_config", reject_legacy_config)
    component = create_component(
        document,
        RecordingSink(),
        llm_factory_builder=fake_builder,
    )

    result = await component.evaluate(_context())

    assert captured_settings == [settings]
    assert (result.direction, result.confidence, result.reasoning) == ("long", 0.72, "db summary")
    assert len(result.details["analyses"]) == 4
    assert len(result.details["debate_turns"]) == 4
    assert [call["model"] for call in calls] == [
        "analysis-db",
        "chain-db",
        "news-db",
        "macro-db",
        "debate-db",
        "debate-db",
        "debate-db",
        "debate-db",
        "summary-db",
    ]


def test_llm_committee_factory_rejects_unresolved_empty_database_role(monkeypatch):
    from cryptotrader.signals.components.llm_committee import create_component

    settings = LlmConfig(
        models=LlmModelsConfig(
            analysis="",
            tech_agent="",
            fallback="",
        )
    )
    document = runtime_document(
        llm=settings,
        signals=signal_config(
            components=(SignalComponentConfig(component_id="llm_committee", enabled=True, weight=1.0),)
        ),
    )

    def reject_legacy_config():
        raise AssertionError("new discovery path read legacy load_config")

    monkeypatch.setattr("cryptotrader.config.load_config", reject_legacy_config)
    with pytest.raises(ValueError, match="tech_agent"):
        create_component(document, NullCycleEventSink())


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

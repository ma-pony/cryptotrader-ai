"""Code-registered signal/market factories and retained LLM dependency contracts."""

from datetime import UTC, datetime

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


def test_registry_constructs_builtin_peers_and_passes_context_to_code_factory(monkeypatch):
    from cryptotrader.configuration import registry
    from cryptotrader.signals.registry import SignalComponentRegistry
    from tests.factories.workbench_extensions import sample_registry

    extensions, calls = sample_registry()
    monkeypatch.setattr(registry, "get_extension_registry", lambda: extensions)
    document = runtime_document(
        signals=signal_config(
            components=(
                SignalComponentConfig(component_id="kronos", enabled=True, weight=0.4),
                SignalComponentConfig(component_id="llm_committee", enabled=True, weight=0.4),
                SignalComponentConfig(component_id="sample_signal", enabled=False, weight=0.2),
            )
        )
    )
    sink = NullCycleEventSink()
    discovered = SignalComponentRegistry.discover(document, sink, llm_gateway_key="explicit-key")
    assert discovered.ids() == ("kronos", "llm_committee", "sample_signal")
    assert len(discovered.get("llm_committee").agents) == 4
    context = discovered.get("sample_signal").context
    assert context.document is document
    assert context.events is sink
    assert context.llm_gateway_key == "explicit-key"
    assert calls == ["signal"]


def test_signal_registry_rejects_configured_unregistered_code_path():
    from cryptotrader.signals.registry import SignalComponentRegistry

    document = runtime_document(
        signals=signal_config(
            components=(SignalComponentConfig(component_id="missing.module:create_component", enabled=True, weight=1),)
        )
    )
    with pytest.raises(ValueError, match="unregistered signal component"):
        SignalComponentRegistry.discover(document, NullCycleEventSink())


def test_runtime_llm_factory_uses_only_the_explicit_vault_key(monkeypatch):
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
        "cryptotrader.llm.retry.wrap_with_retry",
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

    monkeypatch.setenv("OPENAI_API_KEY", "environment-decoy")
    llm = create_runtime_llm_factory(settings, api_key="vault-gateway-key")()  # pragma: allowlist secret

    assert isinstance(llm, FakeChatModel)
    assert built == [
        {
            "model": "analysis-db",
            "temperature": 0.17,
            "timeout": 47,
            "api_key": "vault-gateway-key",  # pragma: allowlist secret
            "base_url": "https://db-gateway.example/v1",
            "streaming": True,
            "callbacks": built[0]["callbacks"],
        }
    ]
    assert built[0]["callbacks"][0].model_costs == {"analysis-db": (1.2, 3.4)}
    assert retries == [settings.retry]

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


def test_runtime_llm_response_observer_records_actual_chat_metadata_not_configured_model(monkeypatch):
    from langchain_core.outputs import ChatGeneration, LLMResult

    from cryptotrader.agents.base import create_runtime_llm_factory

    built = []
    observed = []

    class FakeChatModel:
        def __init__(self, **kwargs) -> None:
            built.append(kwargs)

    monkeypatch.setattr("cryptotrader.agents.base.ChatOpenAI", FakeChatModel)
    monkeypatch.setattr("cryptotrader.llm.retry.wrap_with_retry", lambda llm, _retry: llm)
    settings = LlmConfig(models=LlmModelsConfig(analysis="configured-model", fallback="configured-model"))
    create_runtime_llm_factory(
        settings, api_key="gateway-key", response_observer=lambda role, model: observed.append((role, model))
    )(role="tech_agent")
    observer = built[0]["callbacks"][1]
    observer.on_chat_model_end(
        LLMResult(
            generations=[
                [
                    ChatGeneration(
                        message=AIMessage(content="ok", response_metadata={"model_name": "actual-response-model"})
                    )
                ]
            ]
        )
    )
    observer.on_chat_model_end(LLMResult(generations=[[ChatGeneration(message=AIMessage(content="missing"))]]))

    assert observed == [("tech_agent", "actual-response-model")]


@pytest.mark.asyncio
async def test_llm_committee_factory_resolves_empty_role_from_database_without_legacy_config(monkeypatch):
    from cryptotrader.signals.components.llm_committee import create_component
    from tests.test_runtime_signal_components import _committee_context as _context
    from tests.test_runtime_signal_components import _Sink as RecordingSink

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
    assert [call["role"] for call in calls] == [
        "tech_agent",
        "chain_agent",
        "news_agent",
        "macro_agent",
        "debate",
        "debate",
        "debate",
        "debate",
        "committee_summary",
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

    with pytest.raises(ValueError, match="tech_agent"):
        create_component(document, NullCycleEventSink())


def _install_market(monkeypatch):
    from cryptotrader.configuration import registry
    from cryptotrader.configuration.catalog import PluginConfiguration
    from cryptotrader.configuration.fields import LocalizedText
    from cryptotrader.configuration.parameters import DefaultMarketSourceParameters
    from cryptotrader.configuration.registry import ExtensionRegistration
    from tests.factories.fake_market_source_plugin import create_source

    extensions = registry.get_extension_registry()
    label = LocalizedText("测试市场", "Fixture market")
    extensions.market_sources["fixture-market"] = ExtensionRegistration(
        PluginConfiguration("fixture-market", label, label, DefaultMarketSourceParameters),
        lambda config, **kwargs: create_source(config),
    )
    monkeypatch.setattr(registry, "get_extension_registry", lambda: extensions)


def test_market_source_code_registration_returns_configured_source(monkeypatch):
    from cryptotrader.market_sources.registry import MarketSourceRegistry

    _install_market(monkeypatch)
    discovered = MarketSourceRegistry.discover(
        market_config(source_id="fixture-market", parameters={"market_adapter_id": "binance"})
    )
    assert discovered.ids() == ("fixture-market",)
    assert discovered.require("fixture-market").config.parameters["market_adapter_id"] == "binance"


def test_market_source_registry_rejects_configured_unregistered_source():
    from cryptotrader.market_sources.registry import MarketSourceRegistry

    with pytest.raises(ValueError, match="unregistered market source: missing"):
        MarketSourceRegistry.discover(market_config(source_id="missing"))


@pytest.mark.asyncio
async def test_market_source_context_carries_source_id(monkeypatch):
    from cryptotrader.market_sources.registry import MarketSourceRegistry

    _install_market(monkeypatch)
    source = MarketSourceRegistry.discover(market_config(source_id="fixture-market")).require("fixture-market")
    context = await source.collect(Pair.parse("BTC/USDT:USDT"), datetime(2026, 1, 1, tzinfo=UTC), source.requirements())
    assert context.market_data_source_id == "fixture-market"

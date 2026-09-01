"""Single code-owned registration point, with lazy runtime factory imports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import Field, SecretStr

from cryptotrader.configuration.catalog import ComponentDependency, EnvironmentDefinition, PluginConfiguration
from cryptotrader.configuration.fields import LocalizedText
from cryptotrader.configuration.parameters import (
    DefaultMarketSourceParameters,
    EmptyParameters,
    KronosParameters,
    LlmCommitteeParameters,
    PaperParameters,
    PluginParameters,
)
from cryptotrader.venues.models import ACCOUNT_READS, EXIT_OPERATIONS, VenueCapabilities

if TYPE_CHECKING:
    from collections.abc import Callable

    from cryptotrader.cycle_events import CycleEventSink
    from cryptotrader.runtime_config.models import RuntimeConfigDocument


@dataclass(frozen=True)
class ComponentFactoryContext:
    document: RuntimeConfigDocument
    events: CycleEventSink
    llm_gateway_key: str = field(default="", repr=False)
    llm_factory_builder: Callable | None = None


@dataclass(frozen=True)
class ExtensionRegistration:
    configuration: PluginConfiguration
    factory: Callable[..., Any]


@dataclass(frozen=True)
class ExtensionRegistry:
    components: dict[str, ExtensionRegistration]
    venues: dict[str, ExtensionRegistration]
    market_sources: dict[str, ExtensionRegistration]

    def __post_init__(self):
        for group in (self.components, self.venues, self.market_sources):
            for key, item in group.items():
                if key != item.configuration.id or not callable(item.factory):
                    raise ValueError("invalid extension registration")


class ApiCredentials(PluginParameters):
    api_key: SecretStr = Field(
        min_length=1, title="API key", json_schema_extra={"label": {"zh_CN": "API 密钥", "en_US": "API key"}}
    )
    secret: SecretStr = Field(
        min_length=1, title="API secret", json_schema_extra={"label": {"zh_CN": "API 私钥", "en_US": "API secret"}}
    )


class OkxCredentials(ApiCredentials):
    passphrase: SecretStr = Field(
        min_length=1, title="Passphrase", json_schema_extra={"label": {"zh_CN": "API 口令", "en_US": "Passphrase"}}
    )


def _kronos(context):
    from cryptotrader.signals.components.kronos import create_component

    return create_component(context.document, context.events)


def _committee(context):
    from cryptotrader.signals.components.llm_committee import create_component

    return create_component(
        context.document,
        context.events,
        llm_gateway_key=context.llm_gateway_key,
        llm_factory_builder=context.llm_factory_builder,
    )


def _paper():
    from cryptotrader.venues.paper import create_adapter

    return create_adapter()


def _okx():
    from cryptotrader.venues.okx import create_adapter

    return create_adapter()


def _bybit():
    from cryptotrader.venues.bybit import create_adapter

    return create_adapter()


def _market(config, *, news_provider_key=""):
    from cryptotrader.market_sources.default import create_source

    return create_source(config, news_provider_key=news_provider_key)


def _kronos_dependencies(parameters):
    return (
        ComponentDependency("market", parameters.timeframe, LocalizedText("K线行情", "Candles"), "market_data"),
        ComponentDependency(
            "local_artifact",
            parameters.gate_path,
            LocalizedText("状态门控文件", "Regime gate"),
            "signals.components.kronos.parameters.gate_path",
        ),
        ComponentDependency(
            "local_artifact",
            parameters.model_name,
            LocalizedText("预测模型", "Prediction model"),
            "signals.components.kronos.parameters.model_name",
        ),
        ComponentDependency(
            "local_artifact",
            parameters.tokenizer_name,
            LocalizedText("分词器", "Tokenizer"),
            "signals.components.kronos.parameters.tokenizer_name",
        ),
        ComponentDependency(
            "context", "kronos_aux", LocalizedText("Kronos 辅助行情", "Kronos auxiliary data"), "market_data"
        ),
    )


def _committee_dependencies(parameters):
    return (
        ComponentDependency("market", parameters.default_timeframe, LocalizedText("K线行情", "Candles"), "market_data"),
        ComponentDependency("model_service", "llm-gateway", LocalizedText("大模型服务", "Model service"), "llm"),
        *(
            ComponentDependency("context", key, LocalizedText(zh, en), "market_data")
            for key, zh, en in (
                ("onchain", "链上数据", "On-chain data"),
                ("news", "新闻", "News"),
                ("macro", "宏观数据", "Macro data"),
            )
        ),
    )


def _environment(key, zh, en, scope):
    return EnvironmentDefinition(key, LocalizedText(zh, en), scope)


def get_extension_registry() -> ExtensionRegistry:
    external_capabilities = VenueCapabilities(
        frozenset({"spot", "swap"}),
        True,
        True,
        True,
        frozenset({"market", "limit"}),
        ACCOUNT_READS,
        EXIT_OPERATIONS,
        7,
    )
    paper_capabilities = VenueCapabilities(
        frozenset({"spot", "swap"}),
        True,
        False,
        True,
        frozenset({"market", "limit"}),
        ACCOUNT_READS,
        EXIT_OPERATIONS,
    )
    declarations = (
        (
            "components",
            PluginConfiguration(
                "kronos",
                LocalizedText("Kronos 时序模型", "Kronos time-series model"),
                LocalizedText(
                    "使用市场时序基础模型和状态门控给出方向信号。",
                    "Produces directional signals with a time-series foundation model and regime gate.",
                ),
                KronosParameters,
                dependency_resolver=_kronos_dependencies,
            ),
            _kronos,
        ),
        (
            "components",
            PluginConfiguration(
                "llm_committee",
                LocalizedText("大模型四智能体委员会", "LLM four-agent committee"),
                LocalizedText(
                    "由技术、链上、新闻和宏观智能体进行内部辩论后汇总信号。",
                    "Synthesizes technical, on-chain, news, and macro analysis through an internal debate.",
                ),
                LlmCommitteeParameters,
                dependency_resolver=_committee_dependencies,
            ),
            _committee,
        ),
        (
            "venues",
            PluginConfiguration(
                "paper",
                LocalizedText("本地模拟器", "Paper trading"),
                LocalizedText(
                    "使用虚拟资金在系统内部模拟成交。不会向交易所提交订单。",
                    "Validates strategies in an isolated simulated account without submitting exchange orders.",
                ),
                PaperParameters,
                environments=(_environment("paper", "本地模拟", "Local simulation", "simulated"),),
                margin_modes=("cross",),
                capabilities=paper_capabilities,
            ),
            _paper,
        ),
        (
            "venues",
            PluginConfiguration(
                "okx",
                LocalizedText("OKX", "OKX"),
                LocalizedText("连接 OKX 官方模拟盘或实盘账户。", "Connects an OKX demo or live account."),
                EmptyParameters,
                environments=(
                    _environment("demo", "官方模拟盘", "Demo", "simulated"),
                    _environment("live", "实盘", "Live", "real"),
                ),
                credential_model=OkxCredentials,
                margin_modes=("cross", "isolated"),
                capabilities=external_capabilities,
            ),
            _okx,
        ),
        (
            "venues",
            PluginConfiguration(
                "bybit",
                LocalizedText("Bybit", "Bybit"),
                LocalizedText(
                    "连接 Bybit 测试网、官方模拟盘或实盘账户。", "Connects a Bybit testnet, demo, or live account."
                ),
                EmptyParameters,
                environments=(
                    _environment("testnet", "测试网", "Testnet", "simulated"),
                    _environment("demo", "官方模拟盘", "Demo", "simulated"),
                    _environment("live", "实盘", "Live", "real"),
                ),
                credential_model=ApiCredentials,
                margin_modes=("cross", "isolated"),
                capabilities=external_capabilities,
            ),
            _bybit,
        ),
        (
            "market_sources",
            PluginConfiguration(
                "default",
                LocalizedText("默认市场数据", "Default market data"),
                LocalizedText(
                    "采集公开市场、链上和宏观数据以构建交易上下文。",
                    "Collects public market, on-chain, and macro data for the trading context.",
                ),
                DefaultMarketSourceParameters,
            ),
            _market,
        ),
    )
    groups = {"components": {}, "venues": {}, "market_sources": {}}
    for group, configuration, factory in declarations:
        groups[group][configuration.id] = ExtensionRegistration(configuration, factory)
    return ExtensionRegistry(**groups)

"""A typed code-registration example, not an investment strategy."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from cryptotrader.configuration.catalog import PluginConfiguration
from cryptotrader.configuration.fields import LocalizedText
from cryptotrader.configuration.registry import ExtensionRegistration
from cryptotrader.signals.models import CandleRequirement, ComponentSignal, DataRequirements


class Diagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    enabled: bool = Field(
        default=False,
        json_schema_extra={
            "label": {"zh_CN": "包含诊断信息", "en_US": "Include diagnostics"},
            "description": {
                "zh_CN": "在中性输出中记录所用窗口。",
                "en_US": "Record the chosen window in neutral output.",
            },
            "advanced": True,
        },
    )


class Parameters(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    window: int = Field(
        default=20,
        ge=2,
        le=200,
        json_schema_extra={
            "label": {"zh_CN": "示例窗口", "en_US": "Example window"},
            "description": {"zh_CN": "请求的已闭合 K 线数量。", "en_US": "Number of closed candles requested."},
            "unit": "candles",
        },
    )
    timeframe: Literal["15m", "1h"] = Field(
        default="1h",
        json_schema_extra={
            "label": {"zh_CN": "示例周期", "en_US": "Example timeframe"},
            "description": {"zh_CN": "选择窗口的 K 线周期。", "en_US": "Choose the candle timeframe."},
            "options": [
                {"value": "15m", "label": {"zh_CN": "15 分钟", "en_US": "15 minutes"}},
                {"value": "1h", "label": {"zh_CN": "1 小时", "en_US": "1 hour"}},
            ],
        },
    )
    diagnostics: Diagnostics = Field(default_factory=Diagnostics)


class ExampleComponent:
    id = "configuration_example"
    display_name = "Typed configuration example"
    description = "An offline example that always returns neutral."

    def __init__(self, parameters: Parameters):
        self.parameters = parameters

    def requirements(self):
        return DataRequirements(candles=(CandleRequirement(self.parameters.timeframe, self.parameters.window),))

    async def evaluate(self, context):
        details = {"window": self.parameters.window} if self.parameters.diagnostics.enabled else {}
        return ComponentSignal(
            self.id, "neutral", 0.0, "Configuration example only; no trading recommendation.", details
        )


configuration = PluginConfiguration(
    id="configuration_example",
    label=LocalizedText("类型化配置示例", "Typed configuration example"),
    description=LocalizedText(
        "仅演示后端代码注册组件字段。始终输出中性。不调用模型。",
        "Demonstrates typed fields; always neutral, with no model calls.",
    ),
    parameter_model=Parameters,
)


def create_component(context):
    document = context.document
    configured = next(item for item in document.signals.components if item.component_id == "configuration_example")
    return ExampleComponent(Parameters.model_validate(dict(configured.parameters)))


registration = ExtensionRegistration(configuration, create_component)


def register_example(extensions) -> None:
    """Explicitly add this example during application-owned registry setup."""
    extensions.components[configuration.id] = registration

"""Parameter models consumed by the built-in plugin factories."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from cryptotrader.configuration.fields import LocalizedText


class PluginParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    field_labels: ClassVar[dict[str, LocalizedText]] = {}
    field_descriptions: ClassVar[dict[str, LocalizedText]] = {}


class EmptyParameters(PluginParameters):
    pass


class PaperParameters(PluginParameters):
    field_labels = {"initial_equity": LocalizedText("模拟初始资金", "Initial simulated equity")}
    field_descriptions = {
        "initial_equity": LocalizedText("模拟账户的起始资金。", "Starting balance for this simulated account.")
    }
    initial_equity: float = Field(
        default=10_000,
        gt=0,
        title="Initial simulated equity",
        description="Starting balance for this independent simulated account.",
        json_schema_extra={
            "label": {"zh_CN": "模拟初始资金", "en_US": "Initial simulated equity"},
            "description": {"zh_CN": "该模拟账户的起始资金。", "en_US": "Starting balance for this simulated account."},
            "unit": "USDT",
        },
    )

    @field_validator("initial_equity", mode="before")
    @classmethod
    def reject_boolean_equity(cls, value: Any) -> Any:
        if type(value) is bool:
            raise ValueError("initial_equity must be a positive number")
        return value


class KronosParameters(PluginParameters):
    field_labels = {
        "gate_path": LocalizedText("状态门控文件路径", "Regime gate path"),
        "model_name": LocalizedText("预测模型名称", "Prediction model name"),
        "tokenizer_name": LocalizedText("分词器名称", "Tokenizer name"),
        "device": LocalizedText("推理设备", "Inference device"),
        "lookback": LocalizedText("回看K线数量", "Lookback candles"),
        "pred_len": LocalizedText("预测K线数量", "Prediction horizon"),
        "sample_count": LocalizedText("预测采样次数", "Prediction samples"),
        "step2_short_threshold": LocalizedText("做空信号阈值", "Short signal threshold"),
        "aux_symbol": LocalizedText("辅助市场标的", "Auxiliary symbol"),
        "timeframe": LocalizedText("K线周期", "Candle timeframe"),
        "ohlcv_limit": LocalizedText("K线数据上限", "OHLCV limit"),
    }
    gate_path: str = Field(
        default="artifacts/kronos/gate_v21.pkl", title="Regime gate path", json_schema_extra={"advanced": True}
    )
    model_name: str = Field(default="NeoQuasar/Kronos-base", title="Model name", json_schema_extra={"advanced": True})
    tokenizer_name: str = Field(
        default="NeoQuasar/Kronos-Tokenizer-base", title="Tokenizer name", json_schema_extra={"advanced": True}
    )
    device: str = Field(default="", title="Inference device", json_schema_extra={"advanced": True})
    lookback: int = Field(default=460, ge=1, title="Lookback candles", json_schema_extra={"unit": "candles"})
    pred_len: int = Field(default=50, ge=1, title="Prediction horizon", json_schema_extra={"unit": "candles"})
    sample_count: int = Field(default=5, ge=1, title="Prediction samples", json_schema_extra={"advanced": True})
    step2_short_threshold: float = Field(
        default=0.04, ge=0, title="Short signal threshold", json_schema_extra={"unit": "ratio"}
    )
    aux_symbol: str = Field(default="BTCUSDT", title="Auxiliary symbol", json_schema_extra={"advanced": True})
    timeframe: str = Field(default="4h", title="Candle timeframe")
    ohlcv_limit: int = Field(default=512, ge=1, title="OHLCV limit", json_schema_extra={"unit": "candles"})


class DebateParameters(PluginParameters):
    field_labels = {
        "max_rounds": LocalizedText("最大辩论轮数", "Maximum rounds"),
        "convergence_threshold": LocalizedText("收敛阈值", "Convergence threshold"),
        "divergence_hold_threshold": LocalizedText("分歧观望阈值", "Divergence hold threshold"),
        "skip_debate": LocalizedText("共识明确时跳过辩论", "Skip debate when consensus is clear"),
        "consensus_skip_threshold": LocalizedText("共识跳过阈值", "Consensus skip threshold"),
        "confusion_skip_threshold": LocalizedText("混乱跳过阈值", "Confusion skip threshold"),
        "confusion_max_dispersion": LocalizedText("最大混乱离散度", "Maximum confusion dispersion"),
    }
    max_rounds: int = Field(default=3, ge=1, title="Maximum rounds")
    convergence_threshold: float = Field(default=0.1, ge=0, le=1, title="Convergence threshold")
    divergence_hold_threshold: float = Field(default=0.7, ge=0, le=1, title="Divergence hold threshold")
    skip_debate: bool = Field(default=True, title="Skip debate when consensus is clear")
    consensus_skip_threshold: float = Field(default=0.5, ge=0, le=1, title="Consensus skip threshold")
    confusion_skip_threshold: float = Field(default=0.05, ge=0, le=1, title="Confusion skip threshold")
    confusion_max_dispersion: float = Field(default=0.2, ge=0, le=1, title="Maximum confusion dispersion")


class LlmCommitteeParameters(PluginParameters):
    field_labels = {
        "default_timeframe": LocalizedText("默认K线周期", "Default timeframe"),
        "ohlcv_limit": LocalizedText("K线数据上限", "OHLCV limit"),
    }
    default_timeframe: str = Field(default="1h", title="Default timeframe")
    ohlcv_limit: int = Field(default=100, ge=1, title="OHLCV limit", json_schema_extra={"unit": "candles"})
    debate: DebateParameters = Field(default_factory=DebateParameters, title="Internal debate")


class DefaultMarketSourceParameters(PluginParameters):
    field_labels = {
        "market_adapter_id": LocalizedText("市场数据交易所", "Market data adapter"),
        "kronos_aux_symbol": LocalizedText("Kronos 辅助标的", "Kronos auxiliary symbol"),
    }
    market_adapter_id: str = Field(default="binance", min_length=1, title="Market data adapter")
    kronos_aux_symbol: str = Field(default="BTCUSDT", min_length=1, title="Kronos auxiliary symbol")

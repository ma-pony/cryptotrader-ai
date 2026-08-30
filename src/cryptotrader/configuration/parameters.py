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
    field_descriptions = {
        "gate_path": LocalizedText("服务器上的状态门控模型文件。", "Server-local regime gate model file."),
        "model_name": LocalizedText(
            "预测器加载的 Kronos 模型标识。", "Kronos model identifier loaded by the predictor."
        ),
        "tokenizer_name": LocalizedText(
            "与预测模型配套的分词器标识。", "Tokenizer identifier used with the prediction model."
        ),
        "device": LocalizedText(
            "留空自动选择可用设备。也可指定 cpu、cuda 或 mps。",
            "Leave empty for automatic device selection, or specify cpu, cuda or mps.",
        ),
        "lookback": LocalizedText(
            "每次预测使用的最近K线数量。行情数据需覆盖此窗口。",
            "Recent candles used for each prediction; market data must cover this window.",
        ),
        "pred_len": LocalizedText("向未来预测的K线数量。", "Number of future candles to predict."),
        "sample_count": LocalizedText(
            "每次预测的采样次数。更多采样增加推理工作量。",
            "Samples per prediction; more samples increase inference work.",
        ),
        "step2_short_threshold": LocalizedText(
            "负向原始信号幅度低于此阈值时输出中性信号。",
            "Negative raw signals weaker than this magnitude produce a neutral signal.",
        ),
        "aux_symbol": LocalizedText(
            "状态特征使用的辅助市场标的。", "Auxiliary market symbol used for regime features."
        ),
        "timeframe": LocalizedText(
            "该组件请求行情并生成预测时间戳的K线周期。",
            "Candle timeframe requested by this component and used for forecast timestamps.",
        ),
        "ohlcv_limit": LocalizedText(
            "为该组件请求的历史K线数量。应覆盖回看窗口。",
            "Historical candles requested for this component; should cover the lookback window.",
        ),
    }
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
    field_descriptions = {
        "max_rounds": LocalizedText("内部辩论最多执行的轮数。", "Maximum number of internal debate rounds."),
        "convergence_threshold": LocalizedText(
            "相邻轮次的分歧相对变化低于此值时结束辩论。",
            "End debate when the relative change in divergence between rounds is below this value.",
        ),
        "skip_debate": LocalizedText(
            "允许在强共识或低分歧的共同不确定状态下跳过辩论。",
            "Allow skipping debate for strong consensus or shared uncertainty with low dispersion.",
        ),
        "consensus_skip_threshold": LocalizedText(
            "共识强度高于此值时可跳过辩论。", "Consensus strength above this value may skip debate."
        ),
        "confusion_skip_threshold": LocalizedText(
            "平均方向分数绝对值低于此值。且离散度足够低时。可跳过辩论。",
            "Skip when the absolute mean directional score is below this value and dispersion is also low.",
        ),
        "confusion_max_dispersion": LocalizedText(
            "共同不确定时允许跳过辩论的最大离散度。",
            "Maximum dispersion allowed when skipping debate for shared uncertainty.",
        ),
    }
    field_labels = {
        "max_rounds": LocalizedText("最大辩论轮数", "Maximum rounds"),
        "convergence_threshold": LocalizedText("收敛阈值", "Convergence threshold"),
        "skip_debate": LocalizedText("允许条件性跳过辩论", "Allow conditional debate skipping"),
        "consensus_skip_threshold": LocalizedText("共识跳过阈值", "Consensus skip threshold"),
        "confusion_skip_threshold": LocalizedText("混乱跳过阈值", "Confusion skip threshold"),
        "confusion_max_dispersion": LocalizedText("最大混乱离散度", "Maximum confusion dispersion"),
    }
    max_rounds: int = Field(default=3, ge=1, title="Maximum rounds")
    convergence_threshold: float = Field(
        default=0.1, ge=0, le=1, title="Convergence threshold", json_schema_extra={"advanced": True}
    )
    skip_debate: bool = Field(default=True, title="Skip debate when consensus is clear")
    consensus_skip_threshold: float = Field(
        default=0.5, ge=0, le=1, title="Consensus skip threshold", json_schema_extra={"advanced": True}
    )
    confusion_skip_threshold: float = Field(
        default=0.05, ge=0, le=1, title="Confusion skip threshold", json_schema_extra={"advanced": True}
    )
    confusion_max_dispersion: float = Field(
        default=0.2, ge=0, le=1, title="Maximum confusion dispersion", json_schema_extra={"advanced": True}
    )


class LlmCommitteeParameters(PluginParameters):
    field_descriptions = {
        "default_timeframe": LocalizedText(
            "委员会请求的默认K线周期。", "Default candle timeframe requested by the committee."
        ),
        "ohlcv_limit": LocalizedText(
            "提供给委员会分析的历史K线数量。", "Historical candles supplied for committee analysis."
        ),
    }
    field_labels = {
        "default_timeframe": LocalizedText("默认K线周期", "Default timeframe"),
        "ohlcv_limit": LocalizedText("K线数据上限", "OHLCV limit"),
    }
    default_timeframe: str = Field(default="1h", title="Default timeframe")
    ohlcv_limit: int = Field(default=100, ge=1, title="OHLCV limit", json_schema_extra={"unit": "candles"})
    debate: DebateParameters = Field(default_factory=DebateParameters, title="Internal debate")


class DefaultMarketSourceParameters(PluginParameters):
    field_descriptions = {
        "market_adapter_id": LocalizedText(
            "行情采集使用的已安装交易所适配器。", "Installed exchange adapter used for market data collection."
        ),
        "kronos_aux_symbol": LocalizedText(
            "默认行情源为 Kronos 特征采集的辅助标的。",
            "Auxiliary symbol collected by the default source for Kronos features.",
        ),
        "timeframe": LocalizedText(
            "统一退出策略计算 ATR 时使用的K线周期。",
            "Candle timeframe used to calculate ATR for the shared exit policy.",
        ),
        "limit": LocalizedText(
            "退出策略请求的K线数量。至少 20 根。", "Candles requested for the exit policy, with a minimum of 20."
        ),
    }
    field_labels = {
        "market_adapter_id": LocalizedText("市场数据交易所", "Market data adapter"),
        "kronos_aux_symbol": LocalizedText("Kronos 辅助标的", "Kronos auxiliary symbol"),
        "timeframe": LocalizedText("止损计算K线周期", "Exit calculation timeframe"),
        "limit": LocalizedText("止损计算K线数量", "Exit calculation candles"),
    }
    market_adapter_id: str = Field(default="binance", min_length=1, title="Market data adapter")
    kronos_aux_symbol: str = Field(
        default="BTCUSDT", min_length=1, title="Kronos auxiliary symbol", json_schema_extra={"advanced": True}
    )
    timeframe: str = Field(default="1h", min_length=1, title="Exit calculation timeframe")
    limit: int = Field(default=100, ge=20, title="Exit calculation candles", json_schema_extra={"unit": "candles"})

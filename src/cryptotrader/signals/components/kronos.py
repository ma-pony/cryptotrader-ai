"""Kronos foundation model as a pure directional signal component."""
# ruff: noqa: RUF001

from __future__ import annotations

import asyncio
import math
import pickle
import sys
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from cryptotrader.agents._kronos_features import compute_kronos_features
from cryptotrader.configuration.parameters import KronosParameters
from cryptotrader.signals.component import ComponentExecutionError
from cryptotrader.signals.models import CandleRequirement, ComponentSignal, DataRequirements
from cryptotrader.signals.presentation import (
    Metric,
    MetricsBlock,
    Series,
    SeriesBlock,
    SeriesPoint,
    TextBlock,
    interval_delta,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from cryptotrader.models import DataSnapshot
    from cryptotrader.runtime_config.models import RuntimeConfigDocument
    from cryptotrader.signals.models import SignalContext

_gate_cache: dict[Path, dict[str, Any]] = {}
_predictor_cache: dict[str, Any] = {}


@dataclass(frozen=True)
class KronosSettings:
    gate_path: str = "artifacts/kronos/gate_v21.pkl"
    model_name: str = "NeoQuasar/Kronos-base"
    tokenizer_name: str = "NeoQuasar/Kronos-Tokenizer-base"
    device: str = ""
    lookback: int = 460
    pred_len: int = 50
    sample_count: int = 5
    step2_short_threshold: float = 0.04
    aux_symbol: str = "BTCUSDT"
    timeframe: str = "4h"
    ohlcv_limit: int = 512


def _device(config: KronosSettings) -> str:
    if config.device:
        return config.device
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _gate_path(config: KronosSettings) -> Path:
    path = Path(config.gate_path)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[4] / path


def _load_gate(config: KronosSettings) -> dict[str, Any]:
    path = _gate_path(config)
    if path not in _gate_cache:
        with path.open("rb") as stream:
            _gate_cache[path] = pickle.load(stream)
    return _gate_cache[path]


def _load_predictor(config: KronosSettings):
    device = _device(config)
    cache_key = f"{config.model_name}::{config.tokenizer_name}::{device}"
    if cache_key in _predictor_cache:
        return _predictor_cache[cache_key]

    project_root = str(Path(__file__).resolve().parents[4])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from vendor.kronos_model.kronos import Kronos, KronosPredictor, KronosTokenizer

    tokenizer = KronosTokenizer.from_pretrained(config.tokenizer_name).to(device).eval()
    model = Kronos.from_pretrained(config.model_name).to(device).eval()
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512, clip=5)
    _predictor_cache[cache_key] = predictor
    return predictor


def _input_frame(snapshot: DataSnapshot, lookback: int) -> pd.DataFrame:
    frame = snapshot.market.ohlcv[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
    frame = frame.tail(lookback).copy()
    frame["amount"] = frame["close"] * frame["volume"]
    return frame.reset_index(drop=True)


def _timestamps(snapshot: DataSnapshot, as_of, lookback: int, pred_len: int, timeframe: str):
    frame = snapshot.market.ohlcv
    if "timestamp" in frame.columns:
        raw = frame.dropna(subset=["close"])["timestamp"].tail(lookback)
        unit = "ms" if pd.api.types.is_numeric_dtype(raw) else None
        history = pd.Series(pd.to_datetime(raw, unit=unit, utc=True)).reset_index(drop=True)
    elif isinstance(frame.index, pd.DatetimeIndex):
        history = pd.Series(pd.to_datetime(frame.dropna(subset=["close"]).index[-lookback:], utc=True))
    else:
        raise ValueError("missing candle timestamps")
    offset = interval_delta(timeframe)
    future = pd.Series([history.iloc[-1] + offset * (index + 1) for index in range(pred_len)])
    return history, future


def _horizon_signal(prediction: pd.DataFrame, last_close: float) -> tuple[float, float, float]:
    returns = prediction["close"].to_numpy(dtype=float) / last_close - 1.0
    h10_20 = float(np.mean(returns[9:20])) if len(returns) >= 20 else float(np.mean(returns))
    h30_50 = float(np.mean(returns[29:50])) if len(returns) >= 50 else float(np.mean(returns))
    return 0.5 * h10_20 + 0.5 * h30_50, h10_20, h30_50


def _confidence(gate_proba: float, raw_signal: float, annual_volatility: float, h10_20: float, h30_50: float):
    def clip(value: float) -> float:
        return max(0.0, min(1.0, value))

    c_gate = clip((gate_proba - 0.5) / 0.4)
    c_signal = clip((abs(raw_signal) - 0.005) / 0.045)
    c_volatility = clip((1.0 / (annual_volatility + 0.1) - 0.5) / 4.5)
    c_drift = 0.5
    c_horizon = 1.0 if math.copysign(1, h10_20) == math.copysign(1, h30_50) else 0.0
    value = clip(0.40 * c_gate + 0.20 * c_signal + 0.20 * c_volatility + 0.10 * c_drift + 0.10 * c_horizon)
    return value, {
        "c_gate": c_gate,
        "c_signal": c_signal,
        "c_volatility": c_volatility,
        "c_drift": c_drift,
        "c_horizon": c_horizon,
    }


class KronosComponent:
    id = "kronos"
    display_name = "Kronos"
    description = "Kronos time-series foundation model with a regime gate"

    def __init__(
        self,
        config: KronosSettings,
        *,
        gate_loader: Callable[[KronosSettings], dict[str, Any]] = _load_gate,
        predictor_loader: Callable[[KronosSettings], Any] = _load_predictor,
        feature_computer: Callable[[DataSnapshot, list[str], dict[str, float]], dict[str, float]] = (
            compute_kronos_features
        ),
    ) -> None:
        self.config = config
        self._gate_loader = gate_loader
        self._predictor_loader = predictor_loader
        self._feature_computer = feature_computer

    def requirements(self) -> DataRequirements:
        return DataRequirements(
            candles=(CandleRequirement(self.config.timeframe, self.config.ohlcv_limit),),
            onchain=True,
            macro=True,
            kronos_aux=True,
        )

    async def evaluate(self, context: SignalContext) -> ComponentSignal:
        snapshot = self._stage("input data", lambda: context.snapshots[self.config.timeframe])
        gate = self._stage("gate loading", lambda: self._gate_loader(self.config))
        features, annual_volatility = self._stage(
            "feature computation",
            lambda: self._features(snapshot, gate),
        )
        gate_proba = self._stage(
            "gate classification",
            lambda: self._gate_probability(gate, features),
        )

        if gate_proba < 0.5:
            return self._neutral(
                f"市场状态门控未通过（{gate_proba:.4f}），本次未执行预测。",
                status="skipped",
                reference=context.evaluation_reference,
                gate_proba=gate_proba,
            )

        frame, history_timestamps, future_timestamps = self._stage(
            "input data",
            lambda: self._prepare_input(snapshot, context.as_of),
        )
        prediction = await self._predict(frame, history_timestamps, future_timestamps)
        raw_signal, h10_20, h30_50 = self._stage(
            "prediction output",
            lambda: _horizon_signal(prediction, float(frame["close"].iloc[-1])),
        )

        details = {
            "raw_signal": raw_signal,
            "gate_proba": gate_proba,
            "h10_20": h10_20,
            "h30_50": h30_50,
            "predictor_executed": True,
        }
        curve = self._stage(
            "prediction output", lambda: self._curve(frame, prediction, history_timestamps, future_timestamps)
        )
        confidence, dimensions = _confidence(gate_proba, raw_signal, annual_volatility, h10_20, h30_50)
        details.update(dimensions)
        labels = {
            "c_gate": "门控分项",
            "c_signal": "信号分项",
            "c_volatility": "波动分项",
            "c_drift": "漂移分项",
            "c_horizon": "周期一致性",
        }
        metrics = MetricsBlock(
            title="置信度分项", metrics=tuple(Metric(key=labels[key], value=value) for key, value in dimensions.items())
        )
        if raw_signal == 0.0 or (raw_signal < 0.0 and abs(raw_signal) < self.config.step2_short_threshold):
            return self._neutral(
                "本次已完成预测；零信号或弱空头信号被过滤，方向设为中性。",
                blocks=(curve, metrics),
                reference=context.evaluation_reference,
                **details,
            )

        direction = "long" if raw_signal > 0.0 else "short"
        return ComponentSignal(
            component_id=self.id,
            direction=direction,
            confidence=confidence,
            reasoning=(
                f"Kronos predicts {direction}: signal={raw_signal:.6f}, gate={gate_proba:.4f}, "
                f"h10_20={h10_20:.6f}, h30_50={h30_50:.6f}"
            ),
            details=details,
            blocks=(curve, metrics),
            evaluation_reference=context.evaluation_reference,
        )

    def _neutral(self, reasoning: str, *, blocks=(), status="completed", reference=None, **details) -> ComponentSignal:
        return ComponentSignal(
            self.id,
            "neutral",
            0.0,
            reasoning,
            details,
            blocks=(TextBlock(title="运行说明", body=reasoning), *blocks),
            status=status,
            evaluation_reference=reference,
        )

    @staticmethod
    def _curve(frame, prediction, history, future) -> SeriesBlock:
        if len(prediction) != len(future):
            raise ValueError("prediction length differs from requested timestamps")

        def points(timestamps, values):
            return tuple(
                SeriesPoint(time=time.to_pydatetime(), value=Decimal(str(value)))
                for time, value in zip(timestamps, values, strict=True)
            )

        return SeriesBlock(
            title="历史价格与当次预测",
            forecast_start=future.iloc[0].to_pydatetime(),
            evaluation_target="candle_close",
            series=(
                Series(name="历史收盘价", points=points(history, frame["close"])),
                Series(name="预测收盘价", points=points(future, prediction["close"])),
            ),
        )

    def _features(self, snapshot: DataSnapshot, gate: dict) -> tuple[dict[str, float], float]:
        features = dict(self._feature_computer(snapshot, gate["feat_cols"], gate["medians"]))
        return features, float(features.pop("_vol5", 0.0))

    @staticmethod
    def _gate_probability(gate: dict, features: dict[str, float]) -> float:
        values = np.array([[features[column] for column in gate["feat_cols"]]], dtype=float)
        return float(gate["classifier"].predict_proba(gate["scaler"].transform(values))[0][1])

    def _prepare_input(self, snapshot: DataSnapshot, as_of):
        frame = _input_frame(snapshot, self.config.lookback)
        if len(frame) < self.config.lookback:
            raise ValueError(f"need {self.config.lookback} bars, got {len(frame)}")
        history, future = _timestamps(
            snapshot,
            as_of,
            len(frame),
            self.config.pred_len,
            self.config.timeframe,
        )
        return frame, history, future

    async def _predict(self, frame, history_timestamps, future_timestamps):
        try:
            predictor = await asyncio.to_thread(self._predictor_loader, self.config)
            return await asyncio.to_thread(
                predictor.predict,
                df=frame,
                x_timestamp=history_timestamps,
                y_timestamp=future_timestamps,
                pred_len=self.config.pred_len,
                T=1.0,
                top_p=0.9,
                top_k=0,
                sample_count=self.config.sample_count,
                verbose=False,
            )
        except Exception as error:
            raise self._error("prediction", error) from error

    def _stage(self, stage: str, function: Callable[[], Any]):
        try:
            return function()
        except Exception as error:
            raise self._error(stage, error) from error

    def _error(self, stage: str, cause: BaseException) -> ComponentExecutionError:
        return ComponentExecutionError(self.id, RuntimeError(type(cause).__name__), stage=stage.replace(" ", "_"))


def create_component(document: RuntimeConfigDocument, sink) -> KronosComponent:
    """Build Kronos from its database-owned component parameters."""
    del sink
    configured = next(item for item in document.signals.components if item.component_id == KronosComponent.id)
    parameters = KronosParameters.model_validate(dict(configured.parameters))
    return KronosComponent(KronosSettings(**parameters.model_dump()))

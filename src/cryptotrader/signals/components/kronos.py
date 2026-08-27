"""Kronos foundation model as a pure directional signal component."""

from __future__ import annotations

import asyncio
import math
import pickle
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from cryptotrader.agents._kronos_features import compute_kronos_features
from cryptotrader.signals.component import ComponentExecutionError
from cryptotrader.signals.models import CandleRequirement, ComponentSignal, DataRequirements

if TYPE_CHECKING:
    from collections.abc import Callable

    from cryptotrader.config import KronosConfig
    from cryptotrader.models import DataSnapshot
    from cryptotrader.signals.models import SignalContext

_gate_cache: dict[Path, dict[str, Any]] = {}
_predictor_cache: dict[str, Any] = {}


def _device(config: KronosConfig) -> str:
    if config.device:
        return config.device
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _gate_path(config: KronosConfig) -> Path:
    path = Path(config.gate_path)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[4] / path


def _load_gate(config: KronosConfig) -> dict[str, Any]:
    path = _gate_path(config)
    if path not in _gate_cache:
        with path.open("rb") as stream:
            _gate_cache[path] = pickle.load(stream)
    return _gate_cache[path]


def _load_predictor(config: KronosConfig):
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
        raw = frame["timestamp"].tail(lookback)
        unit = "ms" if pd.api.types.is_numeric_dtype(raw) else None
        history = pd.Series(pd.to_datetime(raw, unit=unit, utc=True)).reset_index(drop=True)
    elif isinstance(frame.index, pd.DatetimeIndex):
        history = pd.Series(pd.to_datetime(frame.index[-lookback:], utc=True))
    else:
        history = pd.Series(pd.date_range(end=as_of, periods=lookback, freq=timeframe, tz="UTC"))
    offset = pd.tseries.frequencies.to_offset(timeframe)
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
        config: KronosConfig,
        *,
        gate_loader: Callable[[KronosConfig], dict[str, Any]] = _load_gate,
        predictor_loader: Callable[[KronosConfig], Any] = _load_predictor,
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
                f"Kronos regime gate rejected at {gate_proba:.4f}",
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
        }
        if raw_signal == 0.0 or (raw_signal < 0.0 and abs(raw_signal) < self.config.step2_short_threshold):
            return self._neutral("Kronos weak short filtered by Step 2", **details)

        confidence, dimensions = _confidence(gate_proba, raw_signal, annual_volatility, h10_20, h30_50)
        details.update(dimensions)
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
        )

    def _neutral(self, reasoning: str, **details) -> ComponentSignal:
        return ComponentSignal(self.id, "neutral", 0.0, reasoning, details)

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
        return ComponentExecutionError(self.id, RuntimeError(f"{stage}: {cause}"))

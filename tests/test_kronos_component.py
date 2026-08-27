"""KronosComponent 只输出方向与置信度, 不拥有仓位和退出策略。"""

from __future__ import annotations

import asyncio
import math
from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cryptotrader.config import KronosConfig
from cryptotrader.models import DataSnapshot, MacroData, MarketData, NewsSentiment, OnchainData
from tests.factories.signal_fusion import context


def _snapshot(rows: int = 20, price: float = 100.0) -> DataSnapshot:
    closes = np.linspace(price * 0.9, price, rows)
    market = MarketData(
        pair="BTC/USDT:USDT",
        ohlcv=pd.DataFrame(
            {
                "open": closes,
                "high": closes + 1.0,
                "low": closes - 1.0,
                "close": closes,
                "volume": np.full(rows, 10.0),
            }
        ),
        ticker={"last": price},
        funding_rate=0.0,
        orderbook_imbalance=0.0,
        volatility=0.02,
    )
    market.premium_index_5d = 0.001
    onchain = OnchainData(open_interest=1_000_000.0)
    onchain.lsr_top_count = 1.5
    macro = MacroData()
    macro.spy_btc_corr_30d = 0.3
    return DataSnapshot(
        datetime(2026, 1, 1, tzinfo=UTC),
        "BTC/USDT:USDT",
        market,
        onchain,
        NewsSentiment(),
        macro,
    )


def _context(rows: int = 20):
    snapshot = _snapshot(rows)
    return context(price=100.0, snapshots={"4h": snapshot})


def _features(snapshot, columns, medians):
    return {**dict.fromkeys(columns, 0.0), "_vol5": 0.2}


class FakePredictor:
    def __init__(self, move: float = 0.1, error: Exception | None = None) -> None:
        self.move = move
        self.error = error

    def predict(self, **kwargs):
        if self.error is not None:
            raise self.error
        last_close = float(kwargs["df"]["close"].iloc[-1])
        return pd.DataFrame({"close": np.full(kwargs["pred_len"], last_close * (1.0 + self.move))})


def _gate(probability: float = 0.8, classifier_error: Exception | None = None):
    class Classifier:
        def predict_proba(self, values):
            if classifier_error is not None:
                raise classifier_error
            return np.array([[1.0 - probability, probability]])

    return {
        "feat_cols": ["feature"],
        "medians": {"feature": 0.0},
        "scaler": SimpleNamespace(transform=lambda values: values),
        "classifier": Classifier(),
    }


def _component(*, probability=0.8, predictor=None, gate=None):
    from cryptotrader.signals.components.kronos import KronosComponent

    config = KronosConfig(lookback=20, pred_len=50, ohlcv_limit=20)
    return KronosComponent(
        config,
        gate_loader=lambda _: gate or _gate(probability),
        predictor_loader=lambda _: predictor or FakePredictor(),
        feature_computer=_features,
    )


def test_requirements_declare_gate_training_timeframe_and_auxiliary_data():
    component = _component()

    requirements = component.requirements()

    assert [(item.timeframe, item.limit) for item in requirements.candles] == [("4h", 20)]
    assert requirements.onchain is True
    assert requirements.macro is True
    assert requirements.kronos_aux is True


def test_real_gate_features_are_finite_and_use_medians_for_missing_auxiliary_data():
    from cryptotrader.agents._kronos_features import compute_kronos_features

    snapshot = _snapshot()
    snapshot.onchain.open_interest = 0.0
    del snapshot.onchain.lsr_top_count
    del snapshot.market.premium_index_5d
    del snapshot.macro.spy_btc_corr_30d
    columns = [
        "vol_ratio",
        "trend_30d",
        "bb_pctb",
        "oi_value",
        "lsr_top_count",
        "premium_close_5d",
        "spy_btc_corr",
    ]
    medians = dict.fromkeys(columns, 0.25)

    features = compute_kronos_features(snapshot, columns, medians)

    assert all(math.isfinite(features[column]) for column in columns)
    assert features["oi_value"] == pytest.approx(0.25)
    assert features["lsr_top_count"] == pytest.approx(0.25)
    assert features["premium_close_5d"] == pytest.approx(0.25)
    assert features["spy_btc_corr"] == pytest.approx(0.25)


@pytest.mark.asyncio
async def test_gate_rejection_is_valid_neutral_signal():
    result = await _component(probability=0.49).evaluate(_context())

    assert result.direction == "neutral"
    assert result.confidence == 0.0
    assert result.details["gate_proba"] == pytest.approx(0.49)


@pytest.mark.asyncio
async def test_accepted_prediction_returns_pure_directional_signal():
    result = await _component(predictor=FakePredictor(move=0.1)).evaluate(_context())

    assert result.component_id == "kronos"
    assert result.direction == "long"
    assert 0.0 < result.confidence <= 1.0
    assert result.details["raw_signal"] == pytest.approx(0.1)
    assert "position_scale" not in result.details
    assert "stop_loss" not in result.details
    assert "take_profit" not in result.details


@pytest.mark.asyncio
async def test_step2_weak_short_is_neutral_signal():
    result = await _component(predictor=FakePredictor(move=-0.02)).evaluate(_context())

    assert result.direction == "neutral"
    assert result.confidence == 0.0
    assert result.details["raw_signal"] == pytest.approx(-0.02)


@pytest.mark.asyncio
async def test_insufficient_history_raises_component_error():
    from cryptotrader.signals.component import ComponentExecutionError

    with pytest.raises(ComponentExecutionError, match="input data"):
        await _component().evaluate(_context(rows=19))


@pytest.mark.asyncio
async def test_gate_load_failure_raises_component_error():
    from cryptotrader.signals.component import ComponentExecutionError
    from cryptotrader.signals.components.kronos import KronosComponent

    def fail(_):
        raise FileNotFoundError("gate missing")

    component = KronosComponent(KronosConfig(), gate_loader=fail)

    with pytest.raises(ComponentExecutionError, match="gate loading"):
        await component.evaluate(_context())


@pytest.mark.asyncio
async def test_gate_classification_failure_raises_component_error():
    from cryptotrader.signals.component import ComponentExecutionError

    component = _component(gate=_gate(classifier_error=RuntimeError("classifier down")))

    with pytest.raises(ComponentExecutionError, match="gate classification"):
        await component.evaluate(_context())


@pytest.mark.asyncio
async def test_predictor_failure_raises_component_error():
    from cryptotrader.signals.component import ComponentExecutionError

    component = _component(predictor=FakePredictor(error=RuntimeError("model down")))

    with pytest.raises(ComponentExecutionError, match="prediction"):
        await component.evaluate(_context())


@pytest.mark.asyncio
async def test_predictor_runs_through_to_thread(monkeypatch):
    calls = []

    async def fake_to_thread(function, *args, **kwargs):
        calls.append(function)
        return function(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

    await _component().evaluate(_context())

    assert any(getattr(function, "__name__", "") == "predict" for function in calls)

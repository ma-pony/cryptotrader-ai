"""Unit tests for the kronos_signal LangGraph node and _kronos_features helper.

The teammate's modules (kronos_signal.py / _kronos_features.py) are imported
inside each test so that import failures at collection time are handled
gracefully via pytest.importorskip-style guards.  Tests for compute_kronos_features
(already present) run immediately; tests for kronos_signal are skipped with a
clear message until the node module lands.
"""

from __future__ import annotations

import math
import pickle
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from cryptotrader._compat import UTC
from cryptotrader.models import DataSnapshot, MacroData, MarketData, NewsSentiment, OnchainData

# ── Gate artifact ────────────────────────────────────────────────────────────

_GATE_PATH = Path(__file__).parent.parent / "artifacts" / "kronos" / "gate_v21.pkl"


def _load_gate() -> dict:
    with _GATE_PATH.open("rb") as fh:
        return pickle.load(fh)


_GATE = _load_gate()
_FEAT_COLS: list[str] = _GATE["feat_cols"]
_MEDIANS: dict[str, float] = _GATE["medians"]


# ── Snapshot helpers ─────────────────────────────────────────────────────────


def _make_ohlcv(n: int = 460, trend: float = 10.0, base: float = 50_000.0) -> pd.DataFrame:
    """Return a minimal OHLCV DataFrame with *n* 4h bars."""
    prices = [base + i * trend for i in range(n)]
    return pd.DataFrame(
        {
            "open": prices,
            "high": [p + 20 for p in prices],
            "low": [p - 20 for p in prices],
            "close": prices,
            "volume": [1000.0] * n,
        }
    )


def _make_snapshot(
    *,
    n_bars: int = 460,
    trend: float = 10.0,
    base_price: float = 50_000.0,
    open_interest: float = 5_000_000_000.0,
    lsr_top_count: float | None = 1.5,
    premium_index_5d: float | None = 0.001,
    spy_btc_corr_30d: float | None = 0.30,
    funding_rate: float = 0.01,
    pair: str = "BTC/USDT",
) -> DataSnapshot:
    """Build a synthetic DataSnapshot for unit tests.

    Pass ``None`` for optional fields to simulate missing / unavailable data.
    """
    ohlcv = _make_ohlcv(n_bars, trend, base_price)

    market = MarketData(
        pair=pair,
        ohlcv=ohlcv,
        ticker={"last": ohlcv["close"].iloc[-1], "baseVolume": 1000},
        funding_rate=funding_rate,
        orderbook_imbalance=0.5,
        volatility=0.02,
    )
    if premium_index_5d is not None:
        market.premium_index_5d = premium_index_5d  # type: ignore[attr-defined]

    onchain = OnchainData(open_interest=open_interest)
    if lsr_top_count is not None:
        onchain.lsr_top_count = lsr_top_count  # type: ignore[attr-defined]

    macro = MacroData()
    if spy_btc_corr_30d is not None:
        macro.spy_btc_corr_30d = spy_btc_corr_30d  # type: ignore[attr-defined]

    return DataSnapshot(
        timestamp=datetime.now(UTC),
        pair=pair,
        market=market,
        onchain=onchain,
        news=NewsSentiment(),
        macro=macro,
    )


def _base_state(snapshot: DataSnapshot | None = None, pair: str = "BTC/USDT") -> dict:
    """Build a minimal ArenaState dict matching project conventions."""
    snap = snapshot or _make_snapshot()
    price = float(snap.market.ticker.get("last", 50_000.0))
    return {
        "messages": [],
        "data": {
            "snapshot": snap,
            "snapshot_summary": {
                "pair": pair,
                "price": price,
                "funding_rate": 0.01,
                "volatility": 0.02,
                "orderbook_imbalance": 0.5,
            },
            "experience": "",
        },
        "metadata": {
            "pair": pair,
            "engine": "paper",
            "backtest_mode": True,
            "models": {},
            "analysis_model": "gpt-4o-mini",
            "debate_model": "gpt-4o-mini",
            "verdict_model": "gpt-4o-mini",
        },
        "debate_round": 0,
        "max_debate_rounds": 2,
        "divergence_scores": [],
    }


# ── Helpers to check whether node module is available ───────────────────────


def _require_kronos_signal():
    """Import kronos_signal, skip the test if not yet available."""
    try:
        import cryptotrader.nodes.kronos_signal as ks

        return ks
    except ImportError as exc:
        pytest.skip(f"cryptotrader.nodes.kronos_signal not yet available: {exc}")


# ── Feature computation tests (no node dependency) ──────────────────────────


class TestKronosFeatures:
    """Tests for compute_kronos_features — no node dependency."""

    def test_features_returns_7_keys(self):
        """All 7 gate feature names must be present, all values finite floats."""
        from cryptotrader.agents._kronos_features import compute_kronos_features

        snap = _make_snapshot()
        result = compute_kronos_features(snap, _FEAT_COLS, _MEDIANS)

        for col in _FEAT_COLS:
            assert col in result, f"Missing feature: {col}"
            val = result[col]
            assert isinstance(val, float), f"{col} is not float: {type(val)}"
            assert math.isfinite(val), f"{col} is not finite: {val}"

    def test_features_uses_median_on_missing(self):
        """Missing oi, lsr, and premium fields → features equal gate medians."""
        from cryptotrader.agents._kronos_features import compute_kronos_features

        snap = _make_snapshot(
            open_interest=0.0,  # triggers median fill (<=0 branch)
            lsr_top_count=None,  # attribute absent → 0.0 → median fill
            premium_index_5d=None,
            spy_btc_corr_30d=None,
        )
        # Remove lsr_top_count attribute entirely if it was never set
        if hasattr(snap.onchain, "lsr_top_count"):
            delattr(snap.onchain, "lsr_top_count")

        result = compute_kronos_features(snap, _FEAT_COLS, _MEDIANS)

        # oi_value: open_interest=0 → should equal median
        assert result["oi_value"] == pytest.approx(_MEDIANS["oi_value"])
        # lsr_top_count: attr absent, getattr returns None → 0.0 → median fill
        assert result["lsr_top_count"] == pytest.approx(_MEDIANS["lsr_top_count"])
        assert result["premium_close_5d"] == pytest.approx(_MEDIANS["premium_close_5d"])
        assert result["spy_btc_corr"] == pytest.approx(_MEDIANS["spy_btc_corr"])

    def test_all_features_finite_flat_ohlcv(self):
        """Even with a perfectly flat price series (zero variance), all features finite."""
        from cryptotrader.agents._kronos_features import compute_kronos_features

        snap = _make_snapshot(trend=0.0)  # flat prices
        result = compute_kronos_features(snap, _FEAT_COLS, _MEDIANS)
        for col in _FEAT_COLS:
            assert math.isfinite(result[col]), f"{col} not finite on flat OHLCV"

    def test_features_finite_tiny_ohlcv(self):
        """With only 3 bars (< 20), OHLCV-based features fall back to medians."""
        from cryptotrader.agents._kronos_features import compute_kronos_features

        snap = _make_snapshot(n_bars=3)
        result = compute_kronos_features(snap, _FEAT_COLS, _MEDIANS)
        for col in _FEAT_COLS:
            assert math.isfinite(result[col])


# ── Kronos node tests (skipped if node not yet present) ─────────────────────


# Deterministic fake predictor output (pred_df returned by Kronos predict)
def _make_pred_df(last_close: float, signal_magnitude: float, direction: int = 1) -> pd.DataFrame:
    """Build a fake pred_df with 50 bars matching the node's expected 'close' column.

    The node computes:
        pred_returns = pred_close / last_close - 1.0
        h10_20 = mean(pred_returns[9:20])
        h30_50 = mean(pred_returns[29:50])
        signal  = 0.5 * h10_20 + 0.5 * h30_50

    We set every close to last_close * (1 + direction * signal_magnitude) so
    the resulting signal ≈ direction * signal_magnitude.

    IMPORTANT: pass the *actual* last_close from the snapshot so that the
    pred_returns are computed relative to the correct reference price.
    """
    n = 50
    close_val = last_close * (1.0 + direction * signal_magnitude)
    return pd.DataFrame(
        {
            "close": [close_val] * n,
            "open": [last_close] * n,
            "high": [close_val * 1.005] * n,
            "low": [last_close * 0.995] * n,
            "volume": [1000.0] * n,
        }
    )


def _snap_last_close(snap) -> float:
    """Return the last close price from a snapshot's OHLCV."""
    return float(snap.market.ohlcv["close"].iloc[-1])


def _make_mock_predictor(pred_df: pd.DataFrame) -> MagicMock:
    """Return a MagicMock predictor whose .predict() returns *pred_df*."""
    mock_predictor = MagicMock()
    mock_predictor.predict.return_value = pred_df
    return mock_predictor


def _patch_predictor(monkeypatch, pred_df: pd.DataFrame):
    """Monkeypatch the module-level Kronos predictor cache to return pred_df.

    Kept for backwards compatibility; prefer _make_mock_predictor + setattr.
    """
    mock_predictor = _make_mock_predictor(pred_df)

    try:
        import cryptotrader.nodes.kronos_signal as ks

        if hasattr(ks, "_get_predictor"):
            monkeypatch.setattr(ks, "_get_predictor", lambda *a, **kw: mock_predictor)
        if hasattr(ks, "_predictor_cache"):
            monkeypatch.setattr(ks, "_predictor_cache", {"predictor": mock_predictor})
        try:
            import cryptotrader.nodes.kronos_aux as ka

            if hasattr(ka, "_get_predictor"):
                monkeypatch.setattr(ka, "_get_predictor", lambda *a, **kw: mock_predictor)
        except ImportError:
            pass
    except ImportError:
        pass

    return mock_predictor


def _get_verdict(result: dict) -> dict:
    """Extract verdict dict from a kronos_signal result.

    The node returns verdict as a plain dict (not TradeVerdict instance),
    following the LangGraph partial-state merge convention.
    """
    v = result["data"]["verdict"]
    # Accept both raw dict and TradeVerdict dataclass for forward-compatibility
    if hasattr(v, "action"):
        return {
            "action": v.action,
            "confidence": v.confidence,
            "position_scale": v.position_scale,
            "reasoning": v.reasoning,
            "stop_loss": v.stop_loss,
            "take_profit": v.take_profit,
            "verdict_source": v.verdict_source,
        }
    return v


def _setup_gate_mock(monkeypatch, ks, proba_class1: float = 0.7):
    """Helper to patch gate classifier + scaler on the kronos_signal module."""
    mock_gate = _load_gate().copy()
    mock_clf = MagicMock()
    mock_clf.predict_proba.return_value = np.array([[1.0 - proba_class1, proba_class1]])
    mock_gate["classifier"] = mock_clf
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.zeros((1, 7))
    mock_gate["scaler"] = mock_scaler

    # Patch _gate_cache so _get_gate() returns our mock immediately
    monkeypatch.setattr(ks, "_gate_cache", {})
    return mock_gate


def test_numeric_backtest_timestamps_are_milliseconds():
    ks = _require_kronos_signal()
    start = int(pd.Timestamp("2025-01-01T00:00:00Z").timestamp() * 1000)
    ohlcv = pd.DataFrame({"timestamp": [start, start + 4 * 60 * 60 * 1000]})

    x_ts, y_ts = ks._build_timestamps(ohlcv, lookback=2, pred_len=1)

    assert x_ts.iloc[0] == pd.Timestamp("2025-01-01T00:00:00Z")
    assert y_ts.iloc[0] == pd.Timestamp("2025-01-01T08:00:00Z")


@pytest.mark.asyncio
async def test_gate_reject_returns_hold(monkeypatch):
    """Gate proba<0.5 → hold verdict."""
    ks = _require_kronos_signal()

    snap = _make_snapshot(spy_btc_corr_30d=0.99)
    state = _base_state(snap)

    mock_gate = _setup_gate_mock(monkeypatch, ks, proba_class1=0.3)  # 0.3 < 0.5 → reject

    # Patch _get_gate to return our controlled mock
    monkeypatch.setattr(ks, "_get_gate", lambda cfg: mock_gate)

    result = await ks.kronos_signal(state)
    verdict = _get_verdict(result)
    assert verdict["action"] == "hold", f"Expected hold, got {verdict['action']!r}"


@pytest.mark.asyncio
async def test_gate_accept_returns_directional(monkeypatch):
    """Gate accept (proba≥0.5) with strong long signal → long with confidence>0 and SL/TP set."""
    ks = _require_kronos_signal()

    snap = _make_snapshot()
    state = _base_state(snap)
    last_close = _snap_last_close(snap)

    pred_df = _make_pred_df(last_close, signal_magnitude=0.12, direction=1)

    mock_gate = _setup_gate_mock(monkeypatch, ks, proba_class1=0.75)
    monkeypatch.setattr(ks, "_get_gate", lambda cfg: mock_gate)
    monkeypatch.setattr(ks, "_get_predictor", lambda cfg: _make_mock_predictor(pred_df))

    result = await ks.kronos_signal(state)
    verdict = _get_verdict(result)

    assert verdict["action"] in ("long", "short"), f"Expected directional, got {verdict['action']!r}"
    assert verdict["confidence"] > 0, "Confidence should be positive"
    assert verdict["stop_loss"] is not None, "stop_loss must be set for directional verdict"
    assert verdict["take_profit"] is not None, "take_profit must be set for directional verdict"


@pytest.mark.asyncio
async def test_insufficient_history_holds_before_loading_predictor(monkeypatch):
    ks = _require_kronos_signal()
    state = _base_state(_make_snapshot(n_bars=459))
    mock_gate = _setup_gate_mock(monkeypatch, ks, proba_class1=0.75)
    get_predictor = MagicMock()
    monkeypatch.setattr(ks, "_get_gate", lambda cfg: mock_gate)
    monkeypatch.setattr(ks, "_get_predictor", get_predictor)

    result = await ks.kronos_signal(state)

    verdict = _get_verdict(result)
    assert verdict["action"] == "hold"
    assert "need ≥460" in verdict["reasoning"]
    get_predictor.assert_not_called()


@pytest.mark.asyncio
async def test_exit_prices_use_current_ticker_price(monkeypatch):
    ks = _require_kronos_signal()
    snap = _make_snapshot()
    last_close = _snap_last_close(snap)
    snap.market.ticker["last"] = 60_000.0
    state = _base_state(snap)
    pred_df = _make_pred_df(last_close, signal_magnitude=0.12, direction=1)

    mock_gate = _setup_gate_mock(monkeypatch, ks, proba_class1=0.75)
    monkeypatch.setattr(ks, "_get_gate", lambda cfg: mock_gate)
    monkeypatch.setattr(ks, "_get_predictor", lambda cfg: _make_mock_predictor(pred_df))

    verdict = _get_verdict(await ks.kronos_signal(state))
    implied_entry = (verdict["take_profit"] + 3 * verdict["stop_loss"]) / 4

    assert verdict["action"] == "long"
    assert implied_entry == pytest.approx(60_000.0)


@pytest.mark.asyncio
async def test_step2_weak_short_filtered(monkeypatch):
    """Weak negative signal (|signal|<0.04) → hold; reasoning references Step 2."""
    ks = _require_kronos_signal()

    snap = _make_snapshot()
    state = _base_state(snap)
    last_close = _snap_last_close(snap)

    # |signal| = 0.02 < threshold 0.04 → Step 2 filter blocks the short
    pred_df = _make_pred_df(last_close, signal_magnitude=0.02, direction=-1)

    mock_gate = _setup_gate_mock(monkeypatch, ks, proba_class1=0.75)
    monkeypatch.setattr(ks, "_get_gate", lambda cfg: mock_gate)
    monkeypatch.setattr(ks, "_get_predictor", lambda cfg: _make_mock_predictor(pred_df))

    result = await ks.kronos_signal(state)
    verdict = _get_verdict(result)

    assert verdict["action"] == "hold", f"Expected hold after Step 2 filter, got {verdict['action']!r}"
    reasoning_lower = verdict["reasoning"].lower()
    assert any(kw in reasoning_lower for kw in ("step 2", "step2", "weak", "filter")), (
        f"Reasoning should mention Step 2 filter, got: {verdict['reasoning']!r}"
    )


@pytest.mark.asyncio
async def test_graceful_degradation_missing_gate(monkeypatch, tmp_path):
    """Missing gate artifact → returns hold verdict without raising."""
    ks = _require_kronos_signal()

    snap = _make_snapshot()
    state = _base_state(snap)

    # Force _get_gate to raise FileNotFoundError (simulate missing pkl)
    def _raise_fnf(cfg):
        raise FileNotFoundError("nonexistent gate path")

    monkeypatch.setattr(ks, "_get_gate", _raise_fnf)
    # Clear cache to ensure our patch takes effect
    monkeypatch.setattr(ks, "_gate_cache", {})

    result = await ks.kronos_signal(state)
    verdict = _get_verdict(result)
    assert verdict["action"] == "hold", f"Expected hold on missing gate, got {verdict['action']!r}"


@pytest.mark.asyncio
async def test_position_scale_in_range(monkeypatch):
    """Accepted verdict exposes sizing without pretending to set leverage."""
    ks = _require_kronos_signal()

    snap = _make_snapshot()
    state = _base_state(snap)
    last_close = _snap_last_close(snap)
    pred_df = _make_pred_df(last_close, signal_magnitude=0.12, direction=1)

    mock_gate = _setup_gate_mock(monkeypatch, ks, proba_class1=0.80)
    monkeypatch.setattr(ks, "_get_gate", lambda cfg: mock_gate)
    monkeypatch.setattr(ks, "_get_predictor", lambda cfg: _make_mock_predictor(pred_df))

    result = await ks.kronos_signal(state)
    verdict = _get_verdict(result)

    if verdict["action"] != "hold":
        meta = result["data"].get("kronos_meta", {})
        assert 0.3 <= verdict["position_scale"] <= 1.0
        assert meta["target_position_scale"] == pytest.approx(verdict["position_scale"], abs=1e-4)
        assert "leverage" not in meta


@pytest.mark.asyncio
async def test_verdict_source_is_kronos(monkeypatch):
    """kronos_signal verdicts should carry their real provenance."""
    ks = _require_kronos_signal()

    snap = _make_snapshot()
    state = _base_state(snap)
    last_close = _snap_last_close(snap)
    pred_df = _make_pred_df(last_close, signal_magnitude=0.12, direction=1)

    mock_gate = _setup_gate_mock(monkeypatch, ks, proba_class1=0.80)
    monkeypatch.setattr(ks, "_get_gate", lambda cfg: mock_gate)
    monkeypatch.setattr(ks, "_get_predictor", lambda cfg: _make_mock_predictor(pred_df))

    result = await ks.kronos_signal(state)
    verdict = _get_verdict(result)
    assert verdict["verdict_source"] == "kronos"

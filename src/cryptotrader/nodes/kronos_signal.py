"""Kronos foundation-model signal node for the LangGraph trading pipeline.

Replaces the LLM-debate signal with a validated quantitative signal:
Kronos time-series foundation model + a logistic-regression regime gate
(gate_v21) that decides WHEN Kronos predictions are trustworthy.

OOS validation achieved Sharpe ~2.6 in the sister Kronos project.

Node contract
-------------
Input  : ArenaState — reads ``state["data"]["snapshot"]`` and
         ``state["metadata"].get("kronos_cfg", {})``.
Output : partial state dict ``{"data": {"verdict": TradeVerdict, ...}}``
         that LangGraph merge_dicts deep-merges into the running state.

Graceful degradation
--------------------
Every expensive operation is wrapped in try/except.  Any failure at any
stage returns a hold verdict with a descriptive reasoning string rather
than crashing the graph.
"""

from __future__ import annotations

import logging
import math
import os
import pickle
from typing import Any

import numpy as np
import pandas as pd

from cryptotrader.models import TradeVerdict
from cryptotrader.state import ArenaState

logger = logging.getLogger(__name__)

# ── Module-level caches (load once, reuse across graph invocations) ──────────
_gate_cache: dict[str, Any] = {}  # gate_path → loaded gate dict
_predictor_cache: dict[str, Any] = {}  # cache_key → KronosPredictor instance

# Default configuration values
_DEFAULT_CFG: dict[str, Any] = {
    "gate_path": "artifacts/kronos/gate_v21.pkl",
    "model_name": "NeoQuasar/Kronos-base",
    "tokenizer_name": "NeoQuasar/Kronos-Tokenizer-base",
    "device": None,  # None → auto-detect mps / cpu
    "lookback": 460,
    "pred_len": 50,
    "sample_count": 5,
    "step2_short_threshold": 0.04,
}


def _default_device() -> str:
    """Return 'mps' when Apple Silicon GPU is available, else 'cpu'."""
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _resolve_cfg(raw_cfg: dict) -> dict:
    """Merge caller-supplied cfg over defaults; resolve device auto-detect."""
    cfg = {**_DEFAULT_CFG, **raw_cfg}
    if not cfg["device"]:
        cfg["device"] = _default_device()
    return cfg


def _get_gate(cfg: dict) -> dict:
    """Load and cache the regime gate pkl.  Returns the gate dict.

    Raises on load failure (caller must handle).
    """
    gate_path: str = cfg["gate_path"]
    # Resolve relative paths from the cryptotrader-ai project root
    if not os.path.isabs(gate_path):
        # Walk up from this file's location to find the repo root heuristically
        here = os.path.dirname(os.path.abspath(__file__))
        for _ in range(6):
            candidate = os.path.join(here, gate_path)
            if os.path.exists(candidate):
                gate_path = candidate
                break
            here = os.path.dirname(here)

    if gate_path in _gate_cache:
        return _gate_cache[gate_path]

    logger.info("Loading Kronos regime gate from %s", gate_path)
    with open(gate_path, "rb") as fh:
        gate = pickle.load(fh)
    _gate_cache[gate_path] = gate
    logger.info(
        "Gate loaded: feat_cols=%s, training=%s",
        gate.get("feat_cols"),
        gate.get("training", {}),
    )
    return gate


def _get_predictor(cfg: dict) -> Any:
    """Load and cache the KronosPredictor (expensive — loads model weights).

    Returns the KronosPredictor instance.  Raises on load failure.
    """
    cache_key = f"{cfg['model_name']}::{cfg['tokenizer_name']}::{cfg['device']}"
    if cache_key in _predictor_cache:
        return _predictor_cache[cache_key]

    logger.info(
        "Loading KronosPredictor (model=%s, tokenizer=%s, device=%s)",
        cfg["model_name"],
        cfg["tokenizer_name"],
        cfg["device"],
    )
    # vendor/ lives at the repo root (sibling of src/), not on sys.path under
    # PYTHONPATH=src. Add the repo root so `vendor.kronos_model` is importable.
    import sys
    from pathlib import Path as _Path

    _repo_root = str(_Path(__file__).resolve().parents[3])
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

    from vendor.kronos_model.kronos import Kronos, KronosPredictor, KronosTokenizer

    tokenizer = KronosTokenizer.from_pretrained(cfg["tokenizer_name"])
    tokenizer = tokenizer.to(cfg["device"]).eval()

    model = Kronos.from_pretrained(cfg["model_name"])
    model = model.to(cfg["device"]).eval()

    predictor = KronosPredictor(
        model,
        tokenizer,
        device=cfg["device"],
        max_context=512,
        clip=5,
    )
    _predictor_cache[cache_key] = predictor
    logger.info("KronosPredictor loaded and cached (key=%s)", cache_key)
    return predictor


def _hold_verdict(reasoning: str) -> dict:
    """Return a minimal hold verdict state delta."""
    verdict = TradeVerdict(
        action="hold",
        confidence=0.0,
        position_scale=0.0,
        reasoning=reasoning,
        verdict_source="kronos",
    )
    return {
        "data": {
            "verdict": {
                "action": verdict.action,
                "confidence": verdict.confidence,
                "position_scale": verdict.position_scale,
                "divergence": 0.0,
                "reasoning": verdict.reasoning,
                "thesis": "",
                "stop_loss": None,
                "take_profit": None,
                "invalidation": "",
                "target_price": "",
                "verdict_source": "kronos",
            }
        }
    }


def _build_ohlcv_df(ohlcv: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Extract the last *lookback* bars and rename columns to Kronos convention.

    The live ohlcv DataFrame has columns: timestamp, open, high, low, close, volume.
    KronosPredictor additionally expects an 'amount' column.  We synthesise it
    as close x volume (notional) when missing.
    """
    cols_needed = ["open", "high", "low", "close", "volume"]
    df = ohlcv[cols_needed].copy()
    df = df.dropna(subset=["close"])

    # Keep only the last *lookback* bars
    if len(df) > lookback:
        df = df.iloc[-lookback:]

    # Add 'amount' (notional turnover) if absent — Kronos expects it
    df = df.copy()  # avoid SettingWithCopyWarning
    df["amount"] = df["close"] * df["volume"]

    return df.reset_index(drop=True)


def _build_timestamps(
    ohlcv: pd.DataFrame,
    lookback: int,
    pred_len: int,
) -> tuple[pd.Series, pd.Series]:
    """Build x_timestamp (past) and y_timestamp (future) for KronosPredictor.

    Uses the 'timestamp' column when available; otherwise synthesises
    4h-spaced timestamps from the last bar.
    """
    freq_4h = pd.Timedelta(hours=4)

    if "timestamp" in ohlcv.columns:
        raw_timestamps = ohlcv["timestamp"]
        if pd.api.types.is_numeric_dtype(raw_timestamps):
            ts_col = pd.to_datetime(raw_timestamps, unit="ms", utc=True)
        else:
            ts_col = pd.to_datetime(raw_timestamps, utc=True)
        ts_col = ts_col.dropna()
        if len(ts_col) >= lookback:
            x_ts = ts_col.iloc[-lookback:].reset_index(drop=True)
        else:
            x_ts = ts_col.reset_index(drop=True)
        last_ts = x_ts.iloc[-1]
    else:
        # Synthesise from now
        last_ts = pd.Timestamp.utcnow()
        x_ts = pd.Series([last_ts - freq_4h * (lookback - 1 - i) for i in range(lookback)])

    y_ts = pd.Series([last_ts + freq_4h * (i + 1) for i in range(pred_len)])
    return x_ts, y_ts


async def kronos_signal(state: ArenaState) -> dict:  # noqa: C901
    """LangGraph signal node: Kronos foundation model + gate_v21 regime filter.

    Returns a partial state dict with ``data.verdict`` and ``data.kronos_meta``.
    Always returns a valid dict — never raises.
    """
    # ── 0. Resolve config ────────────────────────────────────────────────
    raw_cfg = (state.get("metadata") or {}).get("kronos_cfg", {})
    cfg = _resolve_cfg(raw_cfg)

    # ── 1. Read snapshot ─────────────────────────────────────────────────
    snapshot = (state.get("data") or {}).get("snapshot")
    if snapshot is None:
        logger.warning("kronos_signal: no snapshot in state — returning hold")
        return _hold_verdict("No snapshot available for Kronos signal")

    # ── 2. Load gate (lazy) ──────────────────────────────────────────────
    try:
        gate = _get_gate(cfg)
    except Exception:
        logger.exception("kronos_signal: failed to load gate_v21")
        return _hold_verdict("Gate v21 load failed — holding until resolved")

    feat_cols: list[str] = gate["feat_cols"]
    scaler = gate["scaler"]
    classifier = gate["classifier"]
    medians: dict[str, float] = gate["medians"]

    # ── 2b. Live-only: fetch cross-asset aux features and inject onto snapshot ──
    # Backtest mode skips this — fetchers return CURRENT values, which would be
    # look-ahead leakage at historical timestamps. Live mode populates the 3
    # fields _kronos_features reads via getattr (else they fall back to medians).
    backtest_mode = bool((state.get("metadata") or {}).get("backtest_mode", False))
    if not backtest_mode:
        try:
            from cryptotrader.data.kronos_aux import fetch_kronos_aux

            sym = cfg.get("aux_symbol", "BTCUSDT")
            aux = await fetch_kronos_aux(sym)
            if aux.get("lsr_top_count") is not None:
                snapshot.onchain.lsr_top_count = aux["lsr_top_count"]
            if aux.get("premium_close_5d") is not None:
                snapshot.market.premium_index_5d = aux["premium_close_5d"]
            if aux.get("spy_btc_corr") is not None:
                snapshot.macro.spy_btc_corr_30d = aux["spy_btc_corr"]
        except Exception:
            logger.warning("kronos_signal: aux fetch failed — using medians", exc_info=True)

    # ── 3. Compute gate features ─────────────────────────────────────────
    try:
        from cryptotrader.agents._kronos_features import compute_kronos_features

        features = compute_kronos_features(snapshot, feat_cols, medians)
    except Exception:
        logger.exception("kronos_signal: feature computation failed")
        return _hold_verdict("Kronos feature computation failed — holding")

    # Pull vol5 side-channel (set by _kronos_features, private key)
    vol5_annual = float(features.pop("_vol5", 0.0))

    # ── 4. Gate: regime filter ───────────────────────────────────────────
    try:
        x_raw = np.array([[features[c] for c in feat_cols]], dtype=float)
        x_scaled = scaler.transform(x_raw)
        proba = float(classifier.predict_proba(x_scaled)[0][1])
    except Exception:
        logger.exception("kronos_signal: gate classify failed")
        return _hold_verdict("Gate classification failed — holding")

    logger.info("Kronos gate proba=%.4f (threshold=0.50)", proba)

    if proba < 0.5:
        return _hold_verdict(
            f"Kronos regime gate rejected (proba={proba:.4f} < 0.50) — market conditions unfavourable for Kronos signal"
        )

    # ── 5. Build input data for predictor ────────────────────────────────
    try:
        ohlcv = snapshot.market.ohlcv
        lookback: int = cfg["lookback"]
        pred_len: int = cfg["pred_len"]

        x_df = _build_ohlcv_df(ohlcv, lookback)
        if len(x_df) < lookback:
            return _hold_verdict(f"Insufficient OHLCV history ({len(x_df)} bars, need ≥{lookback}) — holding")

        x_ts, y_ts = _build_timestamps(ohlcv, len(x_df), pred_len)
        # Trim x_ts to match x_df length
        x_ts = x_ts.iloc[-len(x_df) :].reset_index(drop=True)

    except Exception:
        logger.exception("kronos_signal: OHLCV preparation failed")
        return _hold_verdict("OHLCV preparation for Kronos failed — holding")

    # ── 6. Load predictor (lazy) ─────────────────────────────────────────
    try:
        predictor = _get_predictor(cfg)
    except Exception:
        logger.exception("kronos_signal: failed to load KronosPredictor")
        return _hold_verdict(f"KronosPredictor load failed — holding (gate proba={proba:.4f})")

    # ── 7. Run prediction ────────────────────────────────────────────────
    try:
        pred_df = predictor.predict(
            df=x_df.reset_index(drop=True),
            x_timestamp=x_ts,
            y_timestamp=y_ts,
            pred_len=pred_len,
            T=1.0,
            top_p=0.9,
            top_k=0,
            sample_count=cfg["sample_count"],
            verbose=False,
        )
    except Exception:
        logger.exception("kronos_signal: predictor.predict failed")
        return _hold_verdict(f"Kronos prediction failed — holding (gate proba={proba:.4f})")

    # ── 8. Compute signal ────────────────────────────────────────────────
    try:
        last_close = float(x_df["close"].iloc[-1])
        if last_close <= 0:
            return _hold_verdict("Last close is non-positive — cannot compute signal")

        pred_close = pred_df["close"].values.astype(float)
        pred_returns = pred_close / last_close - 1.0  # length pred_len

        # Horizon blends: h10_20 = mean(returns[9:20]); h30_50 = mean(returns[29:50])
        h10_20 = float(np.mean(pred_returns[9:20])) if len(pred_returns) >= 20 else float(np.mean(pred_returns))
        h30_50 = float(np.mean(pred_returns[29:50])) if len(pred_returns) >= 50 else float(np.mean(pred_returns))
        signal = 0.5 * h10_20 + 0.5 * h30_50

    except Exception:
        logger.exception("kronos_signal: signal computation failed")
        return _hold_verdict("Signal computation from Kronos output failed — holding")

    logger.info(
        "Kronos raw signal=%.6f  h10_20=%.6f  h30_50=%.6f  proba=%.4f",
        signal,
        h10_20,
        h30_50,
        proba,
    )

    # ── 9. Step 2 filter: suppress weak shorts ───────────────────────────
    step2_threshold: float = cfg["step2_short_threshold"]
    if signal < 0 and abs(signal) < step2_threshold:
        return _hold_verdict(
            f"Weak short filtered by Step 2 gate (signal={signal:.6f}, threshold={step2_threshold:.4f})"
        )

    # ── 10. Determine direction ──────────────────────────────────────────
    direction: int = 1 if signal > 0 else -1
    action: str = "long" if direction == 1 else "short"

    # ── 11. 5-dim confidence ─────────────────────────────────────────────
    def _clip01(x: float) -> float:
        return max(0.0, min(1.0, x))

    c_gate = _clip01((proba - 0.5) / 0.4)
    c_sig = _clip01((abs(signal) - 0.005) / 0.045)
    # vol5_annual: lower vol → higher confidence (inverse relationship)
    c_vol = _clip01((1.0 / (vol5_annual + 0.1) - 0.5) / 4.5)
    c_drift = 0.5  # no rolling IC available live — neutral
    c_horizon = 1.0 if (math.copysign(1, h10_20) == math.copysign(1, h30_50)) else 0.0

    confidence = 0.40 * c_gate + 0.20 * c_sig + 0.20 * c_vol + 0.10 * c_drift + 0.10 * c_horizon
    confidence = _clip01(confidence)

    # Kronos controls position size only. Exchange leverage remains the static
    # value configured for the account and is enforced by execution/risk nodes.
    position_scale = 0.3 + 0.7 * confidence
    risk_multiple = 1.5 + 3.5 * confidence

    # ── 12. Stop-loss / take-profit ──────────────────────────────────────
    # Exit policy from the SL/TP optimisation sweep (2026-05-28, see design doc
    # Part 24): the Kronos edge is the FULL 50-bar horizon move (its IC), so an
    # early take-profit cuts winners short and DESTROYS the edge. Tight Kronos
    # dip-based stops also get wicked out by 4h noise (median 4h range ≈1.06% >
    # a 1% stop → death-by-wicks, 0% win in continuous backtest). The sweep
    # winner = WIDE stop + effectively NO early TP (hold to horizon):
    #   SL = -8%/risk multiple (validated confidence-based stop calibration)
    #   TP = far (R:R 3) so it rarely triggers — approximates hold-to-horizon
    #        while still satisfying the downstream R:R≥1.5 gate + OCO needs a TP.
    current_price = float(snapshot.market.ticker.get("last") or 0.0)
    entry = current_price if current_price > 0 else last_close
    stop_ret = 0.08 / risk_multiple
    sl = entry * (1.0 - direction * stop_ret)
    tp = entry * (1.0 + direction * stop_ret * 3.0)  # R:R 3 — far TP, lets winners run
    reasoning = (
        f"Kronos foundation model signal: action={action}, signal={signal:.6f}, "
        f"gate_proba={proba:.4f}, confidence={confidence:.4f}, position_scale={position_scale:.4f}. "
        f"Horizon blend: h10_20={h10_20:.6f}, h30_50={h30_50:.6f}. "
        f"applied: kronos_signal::gate_v21"
    )

    logger.info(
        "Kronos verdict: action=%s conf=%.4f scale=%.4f risk_multiple=%.2f sl=%.4f tp=%.4f",
        action,
        confidence,
        position_scale,
        risk_multiple,
        sl,
        tp,
    )

    verdict_data: dict = {
        "action": action,
        "confidence": confidence,
        "position_scale": position_scale,
        "divergence": 0.0,
        "reasoning": reasoning,
        "thesis": f"Kronos predicts {action} with {abs(signal) * 100:.2f}% expected move",
        "stop_loss": round(sl, 6),
        "take_profit": round(tp, 6),
        "invalidation": "",
        "target_price": str(round(tp, 2)),
        "verdict_source": "kronos",
    }

    kronos_meta: dict = {
        "target_position_scale": round(position_scale, 4),
        "risk_multiple": round(risk_multiple, 3),
        "signal": round(signal, 8),
        "gate_proba": round(proba, 6),
        "h10_20": round(h10_20, 8),
        "h30_50": round(h30_50, 8),
        "c_gate": round(c_gate, 4),
        "c_sig": round(c_sig, 4),
        "c_vol": round(c_vol, 4),
        "c_drift": c_drift,
        "c_horizon": c_horizon,
    }

    return {"data": {"verdict": verdict_data, "kronos_meta": kronos_meta}}

"""Kronos gate-v21 feature computation from a DataSnapshot.

Maps a cryptotrader DataSnapshot to the 7 regime-gate features expected by
gate_v21.pkl.  All failures are swallowed and replaced with the median for
that feature so this helper never raises in production.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from cryptotrader.models import DataSnapshot

logger = logging.getLogger(__name__)

# Annualisation factor for 4h bars: 6 bars/day x 365 days = 2190 bars/year
_BARS_PER_YEAR_4H = 6 * 365  # 2190


def _safe_float(value: Any, fallback: float) -> float:
    """Coerce *value* to float; return *fallback* on any failure."""
    if value is None:
        return fallback
    try:
        v = float(value)
        return v if math.isfinite(v) else fallback
    except (TypeError, ValueError):
        return fallback


def compute_kronos_features(  # noqa: C901
    snapshot: DataSnapshot,
    feat_cols: list[str],
    medians: dict[str, float],
) -> dict[str, float]:
    """Compute the 7 gate_v21 regime features from a live DataSnapshot.

    Parameters
    ----------
    snapshot:
        Live cryptotrader DataSnapshot (market / onchain / macro sub-objects).
    feat_cols:
        Ordered list of feature names from gate_v21.pkl (used to verify all
        7 keys are present in the returned dict).
    medians:
        Per-feature median fill values from gate_v21.pkl.

    Returns
    -------
    dict[str, float]
        Exactly the keys in *feat_cols*, all finite floats.
    """
    result: dict[str, float] = {}

    # ── 1. vol_ratio: annualised vol(5d) / annualised vol(30d) ───────────
    try:
        ohlcv = snapshot.market.ohlcv
        closes = ohlcv["close"].dropna().values.astype(float)
        if len(closes) >= 2:
            log_ret = np.diff(np.log(closes + 1e-12))
            # 5d ≈ 30 4h-bars; 30d ≈ 180 4h-bars
            seg5 = log_ret[-30:] if len(log_ret) >= 30 else log_ret
            seg30 = log_ret[-180:] if len(log_ret) >= 180 else log_ret
            # ddof=1 (sample std) to match Phase A training (pandas .std() default).
            # Using ddof=0 here would systematically shift vol ~1.7% vs the gate's
            # training distribution, biasing the c_vol confidence term.
            vol5 = float(np.std(seg5, ddof=1) * math.sqrt(_BARS_PER_YEAR_4H)) if len(seg5) >= 2 else 0.0
            vol30 = float(np.std(seg30, ddof=1) * math.sqrt(_BARS_PER_YEAR_4H)) if len(seg30) >= 2 else 0.0
            if vol30 > 1e-12 and math.isfinite(vol5) and math.isfinite(vol30):
                vol_ratio = vol5 / vol30
            else:
                vol_ratio = medians["vol_ratio"]
        else:
            vol_ratio = medians["vol_ratio"]
            vol5 = 0.0
    except Exception:
        logger.warning("vol_ratio computation failed", exc_info=True)
        vol_ratio = medians["vol_ratio"]
        vol5 = 0.0

    result["vol_ratio"] = _safe_float(vol_ratio, medians["vol_ratio"])
    # Expose vol5 as a side-channel so kronos_signal.py can use it for the
    # confidence formula without recomputing.
    result["_vol5"] = float(vol5)

    # ── 2. trend_30d: corrcoef(arange, log_close) over last 180 4h-bars ─
    try:
        ohlcv = snapshot.market.ohlcv
        closes = ohlcv["close"].dropna().values.astype(float)
        seg = closes[-180:] if len(closes) >= 180 else closes
        if len(seg) >= 5:
            log_c = np.log(seg + 1e-12)
            idx = np.arange(len(log_c), dtype=float)
            corr = float(np.corrcoef(idx, log_c)[0, 1])
            trend_30d = corr if math.isfinite(corr) else medians["trend_30d"]
        else:
            trend_30d = medians["trend_30d"]
    except Exception:
        logger.warning("trend_30d computation failed", exc_info=True)
        trend_30d = medians["trend_30d"]

    result["trend_30d"] = _safe_float(trend_30d, medians["trend_30d"])

    # ── 3. bb_pctb: Bollinger %B(20, 2) on close ─────────────────────────
    try:
        from ta.volatility import BollingerBands

        ohlcv = snapshot.market.ohlcv
        close_series = ohlcv["close"].dropna()
        if len(close_series) >= 20:
            bb = BollingerBands(close=close_series, window=20, window_dev=2)
            pband = bb.bollinger_pband()
            last_val = float(pband.iloc[-1])
            bb_pctb = last_val if math.isfinite(last_val) else medians["bb_pctb"]
        else:
            bb_pctb = medians["bb_pctb"]
    except Exception:
        logger.warning("bb_pctb computation failed", exc_info=True)
        bb_pctb = medians["bb_pctb"]

    result["bb_pctb"] = _safe_float(bb_pctb, medians["bb_pctb"])

    # ── 4. oi_value: open interest ────────────────────────────────────────
    try:
        oi = getattr(snapshot.onchain, "open_interest", None)
        oi_value = _safe_float(oi, 0.0)
        if oi_value <= 0:
            oi_value = medians["oi_value"]
    except Exception:
        logger.warning("oi_value read failed", exc_info=True)
        oi_value = medians["oi_value"]

    result["oi_value"] = _safe_float(oi_value, medians["oi_value"])

    # ── 5. lsr_top_count: top-trader long/short ratio ─────────────────────
    # Agent-B (data fetcher) may add `lsr_top_count` to OnchainData in a
    # future cycle; read defensively.
    try:
        lsr = getattr(snapshot.onchain, "lsr_top_count", None)
        lsr_top_count = _safe_float(lsr, 0.0)
        if lsr_top_count <= 0:
            lsr_top_count = medians["lsr_top_count"]
    except Exception:
        logger.warning("lsr_top_count read failed", exc_info=True)
        lsr_top_count = medians["lsr_top_count"]

    result["lsr_top_count"] = _safe_float(lsr_top_count, medians["lsr_top_count"])

    # ── 6. premium_close_5d: 5-day avg premium index ─────────────────────
    # Premium and funding are different features. Missing premium data must
    # use the gate's training median rather than a funding-rate proxy.
    try:
        prem = getattr(snapshot.market, "premium_index_5d", None)
        premium_close_5d = _safe_float(prem, medians["premium_close_5d"])
    except Exception:
        logger.warning("premium_close_5d computation failed", exc_info=True)
        premium_close_5d = medians["premium_close_5d"]

    result["premium_close_5d"] = _safe_float(premium_close_5d, medians["premium_close_5d"])

    # ── 7. spy_btc_corr: 30d rolling SPY-BTC correlation ─────────────────
    # Agent-B may populate `spy_btc_corr_30d` on MacroData; no local
    # computation is feasible with the current snapshot shape.
    try:
        corr = getattr(snapshot.macro, "spy_btc_corr_30d", None)
        spy_btc_corr = _safe_float(corr, medians["spy_btc_corr"])
    except Exception:
        logger.warning("spy_btc_corr read failed", exc_info=True)
        spy_btc_corr = medians["spy_btc_corr"]

    result["spy_btc_corr"] = _safe_float(spy_btc_corr, medians["spy_btc_corr"])

    # ── Verify coverage ───────────────────────────────────────────────────
    for col in feat_cols:
        if col not in result:
            logger.warning("Missing gate feature %r — filling with median", col)
            result[col] = medians.get(col, 0.0)

    return result

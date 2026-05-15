"""Tests for nodes/verdict.py — _extract_ohlcv_returns, _merge_returns, _measure_api_latency, _should_downgrade."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from cryptotrader.nodes.verdict import (
    _extract_ohlcv_returns,
    _inject_weighted_sl_tp,
    _measure_api_latency,
    _merge_returns,
    _post_process_verdict,
    _should_downgrade_to_weighted,
)

# ── _merge_returns ──


class TestMergeReturns:
    def test_pm_returns_sufficient(self):
        result = _merge_returns([0.01, 0.02, -0.01] * 10, [0.05], min_count=20)
        assert len(result) == 30
        assert result == [0.01, 0.02, -0.01] * 10

    def test_pm_returns_empty_uses_ohlcv(self):
        result = _merge_returns([], [0.01, 0.02, -0.01], min_count=3)
        assert result == [0.01, 0.02, -0.01]

    def test_pm_returns_partial_supplement(self):
        pm = [0.01, 0.02]
        ohlcv = [0.05, 0.06, 0.07, 0.08]
        result = _merge_returns(pm, ohlcv, min_count=5)
        assert len(result) == 5
        assert result[-2:] == pm

    def test_both_empty(self):
        assert _merge_returns([], []) == []


# ── _extract_ohlcv_returns ──


class TestExtractOhlcvReturns:
    def test_no_snapshot(self):
        state = {"data": {}, "metadata": {}}
        assert _extract_ohlcv_returns(state) == ([], [])

    def test_no_market(self):
        snap = MagicMock(spec=[])
        state = {"data": {"snapshot": snap}, "metadata": {}}
        assert _extract_ohlcv_returns(state) == ([], [])

    def test_ohlcv_none(self):
        snap = MagicMock()
        snap.market.ohlcv = None
        state = {"data": {"snapshot": snap}, "metadata": {}}
        assert _extract_ohlcv_returns(state) == ([], [])

    def test_single_close(self):
        snap = MagicMock()
        snap.market.ohlcv = {"close": pd.Series([100.0])}
        state = {"data": {"snapshot": snap}, "metadata": {}}
        prices, returns = _extract_ohlcv_returns(state)
        assert prices == [100.0]
        assert returns == []

    def test_hourly_aggregation_to_daily(self):
        closes = [100.0 + i for i in range(50)]
        snap = MagicMock()
        snap.market.ohlcv = {"close": pd.Series(closes)}
        state = {"data": {"snapshot": snap}, "metadata": {"timeframe": "1h"}}
        prices, returns = _extract_ohlcv_returns(state)
        assert len(prices) == 50
        assert len(returns) > 0

    def test_daily_timeframe_no_aggregation(self):
        closes = [100.0, 110.0, 105.0]
        snap = MagicMock()
        snap.market.ohlcv = {"close": pd.Series(closes)}
        state = {"data": {"snapshot": snap}, "metadata": {"timeframe": "1d"}}
        prices, returns = _extract_ohlcv_returns(state)
        assert len(prices) == 3
        assert len(returns) == 2
        assert abs(returns[0] - 0.10) < 0.001


# ── _measure_api_latency ──


class TestMeasureApiLatency:
    @pytest.mark.asyncio
    async def test_paper_mode_returns_100(self):
        state = {"metadata": {"engine": "paper"}}
        assert await _measure_api_latency(state) == 100

    @pytest.mark.asyncio
    async def test_default_engine_returns_100(self):
        state = {"metadata": {}}
        assert await _measure_api_latency(state) == 100


# ── _should_downgrade_to_weighted ──


class TestShouldDowngradeToWeighted:
    @pytest.mark.asyncio
    async def test_has_position_returns_false(self):
        state = {
            "data": {"position_context": {"side": "long"}},
            "metadata": {},
        }
        assert await _should_downgrade_to_weighted(state) is False

    @pytest.mark.asyncio
    async def test_flat_no_circuit_breaker(self):
        mock_rsm = MagicMock()
        mock_rsm.is_circuit_breaker_active = MagicMock(return_value=_coro(False))
        state = {
            "data": {"position_context": {"side": "flat"}},
            "metadata": {"redis_url": None},
        }
        with patch("cryptotrader.risk.state.RedisStateManager", return_value=mock_rsm):
            assert await _should_downgrade_to_weighted(state) is True

    @pytest.mark.asyncio
    async def test_flat_circuit_breaker_active(self):
        mock_rsm = MagicMock()
        mock_rsm.is_circuit_breaker_active = MagicMock(return_value=_coro(True))
        state = {
            "data": {"position_context": {"side": "flat"}},
            "metadata": {"redis_url": None},
        }
        with patch("cryptotrader.risk.state.RedisStateManager", return_value=mock_rsm):
            assert await _should_downgrade_to_weighted(state) is False

    @pytest.mark.asyncio
    async def test_no_position_context(self):
        mock_rsm = MagicMock()
        mock_rsm.is_circuit_breaker_active = MagicMock(return_value=_coro(False))
        state = {
            "data": {},
            "metadata": {"redis_url": None},
        }
        with patch("cryptotrader.risk.state.RedisStateManager", return_value=mock_rsm):
            assert await _should_downgrade_to_weighted(state) is True

    @pytest.mark.asyncio
    async def test_redis_exception_returns_false(self):
        state = {
            "data": {"position_context": {"side": "flat"}},
            "metadata": {"redis_url": None},
        }
        with patch("cryptotrader.risk.state.RedisStateManager", side_effect=Exception("no redis")):
            assert await _should_downgrade_to_weighted(state) is False


class TestInjectWeightedSlTp:
    """Verifies weighted-path SL/TP synthesis for directional verdicts.

    Spec 012 vs Phase 1 conflict resolution: high-consensus pairs route to
    `make_verdict_weighted` (no LLM) so they have no `stop_loss`/`take_profit`,
    and Phase 1's `missing_sl_tp` hard-reject would force every such pair to
    hold. `_inject_weighted_sl_tp` computes deterministic SL/TP from
    entry+ATR using the same floors `_post_process_verdict` enforces, so the
    synthetic values are guaranteed to pass the gates and the directional
    intent survives.
    """

    def test_long_uses_atr_floor(self):
        """ATR floor (1.5xATR) wins when entry x 1% is smaller. Stop padded 0.1%."""
        vd = {"action": "long"}
        # entry=100, atr=2 → atr_floor=3, pct_floor=1 → stop_distance=3*1.001=3.003
        _inject_weighted_sl_tp(vd, entry=100.0, atr=2.0)
        assert vd["stop_loss"] == pytest.approx(96.997)
        assert vd["take_profit"] == pytest.approx(106.006)  # entry + 2 * 3.003
        assert "weighted_sl_tp_injected" in vd["guardrails"]

    def test_short_uses_pct_floor_when_atr_smaller(self):
        """1% pct floor wins when 1.5xATR is smaller — ATR-quiet regimes."""
        vd = {"action": "short"}
        # entry=100, atr=0.1 → atr_floor=0.15, pct_floor=1 → stop_distance=1*1.001=1.001
        _inject_weighted_sl_tp(vd, entry=100.0, atr=0.1)
        assert vd["stop_loss"] == pytest.approx(101.001)
        assert vd["take_profit"] == pytest.approx(97.998)

    def test_short_falls_back_to_pct_when_atr_missing(self):
        """ATR None → only pct floor applies."""
        vd = {"action": "short"}
        # entry=50000, atr=None → stop_distance=500*1.001=500.5, target=1001
        _inject_weighted_sl_tp(vd, entry=50000.0, atr=None)
        assert vd["stop_loss"] == pytest.approx(50500.5)
        assert vd["take_profit"] == pytest.approx(48999.0)

    def test_hold_is_noop(self):
        """Hold/close actions don't get SL/TP — they have no position to protect."""
        vd = {"action": "hold"}
        _inject_weighted_sl_tp(vd, entry=100.0, atr=2.0)
        assert vd.get("stop_loss") is None
        assert vd.get("take_profit") is None
        assert "guardrails" not in vd

    def test_existing_sl_tp_not_overridden(self):
        """If LLM already produced SL/TP, don't overwrite them."""
        vd = {"action": "long", "stop_loss": 95.0, "take_profit": 110.0}
        _inject_weighted_sl_tp(vd, entry=100.0, atr=2.0)
        assert vd["stop_loss"] == 95.0
        assert vd["take_profit"] == 110.0
        assert "guardrails" not in vd

    def test_missing_entry_is_noop(self):
        """Cannot compute SL/TP without an entry reference."""
        vd = {"action": "long"}
        _inject_weighted_sl_tp(vd, entry=None, atr=2.0)
        assert vd.get("stop_loss") is None

    def test_zero_entry_is_noop(self):
        vd = {"action": "short"}
        _inject_weighted_sl_tp(vd, entry=0.0, atr=2.0)
        assert vd.get("stop_loss") is None

    def test_injected_values_pass_post_process_gates(self):
        """Round-trip: synthesized SL/TP must satisfy direction + floor + R:R."""
        vd = {"action": "short", "confidence": 0.65, "position_scale": 0.30, "reasoning": "applied: weighted::default"}
        _inject_weighted_sl_tp(vd, entry=0.1135, atr=0.0015)
        # Now run the same gate that previously rejected the weighted path.
        _post_process_verdict(verdict=MagicMock(), raw_analyses={}, vd_dict=vd, entry_price=0.1135, atr=0.0015)
        # Phase 1 reject would have flipped action to hold — verify it stayed short.
        assert vd["action"] == "short"
        # And the R:R was recorded by the gate (proves we passed the floor).
        assert vd.get("risk_reward_ratio") == pytest.approx(2.0)

    @pytest.mark.parametrize(
        ("entry", "atr"),
        [
            # Cases observed in production where the unpadded version rejected
            # itself via stop_too_tight because abs(sl-entry) re-derived in
            # _post_process_verdict landed 1 ULP below the floor.
            (2256.30, 13.157),  # ETH 2026-05-14 20:44 — pct_floor wins (22.563)
            (2290.79, 17.177),  # ETH 2026-05-14 04:44 — atr_floor wins (25.7665)
            (10.30, 0.0908574),  # LINK 2026-05-14 21:44 — atr_floor wins (0.136286)
            # Stress points where pct_floor and atr_floor coincide.
            (100.0, 0.6666667),  # both floors ≈ 1.0
            (50000.0, 333.3333),  # both floors ≈ 500
        ],
    )
    def test_round_trip_passes_gate_at_boundary(self, entry: float, atr: float):
        """Regression: floats that previously hit stop_too_tight on identity floor.

        Without the 0.1% pad: ``abs(entry + stop_distance - entry)`` rounds
        1 ULP below ``max(atr_floor, pct_floor)`` for many entry/atr pairs,
        causing the gate to force-hold a verdict that should have passed.
        """
        for action in ("long", "short"):
            vd = {
                "action": action,
                "confidence": 0.65,
                "position_scale": 0.30,
                "reasoning": "applied: weighted::default",
            }
            _inject_weighted_sl_tp(vd, entry=entry, atr=atr)
            _post_process_verdict(
                verdict=MagicMock(),
                raw_analyses={},
                vd_dict=vd,
                entry_price=entry,
                atr=atr,
            )
            assert vd["action"] == action, (
                f"{action} entry={entry} atr={atr}: rejected by {vd.get('guardrails')} when it should pass the floor"
            )
            assert vd.get("risk_reward_ratio") is not None


async def _coro(val):
    return val

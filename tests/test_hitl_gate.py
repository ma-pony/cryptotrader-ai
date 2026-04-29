"""Tests for HITL gate node — trigger conditions, passthrough, routing."""

from __future__ import annotations

import pytest

from cryptotrader.config import HitlConfig
from cryptotrader.hitl.gate import _should_trigger, hitl_router


def _state(
    *,
    action: str = "long",
    position_scale: float = 0.3,
    divergence_scores: list[float] | None = None,
    backtest_mode: bool = False,
) -> dict:
    return {
        "data": {"verdict": {"action": action, "position_scale": position_scale}},
        "metadata": {"backtest_mode": backtest_mode},
        "divergence_scores": divergence_scores or [],
    }


def test_passthrough_disabled():
    cfg = HitlConfig(enabled=False)
    should, reason = _should_trigger(_state(), cfg)
    assert should is False
    assert reason == ""


def test_passthrough_backtest():
    cfg = HitlConfig(enabled=True)
    should, _reason = _should_trigger(_state(backtest_mode=True), cfg)
    assert should is False


def test_passthrough_hold_action():
    cfg = HitlConfig(enabled=True)
    should, _reason = _should_trigger(_state(action="hold"), cfg)
    assert should is False


def test_trigger_position_scale():
    cfg = HitlConfig(enabled=True, min_position_scale=0.5)
    should, reason = _should_trigger(_state(position_scale=0.75), cfg)
    assert should is True
    assert reason == "position_scale"


def test_no_trigger_below_threshold():
    cfg = HitlConfig(enabled=True, min_position_scale=0.5)
    should, _reason = _should_trigger(_state(position_scale=0.3), cfg)
    assert should is False


def test_trigger_divergence():
    cfg = HitlConfig(enabled=True, divergence_threshold=0.6)
    should, reason = _should_trigger(_state(divergence_scores=[0.3, 0.65]), cfg)
    assert should is True
    assert reason == "divergence"


def test_no_trigger_low_divergence():
    cfg = HitlConfig(enabled=True, divergence_threshold=0.6)
    should, _reason = _should_trigger(_state(divergence_scores=[0.3, 0.4]), cfg)
    assert should is False


def test_hitl_router_pass():
    assert hitl_router({"hitl": {"decision": "approve"}}) == "pass"


def test_hitl_router_empty_decision():
    assert hitl_router({"hitl": {"decision": ""}}) == "pass"


def test_hitl_router_rejected():
    assert hitl_router({"hitl": {"decision": "reject"}}) == "rejected"


def test_hitl_router_expired():
    assert hitl_router({"hitl": {"decision": "expired"}}) == "rejected"


def test_hitl_router_skipped():
    assert hitl_router({"hitl": {"decision": "approve", "skipped": True}}) == "pass"


def test_hitl_router_no_hitl_key():
    assert hitl_router({}) == "pass"


@pytest.mark.asyncio
async def test_hitl_gate_disabled():
    """hitl_gate node transparently passes through when disabled."""
    from unittest.mock import patch

    from cryptotrader.hitl.gate import hitl_gate

    state = _state()
    with patch("cryptotrader.hitl.gate.load_config") as mock_cfg:
        mock_cfg.return_value.hitl = HitlConfig(enabled=False)
        result = await hitl_gate(state)

    assert result["hitl"]["skipped"] is True
    assert result["hitl"]["decision"] == "approve"


@pytest.mark.asyncio
async def test_hitl_gate_backtest_mode():
    from unittest.mock import patch

    from cryptotrader.hitl.gate import hitl_gate

    state = _state(backtest_mode=True, position_scale=0.9)
    with patch("cryptotrader.hitl.gate.load_config") as mock_cfg:
        mock_cfg.return_value.hitl = HitlConfig(enabled=True, min_position_scale=0.5)
        result = await hitl_gate(state)

    assert result["hitl"]["skipped"] is True

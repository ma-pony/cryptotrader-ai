"""Server-side verdict guardrails (``_post_process_verdict``).

Covers:
  Guardrail 1: Confidence-based sizing cap
  Guardrail 2: Missing FR-026 ``applied:`` citation halves confidence
  Guardrail 3: Silent/mock agent caps confidence at raw − 0.20
  SL/TP hard-reject gate (replaces former soft-penalty guardrails 4 & 5):
    - missing stop_loss / take_profit → action forced to hold
    - direction inverted → hold
    - stop too tight (< max(1.5×ATR, 1.0%)) → hold
    - R:R < 1.5 → hold
"""

from __future__ import annotations

import pytest

from cryptotrader.nodes.verdict import _post_process_verdict

# ── Guardrail 1: confidence ramp ──


def test_g1_high_confidence_no_cap():
    vd = {
        "action": "long",
        "confidence": 0.80,
        "position_scale": 0.50,
        "reasoning": "applied: tech::ok",
        "stop_loss": 78000.0,
        "take_profit": 84000.0,
    }
    _post_process_verdict(None, {}, vd, entry_price=80000.0, atr=200.0)
    assert vd["position_scale"] == 0.50


def test_g1_overcapped_scale_pulled_down():
    vd = {
        "action": "short",
        "confidence": 0.60,
        "position_scale": 0.80,
        "reasoning": "applied: tech::ok",
        "stop_loss": 82000.0,
        "take_profit": 76000.0,
    }
    _post_process_verdict(None, {}, vd, entry_price=80000.0, atr=200.0)
    assert vd["position_scale"] == pytest.approx(0.20)
    assert "confidence_scale_cap" in vd.get("guardrails", [])


def test_g1_skips_hold():
    vd = {"action": "hold", "confidence": 0.50, "position_scale": 1.0, "reasoning": ""}
    _post_process_verdict(None, {}, vd)
    assert vd["position_scale"] == 1.0


# ── Guardrail 2: missing applied: ──


def test_g2_missing_applied_halves_cf_for_short():
    vd = {
        "action": "short",
        "confidence": 0.80,
        "position_scale": 0.30,
        "reasoning": "no citation here",
        "stop_loss": 82000.0,
        "take_profit": 76000.0,
    }
    _post_process_verdict(None, {}, vd, entry_price=80000.0, atr=200.0)
    assert vd["confidence"] == pytest.approx(0.40)
    assert "missing_applied" in vd.get("guardrails", [])


def test_g2_present_applied_passes():
    vd = {
        "action": "long",
        "confidence": 0.80,
        "position_scale": 0.30,
        "reasoning": "applied: tech::breakout_long",
        "stop_loss": 78000.0,
        "take_profit": 84000.0,
    }
    _post_process_verdict(None, {}, vd, entry_price=80000.0, atr=200.0)
    assert vd["confidence"] == pytest.approx(0.80)


def test_g2_skips_hold():
    vd = {"action": "hold", "confidence": 0.50, "position_scale": 0.0, "reasoning": "no applied"}
    _post_process_verdict(None, {}, vd)
    assert vd["confidence"] == 0.50


# ── Guardrail 3: silent/mock agents ──


def test_g3_silent_agent_caps_cf():
    raw = {"chain_agent": {"is_mock": False, "confidence": 0.0}}
    vd = {
        "action": "short",
        "confidence": 0.80,
        "position_scale": 0.30,
        "reasoning": "applied: tech::ok",
        "stop_loss": 82000.0,
        "take_profit": 76000.0,
    }
    _post_process_verdict(None, raw, vd, entry_price=80000.0, atr=200.0)
    assert vd["confidence"] == pytest.approx(0.60)


def test_g3_no_silent_agents_no_change():
    raw = {"tech": {"confidence": 0.7, "is_mock": False}, "chain": {"confidence": 0.6, "is_mock": False}}
    vd = {
        "action": "short",
        "confidence": 0.80,
        "position_scale": 0.30,
        "reasoning": "applied: tech::ok",
        "stop_loss": 82000.0,
        "take_profit": 76000.0,
    }
    _post_process_verdict(None, raw, vd, entry_price=80000.0, atr=200.0)
    assert vd["confidence"] == 0.80


# ── SL/TP hard-reject gate ──


def test_sl_missing_forces_hold():
    """No stop_loss on a long → action forced to hold, cf=0, scale=0."""
    vd = {
        "action": "long",
        "confidence": 0.80,
        "position_scale": 0.30,
        "reasoning": "applied: tech::ok",
        "take_profit": 84000.0,
    }  # stop_loss missing
    _post_process_verdict(None, {}, vd, entry_price=80000.0, atr=200.0)
    assert vd["action"] == "hold"
    assert vd["confidence"] == 0.0
    assert vd["position_scale"] == 0.0
    assert "rejected:missing_sl_tp" in vd.get("guardrails", [])


def test_tp_missing_forces_hold():
    vd = {
        "action": "short",
        "confidence": 0.80,
        "position_scale": 0.30,
        "reasoning": "applied: tech::ok",
        "stop_loss": 82000.0,
    }  # take_profit missing
    _post_process_verdict(None, {}, vd, entry_price=80000.0, atr=200.0)
    assert vd["action"] == "hold"


def test_non_numeric_sl_forces_hold():
    vd = {
        "action": "long",
        "confidence": 0.80,
        "position_scale": 0.30,
        "reasoning": "applied: tech::ok",
        "stop_loss": "abc",
        "take_profit": 84000.0,
    }
    _post_process_verdict(None, {}, vd, entry_price=80000.0, atr=200.0)
    assert vd["action"] == "hold"


def test_long_inverted_direction_forces_hold():
    """long with stop_loss > entry → inverted → hold."""
    vd = {
        "action": "long",
        "confidence": 0.80,
        "position_scale": 0.30,
        "reasoning": "applied: tech::ok",
        "stop_loss": 81000.0,
        "take_profit": 84000.0,
    }  # stop above entry
    _post_process_verdict(None, {}, vd, entry_price=80000.0, atr=200.0)
    assert vd["action"] == "hold"
    assert "rejected:direction_inverted_long" in vd.get("guardrails", [])


def test_short_inverted_direction_forces_hold():
    """short with stop_loss < entry → inverted → hold."""
    vd = {
        "action": "short",
        "confidence": 0.80,
        "position_scale": 0.30,
        "reasoning": "applied: tech::ok",
        "stop_loss": 79000.0,
        "take_profit": 76000.0,
    }  # stop below entry
    _post_process_verdict(None, {}, vd, entry_price=80000.0, atr=200.0)
    assert vd["action"] == "hold"
    assert "rejected:direction_inverted_short" in vd.get("guardrails", [])


def test_stop_too_tight_forces_hold():
    """short with stop $100 from entry $80,000 → 0.125% < 1% floor → hold."""
    vd = {
        "action": "short",
        "confidence": 0.80,
        "position_scale": 0.30,
        "reasoning": "applied: tech::ok",
        "stop_loss": 80100.0,
        "take_profit": 78000.0,
    }
    _post_process_verdict(None, {}, vd, entry_price=80000.0, atr=200.0)
    assert vd["action"] == "hold"
    assert "rejected:stop_too_tight" in vd.get("guardrails", [])


def test_atr_floor_dominates():
    """SOL entry $88 ATR $1.5 → 1.5×ATR=$2.25 floor; stop at $89.50 ($1.50) → hold."""
    vd = {
        "action": "short",
        "confidence": 0.80,
        "position_scale": 0.30,
        "reasoning": "applied: tech::ok",
        "stop_loss": 89.50,
        "take_profit": 84.0,
    }
    _post_process_verdict(None, {}, vd, entry_price=88.0, atr=1.5)
    assert vd["action"] == "hold"
    assert "rejected:stop_too_tight" in vd.get("guardrails", [])


def test_low_rr_forces_hold():
    """short entry $80k stop $82k ($2k risk), target $79k ($1k reward) → R:R 0.5 → hold."""
    vd = {
        "action": "short",
        "confidence": 0.80,
        "position_scale": 0.30,
        "reasoning": "applied: tech::ok",
        "stop_loss": 82000.0,
        "take_profit": 79000.0,
    }
    _post_process_verdict(None, {}, vd, entry_price=80000.0, atr=200.0)
    assert vd["action"] == "hold"
    assert any("rejected:low_rr" in g for g in vd.get("guardrails", []))


def test_valid_sl_tp_passes_and_records_rr():
    """Stop $2k risk, target $4k reward → R:R 2.0 → pass + record."""
    vd = {
        "action": "short",
        "confidence": 0.80,
        "position_scale": 0.30,
        "reasoning": "applied: tech::ok",
        "stop_loss": 82000.0,
        "take_profit": 76000.0,
    }
    _post_process_verdict(None, {}, vd, entry_price=80000.0, atr=200.0)
    assert vd["action"] == "short"
    assert vd["confidence"] == 0.80
    assert vd.get("risk_reward_ratio") == pytest.approx(2.0)


def test_no_entry_skips_sl_tp_gate():
    """Without entry_price the SL/TP gate is skipped (rare path)."""
    vd = {
        "action": "short",
        "confidence": 0.80,
        "position_scale": 0.30,
        "reasoning": "applied: tech::ok",
        "stop_loss": 82000.0,
        "take_profit": 76000.0,
    }
    _post_process_verdict(None, {}, vd, entry_price=None, atr=200.0)
    assert vd["action"] == "short"  # not rejected; nothing to validate against


def test_hold_skips_sl_tp_gate():
    """hold doesn't need SL/TP."""
    vd = {"action": "hold", "confidence": 0.50, "position_scale": 0.0, "reasoning": ""}
    _post_process_verdict(None, {}, vd, entry_price=80000.0, atr=200.0)
    assert vd["action"] == "hold"


def test_close_skips_sl_tp_gate():
    """close (full exit) doesn't need SL/TP either."""
    vd = {"action": "close", "confidence": 0.80, "position_scale": 0.0, "reasoning": "applied: tech::exit"}
    _post_process_verdict(None, {}, vd, entry_price=80000.0, atr=200.0)
    assert vd["action"] == "close"

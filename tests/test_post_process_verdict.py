"""Server-side verdict guardrails (``_post_process_verdict``).

Covers all five guardrails:
  1. Confidence-based sizing cap
  2. Missing FR-026 ``applied:`` citation halves confidence
  3. Silent/mock agent caps confidence at raw − 0.20
  4. (N2) Invalidation stop too tight relative to ATR / 1% halves confidence
  5. (N7) R:R < 1.5 (or missing target_price) halves confidence

The tests use a synthetic verdict dict + raw_analyses dict; the LLM
TradeVerdict object is unused inside the function but accepted for symmetry
with the call site.
"""

from __future__ import annotations

import pytest

from cryptotrader.nodes.verdict import _extract_price, _post_process_verdict

# ── _extract_price ──


class TestExtractPrice:
    def test_dollar_sign_with_comma(self):
        assert _extract_price("Cover at $80,950") == 80950.0

    def test_dollar_sign_without_comma(self):
        assert _extract_price("$2304") == 2304.0

    def test_decimal(self):
        assert _extract_price("$87.50 if it breaks") == 87.5

    def test_no_dollar_sign(self):
        assert _extract_price("Stop 80950 above") == 80950.0

    def test_picks_first_price(self):
        assert _extract_price("entry $80,000 stop $79,500") == 80000.0

    def test_empty(self):
        assert _extract_price("") is None
        assert _extract_price(None) is None
        assert _extract_price("if momentum reverses") is None


# ── Guardrail 1: confidence ramp ──


def test_g1_high_confidence_no_cap():
    """cf=0.80 → ramp 0.60; LLM scale 0.50 ≤ ramp ⇒ no change."""
    vd = {"action": "long", "confidence": 0.80, "position_scale": 0.50, "reasoning": "applied: tech::ok"}
    _post_process_verdict(None, {}, vd)
    assert vd["position_scale"] == 0.50


def test_g1_overcapped_scale_pulled_down():
    """cf=0.60 → ramp 0.20; LLM scale 0.80 → capped to 0.20."""
    vd = {"action": "short", "confidence": 0.60, "position_scale": 0.80, "reasoning": "applied: tech::ok"}
    _post_process_verdict(None, {}, vd)
    assert vd["position_scale"] == pytest.approx(0.20)
    assert "confidence_scale_cap" in vd.get("guardrails", [])


def test_g1_skips_hold():
    """hold action: ramp not applied."""
    vd = {"action": "hold", "confidence": 0.50, "position_scale": 1.0, "reasoning": ""}
    _post_process_verdict(None, {}, vd)
    assert vd["position_scale"] == 1.0


# ── Guardrail 2: missing applied: ──


def test_g2_missing_applied_halves_cf_for_short():
    vd = {"action": "short", "confidence": 0.80, "position_scale": 0.30, "reasoning": "no citation here"}
    _post_process_verdict(None, {}, vd)
    assert vd["confidence"] == pytest.approx(0.40)
    assert "missing_applied" in vd.get("guardrails", [])


def test_g2_present_applied_passes():
    vd = {"action": "long", "confidence": 0.80, "position_scale": 0.30, "reasoning": "applied: tech::breakout_long"}
    _post_process_verdict(None, {}, vd)
    assert vd["confidence"] == pytest.approx(0.80)


def test_g2_skips_hold():
    vd = {"action": "hold", "confidence": 0.50, "position_scale": 0.0, "reasoning": "no applied"}
    _post_process_verdict(None, {}, vd)
    assert vd["confidence"] == 0.50  # untouched


# ── Guardrail 3: silent/mock agents ──


def test_g3_silent_agent_caps_cf():
    raw = {"chain_agent": {"is_mock": False, "confidence": 0.0}}  # silent
    vd = {"action": "short", "confidence": 0.80, "position_scale": 0.30, "reasoning": "applied: tech::ok"}
    _post_process_verdict(None, raw, vd)
    assert vd["confidence"] == pytest.approx(0.60)


def test_g3_no_silent_agents_no_change():
    raw = {"tech": {"confidence": 0.7, "is_mock": False}, "chain": {"confidence": 0.6, "is_mock": False}}
    vd = {"action": "short", "confidence": 0.80, "position_scale": 0.30, "reasoning": "applied: tech::ok"}
    _post_process_verdict(None, raw, vd)
    assert vd["confidence"] == 0.80


# ── Guardrail 4 (N2): stop too tight ──


def test_g4_tight_stop_halves_cf():
    """BTC entry $80,000 stop $80,100 = 0.125% < 1% floor → cf halved."""
    vd = {
        "action": "short",
        "confidence": 0.80,
        "position_scale": 0.30,
        "reasoning": "applied: tech::ok",
        "invalidation": "Cover at $80,100 if BTC reclaims SMA60",
        "target_price": "$78,000",  # avoid R:R penalty
    }
    _post_process_verdict(None, {}, vd, entry_price=80000.0, atr=200.0)
    assert vd["confidence"] == pytest.approx(0.40)
    assert "stop_too_tight" in vd.get("guardrails", [])


def test_g4_atr_floor_dominates():
    """ATR-based floor is wider than 1%: stop must satisfy 1.5×ATR."""
    # SOL entry $88, ATR=$1.5 → 1.5×ATR=$2.25, 1% of price=$0.88 → floor $2.25
    # Stop at $89.50 = $1.50 distance < $2.25 → reject
    vd = {
        "action": "short",
        "confidence": 0.80,
        "position_scale": 0.30,
        "reasoning": "applied: tech::ok",
        "invalidation": "Cover at $89.50",
        "target_price": "$84.00",
    }
    _post_process_verdict(None, {}, vd, entry_price=88.0, atr=1.5)
    assert vd["confidence"] == pytest.approx(0.40)


def test_g4_acceptable_stop_passes():
    """Stop $82,000 from entry $80,000 = 2.5% > 1.5×ATR (1.5×200=$300) → OK."""
    vd = {
        "action": "short",
        "confidence": 0.80,
        "position_scale": 0.30,
        "reasoning": "applied: tech::ok",
        "invalidation": "Cover at $82,000",
        "target_price": "$76,000",
    }
    _post_process_verdict(None, {}, vd, entry_price=80000.0, atr=200.0)
    assert vd["confidence"] == 0.80


def test_g4_no_entry_skips_check():
    """Without entry_price the check can't run; skip rather than false-fail."""
    vd = {
        "action": "short",
        "confidence": 0.80,
        "position_scale": 0.30,
        "reasoning": "applied: tech::ok",
        "invalidation": "Cover at $80,100",
        "target_price": "$78,000",
    }
    _post_process_verdict(None, {}, vd, entry_price=None, atr=200.0)
    assert vd["confidence"] == 0.80


# ── Guardrail 5 (N7): R:R < 1.5 ──


def test_g5_low_rr_halves_cf():
    """Entry $80k, stop $82k ($2k risk), target $81k ($1k reward) → R:R 0.5 → reject."""
    vd = {
        "action": "short",
        "confidence": 0.80,
        "position_scale": 0.30,
        "reasoning": "applied: tech::ok",
        "invalidation": "Cover at $82,000",
        "target_price": "$81,000",
    }
    _post_process_verdict(None, {}, vd, entry_price=80000.0, atr=200.0)
    assert vd["confidence"] == pytest.approx(0.40)
    assert any("low_rr" in g for g in vd.get("guardrails", []))


def test_g5_acceptable_rr_passes_and_records_value():
    """Entry $80k, stop $82k ($2k risk), target $76k ($4k reward) → R:R 2.0 → OK."""
    vd = {
        "action": "short",
        "confidence": 0.80,
        "position_scale": 0.30,
        "reasoning": "applied: tech::ok",
        "invalidation": "Cover at $82,000",
        "target_price": "$76,000",
    }
    _post_process_verdict(None, {}, vd, entry_price=80000.0, atr=200.0)
    assert vd["confidence"] == 0.80
    assert vd.get("risk_reward_ratio") == pytest.approx(2.0)


def test_g5_missing_target_halves_cf():
    """Long/short verdict without target_price gets penalized."""
    vd = {
        "action": "short",
        "confidence": 0.80,
        "position_scale": 0.30,
        "reasoning": "applied: tech::ok",
        "invalidation": "Cover at $82,000",
        "target_price": "",  # empty
    }
    _post_process_verdict(None, {}, vd, entry_price=80000.0, atr=200.0)
    assert vd["confidence"] == pytest.approx(0.40)
    assert "missing_target_price" in vd.get("guardrails", [])


def test_g5_skips_hold():
    """hold doesn't need a target."""
    vd = {
        "action": "hold",
        "confidence": 0.50,
        "position_scale": 0.0,
        "reasoning": "",
        "invalidation": "",
        "target_price": "",
    }
    _post_process_verdict(None, {}, vd, entry_price=80000.0, atr=200.0)
    assert vd["confidence"] == 0.50


# ── Stacked guardrails ──


def test_multiple_guardrails_compound():
    """Tight stop + low R:R + missing applied → cf halved repeatedly: 0.80 → 0.40 → 0.20 → 0.10."""
    vd = {
        "action": "short",
        "confidence": 0.80,
        "position_scale": 0.30,
        "reasoning": "no citation",  # G2 fires
        "invalidation": "Cover at $80,100",  # G4 fires (0.125% stop)
        "target_price": "$80,050",  # G5 fires (R:R 0.5)
    }
    _post_process_verdict(None, {}, vd, entry_price=80000.0, atr=200.0)
    # 0.80 → G2 halve → 0.40 → G4 halve → 0.20 → G5 halve → 0.10
    assert vd["confidence"] == pytest.approx(0.10)
    guards = vd.get("guardrails", [])
    assert "missing_applied" in guards
    assert "stop_too_tight" in guards
    assert any("low_rr" in g for g in guards)

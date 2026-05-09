"""spec 018 IVE unit tests — tests/test_ive.py

SC-Z8: >= 8 use cases PASS.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

if TYPE_CHECKING:
    import pytest

from cryptotrader.agents.skills.schema import CaseRecord
from cryptotrader.learning.evolution.ive import (
    FailureClassification,
    _noise_result,
    classify_case,
)


def _make_case(
    cycle_id: str = "test_cycle",
    pair: str = "BTC/USDT",
    verdict_action: str = "short",
    final_pnl: float | None = -100.0,
    agent_analyses: dict | None = None,
    trade_execution: dict | None = None,
) -> CaseRecord:
    return CaseRecord(
        cycle_id=cycle_id,
        timestamp=datetime.now(UTC),
        pair=pair,
        verdict_action=verdict_action,
        final_pnl=final_pnl,
        agent_analyses=agent_analyses or {"tech": "bearish analysis"},
        trade_execution=trade_execution,
    )


def _make_llm(failure_type: str, confidence: float = 0.8) -> MagicMock:
    """Create a mock LLM callable returning a specific failure type."""
    response = json.dumps(
        {
            "diagnostic_answers": ["yes", "no", "yes", "no", "uncertain"],
            "reasoning": f"Test reasoning for {failure_type}",
            "failure_type": failure_type,
            "confidence": confidence,
        }
    )
    return MagicMock(return_value=response)


# ── basic classification tests ────────────────────────────────────────────────


def test_classify_implementation():
    """T018(a): mock LLM returns implementation -> FailureClassification."""
    case = _make_case()
    mock_llm = _make_llm("implementation")

    result = classify_case(case, llm_callable=mock_llm)

    assert result.failure_type == "implementation"
    assert result.case_id == "test_cycle"
    assert result.confidence > 0
    assert len(result.diagnostic_answers) == 5


def test_classify_fundamental():
    """T018(b): mock LLM returns fundamental."""
    case = _make_case()
    mock_llm = _make_llm("fundamental")

    result = classify_case(case, llm_callable=mock_llm)

    assert result.failure_type == "fundamental"


def test_classify_noise():
    """T018(c): mock LLM returns noise."""
    case = _make_case()
    mock_llm = _make_llm("noise")

    result = classify_case(case, llm_callable=mock_llm)

    assert result.failure_type == "noise"


def test_llm_failure_returns_noise(caplog: pytest.LogCaptureFixture):
    """T018(d): LLM call raises exception -> returns noise + warning log."""

    def _failing_llm(_sys: str, _usr: str) -> str:
        raise RuntimeError("LLM unavailable")

    case = _make_case()

    with caplog.at_level("WARNING"):
        result = classify_case(case, llm_callable=_failing_llm)

    assert result.failure_type == "noise"
    assert result.case_id == "test_cycle"
    assert any("LLM" in r.message or "noise" in r.message for r in caplog.records)


def test_invalid_json_retries_then_noise(caplog: pytest.LogCaptureFixture):
    """T018(e): LLM returns non-JSON -> retry once -> still invalid -> noise."""
    call_count = 0

    def _bad_json_llm(_sys: str, _usr: str) -> str:
        nonlocal call_count
        call_count += 1
        return "this is not json at all"

    case = _make_case()

    with caplog.at_level("WARNING"):
        result = classify_case(case, llm_callable=_bad_json_llm)

    assert result.failure_type == "noise"
    assert call_count == 2  # called twice (initial + retry)


def test_prompt_contains_case_context():
    """T018(f): prompt includes trade_execution data and 5 diagnostic questions."""
    captured_prompts: list[str] = []

    def _capturing_llm(system_prompt: str, user_prompt: str) -> str:
        captured_prompts.append(user_prompt)
        return json.dumps(
            {
                "diagnostic_answers": ["yes"] * 5,
                "reasoning": "test",
                "failure_type": "noise",
                "confidence": 0.5,
            }
        )

    case = _make_case(
        trade_execution={
            "entry_price": 88.45,
            "stop_loss": 90.0,
            "hit_sl": True,
            "fill_status": "stopped_out",
        }
    )
    classify_case(case, llm_callable=_capturing_llm)

    assert len(captured_prompts) == 1
    prompt = captured_prompts[0]
    # Check that trade execution data is included
    assert "88.45" in prompt or "entry_price" in prompt
    # Check 5 diagnostic questions present
    assert "1." in prompt
    assert "5." in prompt


def test_same_regime_context_in_prompt():
    """T018(g): same-regime cases appear in prompt context."""
    captured: list[str] = []

    def _cap_llm(sys_p: str, usr_p: str) -> str:
        captured.append(usr_p)
        return json.dumps(
            {
                "diagnostic_answers": ["uncertain"] * 5,
                "reasoning": "r",
                "failure_type": "noise",
                "confidence": 0.3,
            }
        )

    main_case = _make_case(cycle_id="main_case")
    ctx_case = _make_case(cycle_id="ctx_case_001", pair="BTC/USDT", final_pnl=-50.0)

    classify_case(main_case, llm_callable=_cap_llm, same_regime_cases=[ctx_case])

    prompt = captured[0]
    assert "ctx_case_001" in prompt


def test_empty_trade_execution_prompt_well_formed():
    """T018(h): empty trade_execution -> prompt still well-formed."""

    def _ok_llm(sys_p: str, usr_p: str) -> str:
        # Verify prompt doesn't crash with empty trade_execution
        assert "Trade Execution" in usr_p
        return json.dumps(
            {
                "diagnostic_answers": ["no"] * 5,
                "reasoning": "no data",
                "failure_type": "noise",
                "confidence": 0.1,
            }
        )

    case = _make_case(trade_execution=None)
    result = classify_case(case, llm_callable=_ok_llm)
    assert result.failure_type == "noise"
    assert result.case_id == "test_cycle"


# ── FailureClassification dataclass ──────────────────────────────────────────


def test_failure_classification_to_dict():
    """FailureClassification.to_dict() returns correct structure."""
    fc = FailureClassification(
        case_id="x",
        failure_type="fundamental",
        reasoning="the rule is wrong",
        confidence=0.9,
        diagnostic_answers=["yes", "no", "yes", "yes", "no"],
    )
    d = fc.to_dict()
    assert d["failure_type"] == "fundamental"
    assert d["confidence"] == 0.9
    assert len(d["diagnostic_answers"]) == 5


def test_noise_result_default():
    """_noise_result returns noise classification with 5 uncertain answers."""
    r = _noise_result("test_id")
    assert r.failure_type == "noise"
    assert len(r.diagnostic_answers) == 5
    assert all(a == "uncertain" for a in r.diagnostic_answers)

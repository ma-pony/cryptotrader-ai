"""spec 020a T025 — test_ive_async.py

FR-Z12: IVE unit tests rewritten with pytest.mark.asyncio + await.
All test cases mirror test_ive.py but call classify_case as async coroutine.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import MagicMock

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


def _make_sync_llm(failure_type: str, confidence: float = 0.8) -> MagicMock:
    """Create a mock sync LLM callable returning a specific failure type."""
    response = json.dumps(
        {
            "diagnostic_answers": ["yes", "no", "yes", "no", "uncertain"],
            "reasoning": f"Test reasoning for {failure_type}",
            "failure_type": failure_type,
            "confidence": confidence,
        }
    )
    return MagicMock(return_value=response)


# ── basic async classification tests ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_async_classify_implementation():
    """FR-Z12(a): await classify_case returns implementation classification."""
    case = _make_case()
    mock_llm = _make_sync_llm("implementation")

    result = await classify_case(case, llm_callable=mock_llm)

    assert result.failure_type == "implementation"
    assert result.case_id == "test_cycle"
    assert result.confidence > 0
    assert len(result.diagnostic_answers) == 5


@pytest.mark.asyncio
async def test_async_classify_fundamental():
    """FR-Z12(b): await classify_case returns fundamental classification."""
    case = _make_case()
    mock_llm = _make_sync_llm("fundamental")

    result = await classify_case(case, llm_callable=mock_llm)

    assert result.failure_type == "fundamental"


@pytest.mark.asyncio
async def test_async_classify_noise():
    """FR-Z12(c): await classify_case returns noise classification."""
    case = _make_case()
    mock_llm = _make_sync_llm("noise")

    result = await classify_case(case, llm_callable=mock_llm)

    assert result.failure_type == "noise"


@pytest.mark.asyncio
async def test_async_llm_failure_returns_noise(caplog: pytest.LogCaptureFixture):
    """FR-Z12(d): LLM raises -> noise + warning log."""

    def _failing_llm(_sys: str, _usr: str) -> str:
        raise RuntimeError("LLM unavailable")

    case = _make_case()

    with caplog.at_level("WARNING"):
        result = await classify_case(case, llm_callable=_failing_llm)

    assert result.failure_type == "noise"
    assert result.case_id == "test_cycle"
    assert any("LLM" in r.message or "noise" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_async_invalid_json_retries_then_noise(caplog: pytest.LogCaptureFixture):
    """FR-Z12(e): LLM returns non-JSON -> retry once -> noise."""
    call_count = 0

    def _bad_json_llm(_sys: str, _usr: str) -> str:
        nonlocal call_count
        call_count += 1
        return "this is not json at all"

    case = _make_case()

    with caplog.at_level("WARNING"):
        result = await classify_case(case, llm_callable=_bad_json_llm)

    assert result.failure_type == "noise"
    assert call_count == 2  # initial + retry


@pytest.mark.asyncio
async def test_async_prompt_contains_case_context():
    """FR-Z12(f): prompt includes trade_execution data and 5 diagnostic questions."""
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
    await classify_case(case, llm_callable=_capturing_llm)

    assert len(captured_prompts) == 1
    prompt = captured_prompts[0]
    assert "88.45" in prompt or "entry_price" in prompt
    assert "1." in prompt
    assert "5." in prompt


@pytest.mark.asyncio
async def test_async_same_regime_context_in_prompt():
    """FR-Z12(g): same-regime cases appear in prompt context."""
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

    await classify_case(main_case, llm_callable=_cap_llm, same_regime_cases=[ctx_case])

    prompt = captured[0]
    assert "ctx_case_001" in prompt


@pytest.mark.asyncio
async def test_async_empty_trade_execution():
    """FR-Z12(h): empty trade_execution -> prompt well-formed."""

    def _ok_llm(sys_p: str, usr_p: str) -> str:
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
    result = await classify_case(case, llm_callable=_ok_llm)
    assert result.failure_type == "noise"
    assert result.case_id == "test_cycle"


# ── is coroutine function check ───────────────────────────────────────────────


def test_classify_case_is_coroutine():
    """SC-Z4: classify_case must be an async def (coroutine function)."""
    import asyncio

    assert asyncio.iscoroutinefunction(classify_case), "classify_case must be async def per spec 020a FR-Z10"


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


# ── IveMetricsAggregator integration ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_ive_metrics_recorded_on_success():
    """spec 020a: IveMetricsAggregator receives success=True on successful classify."""
    from unittest.mock import patch

    from cryptotrader.observability.ive_metrics import IveMetricsAggregator

    fresh_agg = IveMetricsAggregator()
    mock_llm = _make_sync_llm("noise")
    case = _make_case()

    with patch(
        "cryptotrader.observability.ive_metrics.get_ive_metrics_aggregator",
        return_value=fresh_agg,
    ):
        await classify_case(case, llm_callable=mock_llm)

    # success recorded -> failure_rate == 0
    assert fresh_agg.failure_rate() == 0.0


@pytest.mark.asyncio
async def test_ive_metrics_recorded_on_failure():
    """spec 020a: IveMetricsAggregator receives success=False on LLM exception."""
    from unittest.mock import patch

    from cryptotrader.observability.ive_metrics import IveMetricsAggregator

    fresh_agg = IveMetricsAggregator()

    def _crash(_sys: str, _usr: str) -> str:
        raise RuntimeError("crash")

    case = _make_case()

    with patch(
        "cryptotrader.observability.ive_metrics.get_ive_metrics_aggregator",
        return_value=fresh_agg,
    ):
        result = await classify_case(case, llm_callable=_crash)

    assert result.failure_type == "noise"
    assert fresh_agg.failure_rate() == 1.0

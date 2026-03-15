"""Task 5.5 — risk rejection structured logs + MetricsCollector instrumentation.

Tests are written BEFORE implementation (TDD red phase).
Covers:
  - risk_check() structured rejection log fields: check_name, current_value, threshold
  - ct_llm_calls_total incremented by create_llm()
  - ct_debate_skipped_total incremented by debate_gate() when skip=True
  - ct_verdict_total incremented by make_verdict()
  - ct_risk_rejected_total incremented by risk_check() on rejection
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_state(verdict_action: str = "long", engine: str = "paper") -> dict:
    """Minimal ArenaState for risk_check tests."""
    return {
        "messages": [],
        "data": {
            "verdict": {
                "action": verdict_action,
                "confidence": 0.8,
                "position_scale": 0.8,
                "divergence": 0.1,
                "reasoning": "test",
                "thesis": "",
                "invalidation": "",
            },
            "portfolio": {
                "total_value": 10_000,
                "positions": {},
                "cash": 10_000,
                "daily_pnl": 0,
                "drawdown": 0,
                "returns_60d": [],
                "recent_prices": [],
                "funding_rate": 0,
                "api_latency_ms": 10,
                "pair": "BTC/USDT",
            },
        },
        "metadata": {
            "pair": "BTC/USDT",
            "engine": engine,
            "database_url": None,
            "redis_url": None,
        },
        "debate_round": 0,
        "max_debate_rounds": 2,
        "divergence_scores": [],
    }


def _make_debate_state(analyses: dict) -> dict:
    return {
        "messages": [],
        "data": {"analyses": analyses},
        "metadata": {"pair": "BTC/USDT", "engine": "paper"},
        "debate_round": 0,
        "max_debate_rounds": 2,
        "divergence_scores": [],
    }


def _mock_app_config(*, skip_debate: bool = True, consensus_threshold: float = 0.5):
    from cryptotrader.config import AppConfig, DebateConfig

    cfg = AppConfig()
    cfg.debate = DebateConfig(
        skip_debate=skip_debate,
        consensus_skip_threshold=consensus_threshold,
    )
    return cfg


def _sample_value(registry, metric_name: str, labels: dict) -> float:
    """Read Counter sample value from Prometheus registry."""
    base_name = metric_name.removesuffix("_total")
    for m in registry.collect():
        if m.name == base_name:
            for sample in m.samples:
                if (sample.name == f"{base_name}_total" or sample.name == base_name) and all(
                    sample.labels.get(k) == v for k, v in labels.items()
                ):
                    return sample.value
    return 0.0


# ===========================================================================
# 1. risk_check() structured rejection log
# ===========================================================================


@pytest.mark.asyncio
async def test_risk_check_rejection_logs_check_name(caplog):
    """When risk gate rejects, log must contain check_name field."""
    import logging

    from cryptotrader.nodes.verdict import risk_check

    state = _base_state()

    # Patch gate.check to return a rejection
    from cryptotrader.models import GateResult

    mock_gate = MagicMock()
    mock_gate.check = AsyncMock(
        return_value=GateResult(
            passed=False,
            rejected_by="volatility_gate",
            reason="Flash crash detected: 25.00% drop",
        )
    )

    with (
        patch("cryptotrader.nodes.verdict._risk_gate_cache", {"_default": mock_gate}),
        caplog.at_level(logging.WARNING, logger="cryptotrader.nodes.verdict"),
    ):
        await risk_check(state)

    # The rejection log must mention the check name
    rejection_logs = [r for r in caplog.records if "REJECTED" in r.getMessage() or "rejected" in r.getMessage().lower()]
    assert rejection_logs, "Expected at least one rejection log record"
    combined = " ".join(r.getMessage() for r in rejection_logs)
    assert "volatility_gate" in combined, f"check_name missing from log: {combined}"


@pytest.mark.asyncio
async def test_risk_check_rejection_logs_reason_summary(caplog):
    """When risk gate rejects, log must contain the reason summary."""
    import logging

    from cryptotrader.nodes.verdict import risk_check

    state = _base_state()

    from cryptotrader.models import GateResult

    mock_gate = MagicMock()
    mock_gate.check = AsyncMock(
        return_value=GateResult(
            passed=False,
            rejected_by="daily_loss_limit",
            reason="Daily loss 5% exceeds limit 3%",
        )
    )

    with (
        patch("cryptotrader.nodes.verdict._risk_gate_cache", {"_default": mock_gate}),
        caplog.at_level(logging.WARNING, logger="cryptotrader.nodes.verdict"),
    ):
        await risk_check(state)

    rejection_logs = [r for r in caplog.records if "REJECTED" in r.getMessage() or "rejected" in r.getMessage().lower()]
    assert rejection_logs, "Expected rejection log"
    combined = " ".join(r.getMessage() for r in rejection_logs)
    assert "daily_loss_limit" in combined
    assert "Daily loss" in combined or "5%" in combined


@pytest.mark.asyncio
async def test_risk_check_passed_does_not_log_rejection(caplog):
    """When risk gate passes, no REJECTED warning is emitted."""
    import logging

    from cryptotrader.nodes.verdict import risk_check

    state = _base_state()

    from cryptotrader.models import GateResult

    mock_gate = MagicMock()
    mock_gate.check = AsyncMock(return_value=GateResult(passed=True))

    with (
        patch("cryptotrader.nodes.verdict._risk_gate_cache", {"_default": mock_gate}),
        caplog.at_level(logging.WARNING, logger="cryptotrader.nodes.verdict"),
    ):
        await risk_check(state)

    rejection_logs = [r for r in caplog.records if "REJECTED" in r.getMessage()]
    assert not rejection_logs, "Passed verdict should not emit REJECTED log"


# ===========================================================================
# 2. ct_risk_rejected_total incremented on risk_check rejection
# ===========================================================================


@pytest.mark.asyncio
async def test_risk_check_rejection_increments_ct_risk_rejected_total():
    """risk_check() rejection must increment ct_risk_rejected_total[check_name]."""
    from prometheus_client import REGISTRY

    from cryptotrader.models import GateResult
    from cryptotrader.nodes.verdict import risk_check

    state = _base_state()
    check_name = "max_position_size"

    mock_gate = MagicMock()
    mock_gate.check = AsyncMock(
        return_value=GateResult(
            passed=False,
            rejected_by=check_name,
            reason="Position too large",
        )
    )

    before = _sample_value(REGISTRY, "ct_risk_rejected_total", {"check_name": check_name})

    with patch("cryptotrader.nodes.verdict._risk_gate_cache", {"_default": mock_gate}):
        await risk_check(state)

    after = _sample_value(REGISTRY, "ct_risk_rejected_total", {"check_name": check_name})
    assert after == before + 1, f"Expected counter to increment by 1, got {after - before}"


@pytest.mark.asyncio
async def test_risk_check_pass_does_not_increment_ct_risk_rejected_total():
    """Passed risk gate must NOT increment ct_risk_rejected_total."""
    from prometheus_client import REGISTRY

    from cryptotrader.models import GateResult
    from cryptotrader.nodes.verdict import risk_check

    state = _base_state()
    check_name = "max_position_size_pass_test"

    mock_gate = MagicMock()
    mock_gate.check = AsyncMock(return_value=GateResult(passed=True))

    before = _sample_value(REGISTRY, "ct_risk_rejected_total", {"check_name": check_name})

    with patch("cryptotrader.nodes.verdict._risk_gate_cache", {"_default": mock_gate}):
        await risk_check(state)

    after = _sample_value(REGISTRY, "ct_risk_rejected_total", {"check_name": check_name})
    assert after == before, "Passed gate should not increment ct_risk_rejected_total"


# ===========================================================================
# 3. ct_verdict_total incremented by make_verdict()
# ===========================================================================


@pytest.mark.asyncio
async def test_make_verdict_increments_ct_verdict_total():
    """make_verdict() must increment ct_verdict_total[action]."""
    from prometheus_client import REGISTRY

    from cryptotrader.nodes.verdict import make_verdict

    state = {
        "messages": [],
        "data": {
            "analyses": {
                "tech": {"direction": "bullish", "confidence": 0.8},
                "chain": {"direction": "bullish", "confidence": 0.7},
            },
            "debate_skipped": True,
            "position_context": {"side": "flat"},
        },
        "metadata": {
            "pair": "BTC/USDT",
            "engine": "paper",
            "llm_verdict": False,  # weighted verdict — no LLM needed
            "divergence_hold_threshold": 0.7,
        },
        "debate_round": 0,
        "max_debate_rounds": 2,
        "divergence_scores": [0.1],
    }

    # Sum all possible action counters before
    actions = ["long", "short", "hold", "close"]
    before_total = sum(_sample_value(REGISTRY, "ct_verdict_total", {"action": a}) for a in actions)

    await make_verdict(state)

    after_total = sum(_sample_value(REGISTRY, "ct_verdict_total", {"action": a}) for a in actions)
    assert after_total == before_total + 1, (
        f"ct_verdict_total should increase by 1; before={before_total}, after={after_total}"
    )


@pytest.mark.asyncio
async def test_make_verdict_all_mock_does_not_increment_ct_verdict_total():
    """When all analyses are mock (LLM outage), make_verdict forces hold — counter still increments."""
    from prometheus_client import REGISTRY

    from cryptotrader.nodes.verdict import make_verdict

    state = {
        "messages": [],
        "data": {
            "analyses": {
                "tech": {"direction": "neutral", "confidence": 0.1, "is_mock": True},
            },
            "debate_skipped": False,
            "position_context": None,
        },
        "metadata": {
            "pair": "BTC/USDT",
            "engine": "paper",
            "llm_verdict": True,
        },
        "debate_round": 0,
        "max_debate_rounds": 2,
        "divergence_scores": [],
    }

    before_hold = _sample_value(REGISTRY, "ct_verdict_total", {"action": "hold"})

    await make_verdict(state)

    after_hold = _sample_value(REGISTRY, "ct_verdict_total", {"action": "hold"})
    assert after_hold == before_hold + 1, "All-mock forced hold must still increment ct_verdict_total"


# ===========================================================================
# 4. ct_debate_skipped_total incremented by debate_gate() on skip
# ===========================================================================


@pytest.mark.asyncio
async def test_debate_gate_skip_increments_ct_debate_skipped_total():
    """debate_gate() with strong consensus must increment ct_debate_skipped_total."""
    from prometheus_client import REGISTRY

    from cryptotrader.nodes.debate import debate_gate

    analyses = {
        "tech": {"direction": "bullish", "confidence": 0.9},
        "chain": {"direction": "bullish", "confidence": 0.85},
        "news": {"direction": "bullish", "confidence": 0.88},
        "macro": {"direction": "bullish", "confidence": 0.92},
    }
    state = _make_debate_state(analyses)

    before = _sample_value(REGISTRY, "ct_debate_skipped_total", {})

    # load_config is locally imported inside debate_gate; patch at config module level
    with patch("cryptotrader.config.load_config", return_value=_mock_app_config(skip_debate=True)):
        result = await debate_gate(state)

    after = _sample_value(REGISTRY, "ct_debate_skipped_total", {})

    # Only increment when debate was actually skipped
    if result["data"]["debate_skipped"]:
        assert after == before + 1, f"ct_debate_skipped_total should increment; before={before}, after={after}"
    else:
        assert after == before, "ct_debate_skipped_total must not increment when debate was not skipped"


@pytest.mark.asyncio
async def test_debate_gate_no_skip_does_not_increment_ct_debate_skipped_total():
    """debate_gate() when debate proceeds must NOT increment ct_debate_skipped_total."""
    from prometheus_client import REGISTRY

    from cryptotrader.nodes.debate import debate_gate

    analyses = {
        "tech": {"direction": "bullish", "confidence": 0.9},
        "chain": {"direction": "bearish", "confidence": 0.9},
        "news": {"direction": "bullish", "confidence": 0.85},
        "macro": {"direction": "bearish", "confidence": 0.85},
    }
    state = _make_debate_state(analyses)

    before = _sample_value(REGISTRY, "ct_debate_skipped_total", {})

    with patch("cryptotrader.config.load_config", return_value=_mock_app_config(skip_debate=True)):
        result = await debate_gate(state)

    after = _sample_value(REGISTRY, "ct_debate_skipped_total", {})

    assert result["data"]["debate_skipped"] is False
    assert after == before, "ct_debate_skipped_total must not increment when debate was NOT skipped"


# ===========================================================================
# 5. ct_llm_calls_total incremented by create_llm()
# ===========================================================================


def test_create_llm_increments_ct_llm_calls_total():
    """create_llm() must increment ct_llm_calls_total[model, node]."""
    from prometheus_client import REGISTRY

    from cryptotrader.agents.base import create_llm
    from cryptotrader.config import AppConfig, LLMConfig, ModelConfig

    cfg = AppConfig()
    cfg.models = ModelConfig(analysis="gpt-4o-mini", fallback="gpt-4o-mini")
    cfg.llm = LLMConfig()

    before = _sample_value(REGISTRY, "ct_llm_calls_total", {"model": "gpt-4o-mini", "node": "create_llm"})

    # Patch ChatOpenAI to avoid needing a real API key; patch load_config for config
    with (
        patch("cryptotrader.config.load_config", return_value=cfg),
        patch("cryptotrader.agents.base.ChatOpenAI", return_value=MagicMock()),
    ):
        create_llm(model="gpt-4o-mini")

    after = _sample_value(REGISTRY, "ct_llm_calls_total", {"model": "gpt-4o-mini", "node": "create_llm"})
    assert after == before + 1, f"ct_llm_calls_total should increment; before={before}, after={after}"


def test_create_llm_resolves_empty_model_and_increments():
    """create_llm('') resolves to config model and still increments counter."""
    from prometheus_client import REGISTRY

    from cryptotrader.agents.base import create_llm
    from cryptotrader.config import AppConfig, LLMConfig, ModelConfig

    cfg = AppConfig()
    cfg.models = ModelConfig(analysis="gpt-4o", fallback="gpt-4o")
    cfg.llm = LLMConfig()

    before = _sample_value(REGISTRY, "ct_llm_calls_total", {"model": "gpt-4o", "node": "create_llm"})

    with (
        patch("cryptotrader.config.load_config", return_value=cfg),
        patch("cryptotrader.agents.base.ChatOpenAI", return_value=MagicMock()),
    ):
        create_llm(model="")

    after = _sample_value(REGISTRY, "ct_llm_calls_total", {"model": "gpt-4o", "node": "create_llm"})
    assert after == before + 1

"""Tests for verdict downgrade when debate is skipped."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from cryptotrader.nodes.verdict import _should_downgrade_to_weighted, make_verdict


def _base_state(debate_skipped=False, position_side="flat", **extra_data):
    """Build minimal state for verdict tests."""
    data = {
        "analyses": {
            "tech_agent": {
                "direction": "bullish",
                "confidence": 0.8,
                "reasoning": "test",
                "key_factors": [],
                "risk_flags": [],
            },
            "chain_agent": {
                "direction": "bullish",
                "confidence": 0.7,
                "reasoning": "test",
                "key_factors": [],
                "risk_flags": [],
            },
        },
        "debate_skipped": debate_skipped,
        "position_context": {"side": position_side},
        **extra_data,
    }
    return {
        "messages": [],
        "data": data,
        "metadata": {
            "pair": "BTC/USDT",
            "engine": "paper",
            "llm_verdict": True,
        },
        "debate_round": 0,
        "max_debate_rounds": 2,
        "divergence_scores": [],
    }


@pytest.mark.asyncio
async def test_downgrade_flat_no_circuit_breaker():
    """Flat position + no circuit breaker → downgrade OK."""
    state = _base_state(position_side="flat")
    mock_rsm = AsyncMock()
    mock_rsm.is_circuit_breaker_active = AsyncMock(return_value=False)
    with patch("cryptotrader.risk.state.RedisStateManager", return_value=mock_rsm):
        result = await _should_downgrade_to_weighted(state)
    assert result is True


@pytest.mark.asyncio
async def test_no_downgrade_with_position():
    """Has position → no downgrade."""
    state = _base_state(position_side="long")
    result = await _should_downgrade_to_weighted(state)
    assert result is False


@pytest.mark.asyncio
async def test_no_downgrade_circuit_breaker_active():
    """Circuit breaker active → no downgrade."""
    state = _base_state(position_side="flat")
    mock_rsm = AsyncMock()
    mock_rsm.is_circuit_breaker_active = AsyncMock(return_value=True)
    with patch("cryptotrader.risk.state.RedisStateManager", return_value=mock_rsm):
        result = await _should_downgrade_to_weighted(state)
    assert result is False


@pytest.mark.asyncio
async def test_no_downgrade_redis_unavailable():
    """Redis unavailable → conservative, no downgrade."""
    state = _base_state(position_side="flat")
    with patch("cryptotrader.risk.state.RedisStateManager", side_effect=Exception("conn refused")):
        result = await _should_downgrade_to_weighted(state)
    assert result is False


@pytest.mark.asyncio
async def test_verdict_uses_weighted_when_debate_skipped():
    """When debate skipped + downgrade conditions met → weighted verdict."""
    state = _base_state(debate_skipped=True, position_side="flat")
    mock_rsm = AsyncMock()
    mock_rsm.is_circuit_breaker_active = AsyncMock(return_value=False)
    with (
        patch("cryptotrader.risk.state.RedisStateManager", return_value=mock_rsm),
        patch("cryptotrader.debate.verdict.make_verdict_weighted") as mock_weighted,
        patch("cryptotrader.debate.verdict.make_verdict_llm") as mock_llm,
    ):
        mock_weighted.return_value = type(
            "V",
            (),
            {
                "action": "hold",
                "confidence": 0.5,
                "position_scale": 0.0,
                "divergence": 0.0,
                "reasoning": "weighted",
                "thesis": "",
                "invalidation": "",
            },
        )()
        result = await make_verdict(state)
        mock_weighted.assert_called_once()
        mock_llm.assert_not_called()
    assert result["data"]["verdict"]["action"] == "hold"


@pytest.mark.asyncio
async def test_verdict_uses_llm_when_debate_not_skipped():
    """Normal flow (debate not skipped) → AI verdict."""
    state = _base_state(debate_skipped=False)
    with (
        patch("cryptotrader.nodes.verdict._gather_risk_constraints", new_callable=AsyncMock, return_value={}),
        patch("cryptotrader.debate.verdict.make_verdict_llm", new_callable=AsyncMock) as mock_llm,
        patch("cryptotrader.debate.verdict.make_verdict_weighted") as mock_weighted,
        patch("cryptotrader.config.load_config") as mock_cfg,
    ):
        mock_cfg.return_value.models.verdict = "test-model"
        mock_cfg.return_value.models.fallback = "fallback"
        mock_llm.return_value = type(
            "V",
            (),
            {
                "action": "long",
                "confidence": 0.8,
                "position_scale": 0.5,
                "divergence": 0.1,
                "reasoning": "ai verdict",
                "thesis": "test",
                "invalidation": "test",
            },
        )()
        result = await make_verdict(state)
        mock_llm.assert_called_once()
        mock_weighted.assert_not_called()
    assert result["data"]["verdict"]["action"] == "long"

"""Tests for debate turn capture + classification in nodes/debate.py."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage

from cryptotrader.nodes.debate import _classify_move, _debate_one_agent, _dir_label


class TestClassifyMove:
    def test_same_direction_conf_up_strengthens(self) -> None:
        assert _classify_move("bullish", 0.6, "bullish", 0.8) == "强化"

    def test_same_direction_conf_down_weakens(self) -> None:
        assert _classify_move("bearish", 0.8, "bearish", 0.6) == "弱化"

    def test_same_direction_small_delta_holds(self) -> None:
        assert _classify_move("bullish", 0.55, "bullish", 0.57) == "保持"

    def test_flip_bullish_to_bearish_yields_concession_label(self) -> None:
        result = _classify_move("bullish", 0.6, "bearish", 0.5)
        assert "让步" in result
        assert "看多" in result
        assert "看空" in result

    def test_flip_bullish_to_neutral_yields_concession(self) -> None:
        assert _classify_move("bullish", 0.6, "neutral", 0.3) == "让步(由看多转中性)"


class TestDirLabel:
    def test_all_canonical(self) -> None:
        assert _dir_label("bullish") == "看多"
        assert _dir_label("bearish") == "看空"
        assert _dir_label("neutral") == "中性"

    def test_unknown_passthrough(self) -> None:
        assert _dir_label("unknown") == "unknown"


def _make_analysis(direction: str = "bullish", confidence: float = 0.6) -> dict:
    return {
        "direction": direction,
        "confidence": confidence,
        "reasoning": "initial take",
        "key_factors": ["factor_a"],
        "risk_flags": [],
    }


@pytest.mark.asyncio
async def test_debate_one_agent_emits_turn_with_before_after() -> None:
    analysis = _make_analysis("bullish", 0.6)
    others = {"chain_agent": _make_analysis("bearish", 0.5)}
    response_json = (
        '{"direction": "bullish", "confidence": 0.85, "reasoning": "updated",'
        ' "key_factors": ["f1"], "risk_flags": [], "new_findings": "cross-domain insight"}'
    )
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content=response_json))
    with patch("cryptotrader.nodes.debate.create_llm", return_value=mock_llm):
        aid, result, turn = await _debate_one_agent(
            "tech_agent",
            analysis,
            others,
            "BTC/USDT",
            "test-model",
            60,
            2,
        )
    assert aid == "tech_agent"
    assert result["confidence"] == 0.85
    assert turn["round"] == 2
    assert turn["from"] == "tech_agent"
    assert turn["to"] == "chain_agent"
    assert turn["before"] == {"direction": "bullish", "confidence": 0.6}
    assert turn["after"] == {"direction": "bullish", "confidence": 0.85}
    assert turn["move"] == "强化"
    assert turn["new_findings"] == "cross-domain insight"
    assert turn["errored"] is False


@pytest.mark.asyncio
async def test_debate_one_agent_timeout_records_errored_turn() -> None:
    analysis = _make_analysis("bullish", 0.6)
    others = {"chain_agent": _make_analysis("bearish", 0.5)}
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(side_effect=TimeoutError())
    with patch("cryptotrader.nodes.debate.create_llm", return_value=mock_llm):
        aid, result, turn = await _debate_one_agent(
            "tech_agent",
            analysis,
            others,
            "BTC/USDT",
            "test-model",
            0.01,
            1,
        )
    assert aid == "tech_agent"
    assert result == analysis  # unchanged
    assert turn["errored"] is True
    assert turn["move"] == "保持"  # no delta on error


@pytest.mark.asyncio
async def test_debate_one_agent_monologue_when_no_others() -> None:
    analysis = _make_analysis("neutral", 0.4)
    others: dict[str, dict] = {}
    response_json = (
        '{"direction": "neutral", "confidence": 0.4, "reasoning": "solo",'
        ' "key_factors": [], "risk_flags": [], "new_findings": ""}'
    )
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content=response_json))
    with patch("cryptotrader.nodes.debate.create_llm", return_value=mock_llm):
        _aid, _result, turn = await _debate_one_agent(
            "macro_agent",
            analysis,
            others,
            "BTC/USDT",
            "test-model",
            60,
            1,
        )
    assert turn["to"] is None  # UI renders as "独白" when no addressee

"""严格交叉辩论生成可持久化的 before/after turn。"""

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage

from cryptotrader.debate.challenge import challenge_agent, classify_move, direction_label


def _analysis(direction="bullish", confidence=0.6):
    return {
        "direction": direction,
        "confidence": confidence,
        "reasoning": "initial",
        "key_factors": [],
        "risk_flags": [],
    }


@pytest.mark.parametrize(
    ("before", "before_conf", "after", "after_conf", "expected"),
    [
        ("bullish", 0.6, "bullish", 0.8, "强化"),
        ("bearish", 0.8, "bearish", 0.6, "弱化"),
        ("bullish", 0.55, "bullish", 0.57, "保持"),
        ("bullish", 0.6, "neutral", 0.3, "让步(由看多转中性)"),
    ],
)
def test_classify_move(before, before_conf, after, after_conf, expected):
    assert classify_move(before, before_conf, after, after_conf) == expected


def test_direction_label_preserves_unknown_values():
    assert direction_label("bearish") == "看空"
    assert direction_label("unknown") == "unknown"


@pytest.mark.asyncio
async def test_challenge_captures_turn_with_before_and_after():
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(return_value=AIMessage(content="{}"))
    payload = {
        "direction": "bullish",
        "confidence": 0.85,
        "reasoning": "updated",
        "key_factors": ["f1"],
        "risk_flags": [],
        "new_findings": "[NEW] chain funding 0.03%",
    }
    with (
        patch("cryptotrader.debate.challenge.create_llm", return_value=llm),
        patch("cryptotrader.debate.challenge.extract_json_with_retry", new=AsyncMock(return_value=payload)),
    ):
        updated, turn = await challenge_agent(
            "tech_agent",
            _analysis(),
            {"chain_agent": _analysis("bearish", 0.5)},
            "BTC/USDT",
            "test",
            30,
            2,
        )

    assert updated["confidence"] == 0.85
    assert turn["round"] == 2
    assert turn["from"] == "tech_agent"
    assert turn["to"] == "chain_agent"
    assert turn["before"] == {"direction": "bullish", "confidence": 0.6}
    assert turn["after"] == {"direction": "bullish", "confidence": 0.85}
    assert turn["move"] == "强化"
    assert turn["errored"] is False


@pytest.mark.asyncio
async def test_challenge_failure_propagates_instead_of_preserving_original():
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(side_effect=TimeoutError("timeout"))
    with (
        patch("cryptotrader.debate.challenge.create_llm", return_value=llm),
        pytest.raises(TimeoutError),
    ):
        await challenge_agent("tech_agent", _analysis(), {}, "BTC/USDT", "test", 0.01, 1)


@pytest.mark.asyncio
async def test_challenge_without_opponents_is_a_monologue():
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(return_value=AIMessage(content="{}"))
    payload = {
        "direction": "neutral",
        "confidence": 0.4,
        "reasoning": "solo",
        "key_factors": [],
        "risk_flags": [],
        "new_findings": "",
    }
    with (
        patch("cryptotrader.debate.challenge.create_llm", return_value=llm),
        patch("cryptotrader.debate.challenge.extract_json_with_retry", new=AsyncMock(return_value=payload)),
    ):
        _, turn = await challenge_agent("macro_agent", _analysis("neutral", 0.4), {}, "BTC", "test", 30, 1)

    assert turn["to"] is None

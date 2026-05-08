"""N4: server-side anti-ratchet enforcement in ``_debate_one_agent``.

A confidence raise > +0.02 with the same direction must be backed by a
``new_findings`` field starting with the literal token ``[NEW]``. Otherwise
the raise is rolled back to ``before + 0.02``. Direction flips bypass the
rule (a real change-of-mind is allowed without the discipline; the
discipline targets confidence drift on an unchanged stance).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage

from cryptotrader.nodes.debate import _debate_one_agent


def _llm_response(direction: str, confidence: float, new_findings: str = "", reasoning: str = "x") -> AIMessage:
    """Mock an LLM JSON response with the given fields."""
    import json

    payload = {
        "direction": direction,
        "confidence": confidence,
        "reasoning": reasoning,
        "key_factors": [],
        "risk_flags": [],
        "new_findings": new_findings,
    }
    return AIMessage(content=json.dumps(payload))


@pytest.mark.asyncio
async def test_raise_without_new_tag_rolled_back():
    """+0.10 confidence raise without [NEW] → snap to before + 0.02."""
    analysis = {"direction": "bearish", "confidence": 0.60, "reasoning": "old"}
    fake = _llm_response("bearish", 0.70, new_findings="other agents agree")
    with patch("cryptotrader.nodes.debate.create_llm") as cm:
        cm.return_value.ainvoke = AsyncMock(return_value=fake)
        with patch(
            "cryptotrader.llm.json_retry.extract_json_with_retry",
            new_callable=AsyncMock,
            return_value={
                "direction": "bearish",
                "confidence": 0.70,
                "reasoning": "x",
                "key_factors": [],
                "risk_flags": [],
                "new_findings": "other agents agree",
            },
        ):
            _, merged, _ = await _debate_one_agent(
                "tech_agent",
                analysis,
                others={},
                pair="BTC",
                model="m",
                timeout_seconds=30,
                round_number=1,
            )
    assert merged["confidence"] == pytest.approx(0.62)  # 0.60 + 0.02 cap


@pytest.mark.asyncio
async def test_raise_with_new_tag_passes():
    """+0.08 raise with [NEW] prefix → unchanged."""
    analysis = {"direction": "bearish", "confidence": 0.60, "reasoning": "old"}
    with patch("cryptotrader.nodes.debate.create_llm") as cm:
        cm.return_value.ainvoke = AsyncMock(return_value=AIMessage(content="{}"))
        with patch(
            "cryptotrader.llm.json_retry.extract_json_with_retry",
            new_callable=AsyncMock,
            return_value={
                "direction": "bearish",
                "confidence": 0.68,
                "reasoning": "x",
                "key_factors": [],
                "risk_flags": [],
                "new_findings": "[NEW] chain_agent's long/short ratio 1.78",
            },
        ):
            _, merged, _ = await _debate_one_agent(
                "tech_agent",
                analysis,
                others={},
                pair="BTC",
                model="m",
                timeout_seconds=30,
                round_number=1,
            )
    assert merged["confidence"] == pytest.approx(0.68)


@pytest.mark.asyncio
async def test_small_raise_within_tolerance_passes():
    """+0.02 raise without [NEW] is within the ±0.02 tolerance band → unchanged."""
    analysis = {"direction": "bearish", "confidence": 0.60, "reasoning": "old"}
    with patch("cryptotrader.nodes.debate.create_llm") as cm:
        cm.return_value.ainvoke = AsyncMock(return_value=AIMessage(content="{}"))
        with patch(
            "cryptotrader.llm.json_retry.extract_json_with_retry",
            new_callable=AsyncMock,
            return_value={
                "direction": "bearish",
                "confidence": 0.62,
                "reasoning": "x",
                "key_factors": [],
                "risk_flags": [],
                "new_findings": "no new evidence",
            },
        ):
            _, merged, _ = await _debate_one_agent(
                "tech_agent",
                analysis,
                others={},
                pair="BTC",
                model="m",
                timeout_seconds=30,
                round_number=1,
            )
    assert merged["confidence"] == pytest.approx(0.62)  # +0.02 is allowed


@pytest.mark.asyncio
async def test_lower_confidence_always_passes():
    """Anti-ratchet only restricts raises; lowering is always allowed."""
    analysis = {"direction": "bearish", "confidence": 0.60, "reasoning": "old"}
    with patch("cryptotrader.nodes.debate.create_llm") as cm:
        cm.return_value.ainvoke = AsyncMock(return_value=AIMessage(content="{}"))
        with patch(
            "cryptotrader.llm.json_retry.extract_json_with_retry",
            new_callable=AsyncMock,
            return_value={
                "direction": "bearish",
                "confidence": 0.40,
                "reasoning": "x",
                "key_factors": [],
                "risk_flags": [],
                "new_findings": "i was wrong",
            },
        ):
            _, merged, _ = await _debate_one_agent(
                "tech_agent",
                analysis,
                others={},
                pair="BTC",
                model="m",
                timeout_seconds=30,
                round_number=1,
            )
    assert merged["confidence"] == pytest.approx(0.40)


@pytest.mark.asyncio
async def test_direction_flip_bypasses_check():
    """A real change of mind (direction flip) doesn't need [NEW]."""
    analysis = {"direction": "bearish", "confidence": 0.60, "reasoning": "old"}
    with patch("cryptotrader.nodes.debate.create_llm") as cm:
        cm.return_value.ainvoke = AsyncMock(return_value=AIMessage(content="{}"))
        with patch(
            "cryptotrader.llm.json_retry.extract_json_with_retry",
            new_callable=AsyncMock,
            return_value={
                "direction": "bullish",  # flipped
                "confidence": 0.70,  # higher than 0.60 + 0.02
                "reasoning": "x",
                "key_factors": [],
                "risk_flags": [],
                "new_findings": "saw something different",  # no [NEW] tag
            },
        ):
            _, merged, _ = await _debate_one_agent(
                "tech_agent",
                analysis,
                others={},
                pair="BTC",
                model="m",
                timeout_seconds=30,
                round_number=1,
            )
    # Direction flip: anti-ratchet does NOT apply, raise allowed.
    assert merged["confidence"] == pytest.approx(0.70)
    assert merged["direction"] == "bullish"


@pytest.mark.asyncio
async def test_new_tag_case_insensitive():
    """[new] / [New] also count — be forgiving on case."""
    analysis = {"direction": "bearish", "confidence": 0.60, "reasoning": "old"}
    with patch("cryptotrader.nodes.debate.create_llm") as cm:
        cm.return_value.ainvoke = AsyncMock(return_value=AIMessage(content="{}"))
        with patch(
            "cryptotrader.llm.json_retry.extract_json_with_retry",
            new_callable=AsyncMock,
            return_value={
                "direction": "bearish",
                "confidence": 0.72,
                "reasoning": "x",
                "key_factors": [],
                "risk_flags": [],
                "new_findings": "[new] real datum here",
            },
        ):
            _, merged, _ = await _debate_one_agent(
                "tech_agent",
                analysis,
                others={},
                pair="BTC",
                model="m",
                timeout_seconds=30,
                round_number=1,
            )
    assert merged["confidence"] == pytest.approx(0.72)

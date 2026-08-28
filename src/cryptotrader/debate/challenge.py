"""Cross-challenge prompt builder for multi-agent debate.

Phase 4C: Show full agent analyses, demand new_findings, anti-convergence stance.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from cryptotrader.agents.base import create_llm, extract_content
from cryptotrader.llm.json_retry import extract_json_with_retry

if TYPE_CHECKING:
    from collections.abc import Callable

_DEBATE_ROLES = {
    "tech_agent": "technical analysis",
    "chain_agent": "on-chain and derivatives analysis",
    "news_agent": "news and sentiment analysis",
    "macro_agent": "macroeconomic analysis",
}

_DIRECTION_LABELS = {"bullish": "看多", "bearish": "看空", "neutral": "中性"}

DEBATE_SYSTEM = """You are a {role} specialist in a multi-agent trading debate.

Base every claim on specific data. Maintain your stance when the evidence supports it.
Change direction only when another analyst provides relevant evidence you had not considered.
A confidence increase above 0.02 requires new_findings to start with [NEW] and cite that evidence.

Return JSON only with direction, confidence, reasoning, key_factors, risk_flags and new_findings.
Direction must be bullish, bearish or neutral. Free-text fields must use Simplified Chinese."""


def build_challenge_prompt(
    agent_role: str,
    pair: str,
    own_analysis: dict,
    other_analyses: dict[str, dict],
) -> str:
    # Show full analyses — not just direction+confidence summaries
    others = "\n\n".join(
        f"── {name.upper()} ──\n{json.dumps(a, indent=2, default=str)}" for name, a in other_analyses.items()
    )
    own = json.dumps(own_analysis, indent=2, default=str)

    return (
        f"You are a {agent_role} analyst reviewing {pair}.\n\n"
        f"YOUR PREVIOUS ANALYSIS:\n{own}\n\n"
        f"OTHER AGENTS' ANALYSES:\n{others}\n\n"
        "CHALLENGE PROTOCOL:\n"
        "1. Attack weak arguments: For each other agent, identify the weakest claim. "
        "Does their reasoning cite specific data, or is it vague? Are there logical leaps?\n"
        "2. Defend your position: What counter-evidence did others raise against your view? "
        "Is it strong enough to change your mind, or does your data still hold?\n"
        "3. Surface new findings: What did you notice in OTHER agents' data that they missed "
        "or misinterpreted? Cross-domain insights (e.g., on-chain data contradicting news sentiment) "
        "are especially valuable.\n\n"
        "ANTI-CONVERGENCE RULES:\n"
        "- Do NOT move toward consensus unless you see genuinely new evidence that changes your view.\n"
        "- 'The other agents also think X' is NOT evidence. Only data is evidence.\n"
        "- If your original analysis was correct, MAINTAIN your stance — even if you're the only one.\n"
        "- Lowering confidence just because others disagree is intellectual cowardice. Don't do it.\n"
        "- If you DO change your view, explain EXACTLY which data point changed your mind.\n\n"
        "LANGUAGE POLICY (MANDATORY):\n"
        "- 所有自由文本字段（`reasoning` / `key_factors` / `risk_flags` / `new_findings`）必须使用 **简体中文** 输出。\n"
        '- JSON keys 和 `direction` enum 值（`"bullish"` / `"bearish"` / `"neutral"`）保持英文小写字面值。\n'
        "- 指标名（RSI / MACD / OI / funding / TVL 等）作为术语保留英文；数字保持裸数字。\n\n"
        "Output JSON with these fields:\n"
        '{"direction": "bullish|bearish|neutral", "confidence": 0.0-1.0, '
        '"reasoning": "中文 2-3 句", "key_factors": [...中文...], "risk_flags": [...中文...], '
        '"new_findings": "中文：从其他 agent 数据中发现的跨域洞察"}'
    )


def direction_label(direction: str) -> str:
    return _DIRECTION_LABELS.get(direction, direction)


def classify_move(
    before_direction: str, before_confidence: float, after_direction: str, after_confidence: float
) -> str:
    if before_direction != after_direction:
        return f"让步(由{direction_label(before_direction)}转{direction_label(after_direction)})"
    difference = after_confidence - before_confidence
    if difference >= 0.05:
        return "强化"
    if difference <= -0.05:
        return "弱化"
    return "保持"


def apply_anti_ratchet(
    before_direction: str,
    before_confidence: float,
    after_direction: str,
    after_confidence: float,
    new_findings: str,
) -> float:
    if after_direction != before_direction:
        return after_confidence
    if after_confidence - before_confidence <= 0.02:
        return after_confidence
    if new_findings.lstrip().upper().startswith("[NEW]"):
        return after_confidence
    return round(before_confidence + 0.02, 4)


async def challenge_agent(
    agent_id: str,
    analysis: dict,
    others: dict[str, dict],
    pair: str,
    model: str,
    timeout_seconds: float,
    round_number: int,
    *,
    llm_factory: Callable | None = None,
    prompt_caching: bool = False,
) -> tuple[dict, dict]:
    """Run one strict cross-challenge turn; failures propagate to the committee."""
    prompt = build_challenge_prompt(agent_id, pair, analysis, others)
    llm = (llm_factory or create_llm)(model=model, temperature=0.3)
    messages = [
        SystemMessage(content=DEBATE_SYSTEM.format(role=_DEBATE_ROLES.get(agent_id, agent_id))),
        HumanMessage(content=prompt),
    ]
    if prompt_caching:
        from cryptotrader.llm.prompt_cache import apply_cache_control, is_anthropic_model

        if is_anthropic_model(model):
            messages = apply_cache_control(messages)
    response = await asyncio.wait_for(
        llm.ainvoke(messages),
        timeout=timeout_seconds,
    )
    payload = await extract_json_with_retry(
        extract_content(response),
        llm=llm,
        schema_hint="direction,confidence,reasoning,key_factors,risk_flags,new_findings",
        max_retries=2,
    )
    before_direction = analysis.get("direction", "neutral")
    before_confidence = float(analysis.get("confidence", 0.0) or 0.0)
    after_direction = payload["direction"]
    if after_direction not in {"bullish", "bearish", "neutral"}:
        raise ValueError(f"invalid debate direction: {after_direction}")
    new_findings = str(payload.get("new_findings", "") or "")
    after_confidence = apply_anti_ratchet(
        before_direction,
        before_confidence,
        after_direction,
        float(payload["confidence"]),
        new_findings,
    )
    if not 0.0 <= after_confidence <= 1.0:
        raise ValueError("debate confidence must be in [0, 1]")

    updated = {
        **analysis,
        "direction": after_direction,
        "confidence": after_confidence,
        "reasoning": payload["reasoning"],
        "key_factors": payload.get("key_factors", []),
        "risk_flags": payload.get("risk_flags", []),
        "new_findings": new_findings,
    }
    addressee = min(others, key=lambda key: float(others[key].get("confidence", 0.0) or 0.0)) if others else None
    turn = {
        "round": round_number,
        "from": agent_id,
        "to": addressee,
        "before": {"direction": before_direction, "confidence": before_confidence},
        "after": {"direction": after_direction, "confidence": after_confidence},
        "move": classify_move(before_direction, before_confidence, after_direction, after_confidence),
        "reasoning": payload["reasoning"],
        "new_findings": new_findings,
        "errored": False,
    }
    return updated, turn

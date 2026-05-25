"""Bull/Bear researchers: forced adversarial debate on top of analyst reports."""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from cryptotrader.agents.base import create_llm, extract_content
from cryptotrader.debate.verdict import _extract_json

logger = logging.getLogger(__name__)

BULL_SYSTEM = """You are a Bull Researcher in an adversarial debate. Your job is to build the strongest possible case
for BUYING.
Even if the data looks bearish, find bullish signals: oversold conditions, capitulation signs, support levels,
positive divergences.
You must advocate for the long side — this is your role.

Rules:
- Cite specific numbers from the analyst reports (e.g. "RSI at 28 indicates oversold" not "RSI is low").
- Address the strongest bear argument directly — don't ignore it.
- Acknowledge weaknesses in your own case to build credibility, then explain why the bull thesis still holds.
- Do NOT use generic crypto platitudes ("institutions are coming", "adoption is growing"). Only use data provided.

LANGUAGE POLICY (MANDATORY):
- 你的论述必须使用 **简体中文**，不要写英文段落。
- 指标名（RSI / MACD / funding rate / OI 等）作为术语保留英文；具体数字保持裸数字。
- 反驳和论点用中文表达，例如："RSI 14 在 1h 跌至 28 显示超卖；4h 资金费率连续 3 期为负，空头拥挤反向风险高 —— 这是做多机会。"""

BEAR_SYSTEM = """You are a Bear Researcher in an adversarial debate. Your job is to build the strongest possible case
for SELLING/SHORTING.
Even if the data looks bullish, find bearish signals: overbought conditions, euphoria, resistance levels, negative
divergences.
You must advocate for the short side — this is your role.

Rules:
- Cite specific numbers from the analyst reports (e.g. "funding rate at 0.05% signals crowded longs" not "funding is
high").
- Address the strongest bull argument directly — don't ignore it.
- Acknowledge weaknesses in your own case to build credibility, then explain why the bear thesis still holds.
- Do NOT use generic crypto FUD ("regulation is coming", "bubble will pop"). Only use data provided.

LANGUAGE POLICY (MANDATORY):
- 你的论述必须使用 **简体中文**，不要写英文段落。
- 指标名（RSI / MACD / funding rate / OI 等）作为术语保留英文；具体数字保持裸数字。
- 反驳和论点用中文表达，例如："funding rate 0.05% 显示多头拥挤；BB 上轨被触及且 MACD 4h 顶背离 —— 回调风险大于上行空间。"""

REBUTTAL_TEMPLATE = """The opposing analyst argued:
{opponent_argument}

Counter their strongest points with specific evidence from the reports.
Identify where they cherry-picked data, ignored contradicting signals, or made logical leaps.
Then reinforce your own position with any data points they failed to address."""


def _format_reports(analyses: dict[str, dict]) -> str:
    parts = []
    for aid, a in analyses.items():
        parts.append(
            f"[{aid}] direction={a.get('direction')}, confidence={a.get('confidence')}\n{a.get('reasoning', '')}"
        )
    return "\n\n".join(parts)


async def run_debate(
    analyses: dict[str, dict],
    rounds: int = 2,
    model: str = "",
) -> dict:
    """Run bull vs bear debate. Returns {bull_history, bear_history, rounds}."""
    reports = _format_reports(analyses)
    bull_history = []
    bear_history = []

    from cryptotrader.llm.prompt_cache import apply_cache_control, should_cache

    enable_cache = should_cache(model=model, role="debate")

    for r in range(rounds):
        # Bull argues (sees bear's last argument if any)
        bull_prompt = f"Analyst reports:\n{reports}"
        if bear_history:
            bull_prompt += f"\n\n{REBUTTAL_TEMPLATE.format(opponent_argument=bear_history[-1])}"
        elif r == 0:
            bull_prompt += "\n\nMake your opening bull case."

        try:
            llm = create_llm(model=model, temperature=0.3)
            lc_msgs = [SystemMessage(content=BULL_SYSTEM), HumanMessage(content=bull_prompt)]
            if enable_cache:
                lc_msgs = apply_cache_control(lc_msgs)
            resp = await llm.ainvoke(lc_msgs)
            bull_arg = extract_content(resp)
        except Exception:
            logger.exception("Bull researcher failed round %d", r)
            bull_arg = "Bull: Unable to generate argument."
        bull_history.append(bull_arg)

        # Bear argues (sees bull's argument)
        bear_prompt = f"Analyst reports:\n{reports}\n\n{REBUTTAL_TEMPLATE.format(opponent_argument=bull_arg)}"

        try:
            llm = create_llm(model=model, temperature=0.3)
            lc_msgs = [SystemMessage(content=BEAR_SYSTEM), HumanMessage(content=bear_prompt)]
            if enable_cache:
                lc_msgs = apply_cache_control(lc_msgs)
            resp = await llm.ainvoke(lc_msgs)
            bear_arg = extract_content(resp)
        except Exception:
            logger.exception("Bear researcher failed round %d", r)
            bear_arg = "Bear: Unable to generate argument."
        bear_history.append(bear_arg)

    return {
        "bull_history": bull_history,
        "bear_history": bear_history,
        "rounds": rounds,
        "full_debate": "\n\n".join(
            f"--- Round {i + 1} ---\nBULL: {bull_history[i]}\nBEAR: {bear_history[i]}" for i in range(rounds)
        ),
    }


JUDGE_PROMPT = """You are the Research Manager making the final trading decision for {pair}.
You observed a structured bull vs bear debate.

Evaluate argument QUALITY, not quantity. Judge each side on:
1. Evidence specificity: Did they cite concrete data points or speak in generalities?
2. Rebuttal strength: Did they address the opponent's best arguments or dodge them?
3. Internal consistency: Did their conclusion follow logically from their evidence?
4. Acknowledged weaknesses: Did they honestly address gaps in their own case?

Rules:
- If bull arguments are stronger → action: "long"
- If bear arguments are stronger → action: "short"
- ONLY choose "hold" if both sides are equally weak or data is truly insufficient
- Do NOT default to hold. Take a stance.
- Confidence should reflect how decisive the winner was, not a compromise.

LANGUAGE POLICY (MANDATORY):
- `reasoning` 字段必须使用 **简体中文** 输出。
- JSON keys 和 `action` 的 enum 值（`"long"` / `"short"` / `"hold"`）必须保持英文小写字面值。
- 数字保持裸数字。
- 示例：`{{"action": "short", "confidence": 0.65, "reasoning": "空方证据更扎实：funding 转负 + 4h RSI 顶背离 + 鲸鱼净流入交易所，多方仅靠 oversold 论据偏弱。"}}`

Respond ONLY with JSON: {{"action": "long|short|hold", "confidence": 0.0-1.0, "reasoning": "中文一句：哪一方胜出，关键证据是什么"}}"""


async def judge_debate(
    debate: dict,
    pair: str,
    model: str = "",
) -> dict:
    """Research manager judges the debate. Returns {action, confidence, reasoning}."""
    from cryptotrader.llm.prompt_cache import apply_cache_control, should_cache

    try:
        llm = create_llm(model=model, temperature=0.1)
        lc_msgs = [
            SystemMessage(content=JUDGE_PROMPT.format(pair=pair)),
            HumanMessage(content=debate["full_debate"]),
        ]
        if should_cache(model=model, role="debate"):
            lc_msgs = apply_cache_control(lc_msgs)
        resp = await llm.ainvoke(lc_msgs)
        text = extract_content(resp)
        data = _extract_json(text)
        action = data.get("action", "hold").strip().lower()
        if action in ("buy", "bullish"):
            action = "long"
        elif action in ("sell", "bearish"):
            action = "short"
        elif action not in ("long", "short", "hold"):
            action = "hold"
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
        return {
            "action": action,
            "confidence": confidence,
            "reasoning": data.get("reasoning", ""),
        }
    except Exception:
        logger.exception("Judge debate failed")
        return {"action": "hold", "confidence": 0.2, "reasoning": "Judge failed"}

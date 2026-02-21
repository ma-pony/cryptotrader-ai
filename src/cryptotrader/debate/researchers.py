"""Bull/Bear researchers: forced adversarial debate on top of analyst reports."""

from __future__ import annotations

import json
import logging

import litellm

logger = logging.getLogger(__name__)

BULL_SYSTEM = """You are a Bull Researcher. Your job is to build the strongest possible case for BUYING.
Even if the data looks bearish, find bullish signals: oversold conditions, capitulation signs, support levels, positive divergences.
You must advocate for the long side — this is your role in an adversarial debate.
Be specific: cite data points from the analyst reports. Don't just list generic reasons."""

BEAR_SYSTEM = """You are a Bear Researcher. Your job is to build the strongest possible case for SELLING/SHORTING.
Even if the data looks bullish, find bearish signals: overbought conditions, euphoria, resistance levels, negative divergences.
You must advocate for the short side — this is your role in an adversarial debate.
Be specific: cite data points from the analyst reports. Don't just list generic reasons."""

REBUTTAL_TEMPLATE = """The opposing analyst argued:
{opponent_argument}

Counter their points with specific evidence from the reports. Where are they wrong or cherry-picking?"""


def _format_reports(analyses: dict[str, dict]) -> str:
    parts = []
    for aid, a in analyses.items():
        parts.append(f"[{aid}] direction={a.get('direction')}, confidence={a.get('confidence')}\n{a.get('reasoning','')}")
    return "\n\n".join(parts)


async def run_debate(
    analyses: dict[str, dict],
    rounds: int = 2,
    model: str = "gpt-4o-mini",
) -> dict:
    """Run bull vs bear debate. Returns {bull_history, bear_history, rounds}."""
    reports = _format_reports(analyses)
    bull_history = []
    bear_history = []

    for r in range(rounds):
        # Bull argues (sees bear's last argument if any)
        bull_msgs = [{"role": "system", "content": BULL_SYSTEM}]
        bull_prompt = f"Analyst reports:\n{reports}"
        if bear_history:
            bull_prompt += f"\n\n{REBUTTAL_TEMPLATE.format(opponent_argument=bear_history[-1])}"
        elif r == 0:
            bull_prompt += "\n\nMake your opening bull case."
        bull_msgs.append({"role": "user", "content": bull_prompt})

        try:
            resp = await litellm.acompletion(model=model, messages=bull_msgs, temperature=0.3, max_tokens=512)
            bull_arg = resp.choices[0].message.content
        except Exception:
            logger.exception("Bull researcher failed round %d", r)
            bull_arg = "Bull: Unable to generate argument."
        bull_history.append(bull_arg)

        # Bear argues (sees bull's argument)
        bear_msgs = [{"role": "system", "content": BEAR_SYSTEM}]
        bear_prompt = f"Analyst reports:\n{reports}\n\n{REBUTTAL_TEMPLATE.format(opponent_argument=bull_arg)}"
        bear_msgs.append({"role": "user", "content": bear_prompt})

        try:
            resp = await litellm.acompletion(model=model, messages=bear_msgs, temperature=0.3, max_tokens=512)
            bear_arg = resp.choices[0].message.content
        except Exception:
            logger.exception("Bear researcher failed round %d", r)
            bear_arg = "Bear: Unable to generate argument."
        bear_history.append(bear_arg)

    return {
        "bull_history": bull_history,
        "bear_history": bear_history,
        "rounds": rounds,
        "full_debate": "\n\n".join(
            f"--- Round {i+1} ---\nBULL: {bull_history[i]}\nBEAR: {bear_history[i]}"
            for i in range(rounds)
        ),
    }


JUDGE_PROMPT = """You are the Research Manager making the final trading decision. You just observed a structured debate between a Bull and Bear analyst about {pair}.

Your job: evaluate argument QUALITY, not quantity. Which side presented more compelling, evidence-based reasoning?

Rules:
- If bull arguments are stronger → action: "long"
- If bear arguments are stronger → action: "short"  
- ONLY choose "hold" if both sides are equally weak or data is truly insufficient
- Do NOT default to hold. Take a stance.

Respond ONLY with JSON: {{"action": "long|short|hold", "confidence": 0.0-1.0, "reasoning": "one sentence explaining which side won and why"}}"""


async def judge_debate(
    debate: dict,
    pair: str,
    model: str = "gpt-4o-mini",
) -> dict:
    """Research manager judges the debate. Returns {action, confidence, reasoning}."""
    msgs = [
        {"role": "system", "content": JUDGE_PROMPT.format(pair=pair)},
        {"role": "user", "content": debate["full_debate"]},
    ]
    try:
        resp = await litellm.acompletion(model=model, messages=msgs, temperature=0.1, max_tokens=256)
        text = resp.choices[0].message.content
        data = json.loads(text[text.index("{"):text.rindex("}") + 1])
        return {
            "action": data.get("action", "hold"),
            "confidence": float(data.get("confidence", 0.5)),
            "reasoning": data.get("reasoning", ""),
        }
    except Exception:
        logger.exception("Judge debate failed")
        return {"action": "hold", "confidence": 0.2, "reasoning": "Judge failed"}

"""Cross-challenge prompt builder for multi-agent debate."""


def build_round1_prompt(agent_role: str, pair: str, data_summary: str, experience: str) -> str:
    return (
        f"You are a {agent_role} analyst.\n"
        f"Analyze {pair} for a trading decision.\n\n"
        f"Market data:\n{data_summary}\n\n"
        f"Your experience:\n{experience}\n\n"
        "Respond with: direction (bullish/bearish/neutral), confidence (0-1), "
        "reasoning, key_factors, and risk_flags."
    )


def build_challenge_prompt(
    agent_role: str,
    pair: str,
    own_analysis: dict,
    other_analyses: dict[str, dict],
) -> str:
    others = "\n\n".join(
        f"[{name}] direction={a.get('direction')}, confidence={a.get('confidence')}\n"
        f"reasoning: {a.get('reasoning')}"
        for name, a in other_analyses.items()
    )
    return (
        f"You are a {agent_role} analyst reviewing {pair}.\n\n"
        f"Your previous analysis:\n"
        f"direction={own_analysis.get('direction')}, confidence={own_analysis.get('confidence')}\n"
        f"reasoning: {own_analysis.get('reasoning')}\n\n"
        f"Other agents' analyses:\n{others}\n\n"
        "Challenge the weakest arguments from other agents. "
        "Then update your own direction, confidence, reasoning, key_factors, and risk_flags."
    )

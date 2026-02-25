"""Cross-challenge prompt builder for multi-agent debate."""


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
        "Evaluate each agent's argument on these criteria:\n"
        "1. Data support: Does their reasoning cite specific data points, or is it vague?\n"
        "2. Logic: Are there leaps from data to conclusion, or missing steps?\n"
        "3. Blind spots: What counter-evidence did they ignore?\n\n"
        "Challenge the weakest arguments with specific counter-evidence.\n"
        "Then revise YOUR OWN position honestly:\n"
        "- If others raised valid points you missed, adjust your direction or confidence.\n"
        "- If your original data still holds, maintain your stance — do not converge just for agreement.\n"
        "- Confidence should reflect the strength of evidence, not social pressure.\n\n"
        "Update your direction, confidence, reasoning, key_factors, and risk_flags."
    )

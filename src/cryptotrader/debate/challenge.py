"""Cross-challenge prompt builder for multi-agent debate.

Phase 4C: Show full agent analyses, demand new_findings, anti-convergence stance.
"""

import json


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
        "Output JSON with these fields:\n"
        '{"direction": "bullish|bearish|neutral", "confidence": 0.0-1.0, '
        '"reasoning": "2-3 sentences", "key_factors": [...], "risk_flags": [...], '
        '"new_findings": "cross-domain insight you discovered from reviewing other agents\' data"}'
    )

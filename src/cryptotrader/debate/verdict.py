"""Verdict generation from multi-agent debate results."""

from cryptotrader.models import TradeVerdict

_DIR_MAP = {"bullish": 1.0, "neutral": 0.0, "bearish": -1.0}


def make_verdict(
    analyses: dict[str, dict],
    divergence: float,
    divergence_threshold: float = 0.7,
) -> TradeVerdict:
    if divergence > divergence_threshold:
        return TradeVerdict(action="hold", divergence=divergence, reasoning="High divergence among agents")

    score = sum(
        a["confidence"] * _DIR_MAP.get(a["direction"], 0.0)
        for a in analyses.values()
    )
    position_scale = max(0.0, 1.0 - divergence)

    if score > 0:
        action = "long"
    elif score < 0:
        action = "short"
    else:
        action = "hold"

    # Confidence = average confidence of agents agreeing with majority direction
    if action != "hold":
        target_dir = "bullish" if action == "long" else "bearish"
        agreeing = [a["confidence"] for a in analyses.values()
                    if a["direction"] == target_dir]
        confidence = sum(agreeing) / len(agreeing) if agreeing else 0.0
    else:
        confidence = 0.0

    return TradeVerdict(
        action=action,
        confidence=confidence,
        position_scale=position_scale,
        divergence=divergence,
        reasoning=f"Weighted score={score:.3f}, divergence={divergence:.3f}",
    )

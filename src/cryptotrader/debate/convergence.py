"""Convergence detection for multi-agent debate."""

import statistics

_DIR_MAP = {"bullish": 1.0, "neutral": 0.0, "bearish": -1.0}


def compute_divergence(analyses: dict[str, dict]) -> float:
    values = [a["confidence"] * _DIR_MAP.get(a["direction"], 0.0) for a in analyses.values()]
    if len(values) < 2:
        return 0.0
    return statistics.pstdev(values)


def compute_consensus_strength(analyses: dict[str, dict]) -> tuple[float, float]:
    """Return (consensus_strength, mean_score).

    consensus_strength = abs(mean_score) * (1 - pstdev)
    - Strong consensus (same direction + low dispersion) → high value
    - Shared confusion (weak direction + low dispersion) → low abs(mean_score)
    """
    scores = [a["confidence"] * _DIR_MAP.get(a["direction"], 0.0) for a in analyses.values()]
    if len(scores) < 2:
        return 0.0, 0.0
    mean_score = sum(scores) / len(scores)
    dispersion = statistics.pstdev(scores)
    strength = abs(mean_score) * (1 - dispersion)
    return strength, mean_score


def check_convergence(
    divergence_scores: list[float],
    current_divergence: float,
    threshold: float = 0.1,
) -> bool:
    if not divergence_scores:
        return False
    last = divergence_scores[-1]
    # Both near zero — already converged
    if abs(last) < 1e-9:
        return abs(current_divergence) < 1e-9
    return abs(current_divergence - last) / abs(last) < threshold


def consensus_metrics(analyses: dict[str, dict]) -> dict[str, float]:
    strength, mean_score = compute_consensus_strength(analyses)
    return {
        "strength": strength,
        "mean_score": mean_score,
        "dispersion": compute_divergence(analyses),
    }


def debate_gate_decision(analyses: dict[str, dict], config) -> tuple[bool, str, dict[str, float]]:
    metrics = consensus_metrics(analyses)
    if not config.skip_debate:
        return False, "", metrics
    if metrics["strength"] > config.consensus_skip_threshold:
        return True, "consensus", metrics
    shared_confusion = (
        abs(metrics["mean_score"]) < config.confusion_skip_threshold
        and metrics["dispersion"] < config.confusion_max_dispersion
    )
    if shared_confusion:
        return True, "confusion", metrics
    return False, "", metrics

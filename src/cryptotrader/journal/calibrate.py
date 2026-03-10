"""Calibration, accuracy reporting, and bias detection for agent predictions.

Phase 4D: Generate meta-prompt corrections based on historical accuracy patterns.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cryptotrader.journal.store import JournalStore


async def accuracy_report(store: JournalStore, days: int = 30) -> dict[str, float]:
    """Check if each agent's direction matched the pnl outcome.

    Only directional calls (bullish/bearish) count — neutral calls are excluded
    from both numerator and denominator to avoid diluting accuracy.
    """
    commits = await store.log(limit=1000)
    cutoff = datetime.now(UTC) - timedelta(days=days)
    correct: dict[str, int] = {}
    directional: dict[str, int] = {}
    for dc in commits:
        if dc.timestamp < cutoff or dc.pnl is None:
            continue
        for agent_id, analysis in dc.analyses.items():
            if analysis.direction == "neutral":
                continue
            directional[agent_id] = directional.get(agent_id, 0) + 1
            pnl_positive = dc.pnl > 0
            bullish = analysis.direction == "bullish"
            if pnl_positive == bullish:
                correct[agent_id] = correct.get(agent_id, 0) + 1
    return {a: correct.get(a, 0) / directional[a] for a in directional}


async def calibrate_weights(store: JournalStore, days: int = 30) -> dict[str, float]:
    """Return normalized weights based on accuracy."""
    acc = await accuracy_report(store, days)
    if not acc:
        return {}
    total = sum(acc.values())
    if total == 0:
        return {a: 1.0 / len(acc) for a in acc}
    return {a: v / total for a, v in acc.items()}


async def detect_biases(store: JournalStore, days: int = 30) -> dict[str, dict]:
    """Analyze each agent's historical patterns to detect systematic biases.

    Returns per-agent bias report with:
    - overconfidence: avg confidence on wrong calls vs right calls
    - directional_bias: ratio of bullish vs bearish calls
    - neutral_rate: fraction of neutral calls (too many = not taking stances)
    - accuracy: overall accuracy rate
    """
    commits = await store.log(limit=1000)
    cutoff = datetime.now(UTC) - timedelta(days=days)

    # Per-agent tracking
    stats: dict[str, dict] = {}

    for dc in commits:
        if dc.timestamp < cutoff or dc.pnl is None:
            continue
        for agent_id, analysis in dc.analyses.items():
            if agent_id not in stats:
                stats[agent_id] = {
                    "total": 0,
                    "correct": 0,
                    "bullish": 0,
                    "bearish": 0,
                    "neutral": 0,
                    "conf_when_right": [],
                    "conf_when_wrong": [],
                }
            s = stats[agent_id]
            s["total"] += 1

            if analysis.direction == "bullish":
                s["bullish"] += 1
            elif analysis.direction == "bearish":
                s["bearish"] += 1
            else:
                s["neutral"] += 1
                continue

            pnl_positive = dc.pnl > 0
            was_right = pnl_positive == (analysis.direction == "bullish")
            if was_right:
                s["correct"] += 1
                s["conf_when_right"].append(analysis.confidence)
            else:
                s["conf_when_wrong"].append(analysis.confidence)

    biases: dict[str, dict] = {}
    for agent_id, s in stats.items():
        if s["total"] < 3:
            continue  # Not enough data
        directional = s["bullish"] + s["bearish"]
        biases[agent_id] = {
            "accuracy": s["correct"] / directional if directional > 0 else 0.0,
            "neutral_rate": s["neutral"] / s["total"],
            "bullish_rate": s["bullish"] / s["total"],
            "bearish_rate": s["bearish"] / s["total"],
            "avg_conf_when_right": (
                sum(s["conf_when_right"]) / len(s["conf_when_right"]) if s["conf_when_right"] else 0.0
            ),
            "avg_conf_when_wrong": (
                sum(s["conf_when_wrong"]) / len(s["conf_when_wrong"]) if s["conf_when_wrong"] else 0.0
            ),
            "sample_size": s["total"],
        }
    return biases


def _build_agent_warnings(b: dict) -> list[str]:
    """Build warning strings for a single agent's bias report."""
    warnings: list[str] = []

    # Overconfidence: high confidence on wrong calls
    if b["avg_conf_when_wrong"] > 0.65:
        warnings.append(
            f"OVERCONFIDENT — your avg confidence on wrong calls is {b['avg_conf_when_wrong']:.0%}. "
            "Lower your confidence unless evidence is exceptionally strong."
        )

    # Directional bias
    if b["bullish_rate"] > 0.65:
        warnings.append(
            f"BULLISH BIAS — {b['bullish_rate']:.0%} of your calls are bullish. "
            "Actively look for bearish evidence you might be ignoring."
        )
    elif b["bearish_rate"] > 0.65:
        warnings.append(
            f"BEARISH BIAS — {b['bearish_rate']:.0%} of your calls are bearish. "
            "Actively look for bullish evidence you might be ignoring."
        )

    # Neutral-defaulting
    if b["neutral_rate"] > 0.5:
        warnings.append(
            f"NEUTRAL DEFAULTING — {b['neutral_rate']:.0%} of your calls are neutral. "
            "Take a directional stance when data supports one. Neutral should be rare."
        )

    # Low accuracy
    if b["accuracy"] < 0.4 and b["sample_size"] >= 5:
        warnings.append(
            f"LOW ACCURACY — only {b['accuracy']:.0%} correct over {b['sample_size']} calls. "
            "Re-examine your analytical framework. What are you consistently getting wrong?"
        )

    return warnings


def generate_bias_correction(biases: dict[str, dict]) -> str:
    """Generate a combined meta-prompt correction string (legacy, all agents).

    Prefer generate_per_agent_corrections() for targeted per-agent injection.
    """
    if not biases:
        return ""

    lines: list[str] = []
    for agent_id, b in biases.items():
        warnings = _build_agent_warnings(b)
        if warnings:
            label = agent_id.replace("_agent", "").upper()
            lines.append(f"[{label}] " + " | ".join(warnings))

    if not lines:
        return ""
    return "Calibration warnings (based on your track record):\n" + "\n".join(lines)


def generate_per_agent_corrections(biases: dict[str, dict]) -> dict[str, str]:
    """Generate per-agent bias correction strings.

    Returns a dict mapping agent_id → correction text (only for agents with warnings).
    Each agent only sees their own calibration warnings.
    """
    if not biases:
        return {}

    result: dict[str, str] = {}
    for agent_id, b in biases.items():
        warnings = _build_agent_warnings(b)
        if warnings:
            result[agent_id] = "Calibration warnings (based on YOUR track record):\n" + " | ".join(warnings)
    return result


def generate_verdict_calibration(biases: dict[str, dict]) -> str:
    """Generate calibration context for the verdict AI.

    Tells the verdict which agents have been historically reliable vs unreliable.
    """
    if not biases:
        return ""

    lines: list[str] = []
    for agent_id, b in biases.items():
        if b["sample_size"] < 5:
            continue
        label = agent_id.replace("_agent", "").upper()
        acc_pct = f"{b['accuracy']:.0%}"
        conf_wrong = f"{b['avg_conf_when_wrong']:.0%}"
        note = ""
        if b["accuracy"] >= 0.6:
            note = " (reliable)"
        elif b["accuracy"] < 0.4:
            note = " (unreliable — weight their arguments less)"
        if b["avg_conf_when_wrong"] > 0.65:
            note += " (tends to be overconfident when wrong)"
        lines.append(f"- {label}: {acc_pct} accuracy, avg confidence on wrong calls: {conf_wrong}{note}")

    if not lines:
        return ""
    return "AGENT TRACK RECORDS (factor this into your evaluation):\n" + "\n".join(lines)

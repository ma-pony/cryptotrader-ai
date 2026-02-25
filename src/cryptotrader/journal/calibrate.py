"""Calibration and accuracy reporting for agent predictions."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from cryptotrader.journal.store import JournalStore


async def accuracy_report(store: JournalStore, days: int = 30) -> dict[str, float]:
    """Check if each agent's direction matched the pnl outcome."""
    commits = await store.log(limit=1000)
    cutoff = datetime.now(UTC) - timedelta(days=days)
    correct: dict[str, int] = {}
    total: dict[str, int] = {}
    for dc in commits:
        if dc.timestamp < cutoff or dc.pnl is None:
            continue
        for agent_id, analysis in dc.analyses.items():
            total[agent_id] = total.get(agent_id, 0) + 1
            # Neutral predictions are not counted as correct — agents
            # that never take a stance shouldn't be rewarded
            if analysis.direction == "neutral":
                continue
            pnl_positive = dc.pnl > 0
            bullish = analysis.direction == "bullish"
            if pnl_positive == bullish:
                correct[agent_id] = correct.get(agent_id, 0) + 1
    return {a: correct.get(a, 0) / total[a] for a in total}


async def calibrate_weights(store: JournalStore, days: int = 30) -> dict[str, float]:
    """Return normalized weights based on accuracy."""
    acc = await accuracy_report(store, days)
    if not acc:
        return {}
    total = sum(acc.values())
    if total == 0:
        return {a: 1.0 / len(acc) for a in acc}
    return {a: v / total for a, v in acc.items()}

"""spec 018 — Pareto frontier ranking for PatternRecord.

FR-Z14: rank_rules(rules: list[PatternRecord]) -> list[PatternRecord]
Dual-objective Pareto frontier:
  obj1 = win_rate  (from pnl_track)
  obj2 = confidence_proxy = importance * maturity_weight
Output: Pareto layers sorted by dominance, within each layer by win_rate * confidence_proxy desc.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cryptotrader.agents.skills.schema import PatternRecord

logger = logging.getLogger(__name__)

# maturity_weight (Decision 2 / FR-Z14)
_MATURITY_WEIGHT: dict[str, float] = {
    "active": 1.0,
    "probationary": 0.6,
    "observed": 0.3,
    "deprecated": 0.0,
    "archived": 0.0,
}


def _win_rate(rule: PatternRecord) -> float:
    """Compute win_rate: successes / (successes + losses). Default 0.5 for 0 trades."""
    pt = rule.pnl_track
    wins = getattr(pt, "wins", 0)
    cases = getattr(pt, "cases", 0)
    if cases == 0:
        return 0.5  # no trade data → neutral
    return wins / cases


def _confidence_proxy(rule: PatternRecord) -> float:
    """importance * maturity_weight."""
    w = _MATURITY_WEIGHT.get(rule.maturity, 0.0)
    return rule.importance * w


def _dominates(a_wr: float, a_cp: float, b_wr: float, b_cp: float) -> bool:
    """Return True if (a_wr, a_cp) Pareto-dominates (b_wr, b_cp).

    a dominates b iff a is >= b on both objectives and strictly > on at least one.
    """
    return (a_wr >= b_wr and a_cp >= b_cp) and (a_wr > b_wr or a_cp > b_cp)


def rank_rules(rules: list[PatternRecord]) -> list[PatternRecord]:
    """Rank rules by Pareto frontier layers, then by win_rate * confidence_proxy within each layer.

    FR-Z14: dual-objective (win_rate, confidence_proxy).
    Returns a new list sorted from best to worst.
    """
    if not rules:
        return []

    # Pre-compute scores
    scored = [(r, _win_rate(r), _confidence_proxy(r)) for r in rules]

    # Assign Pareto layer (layer 0 = non-dominated)
    n = len(scored)
    layer = [0] * n

    for i in range(n):
        _, wr_i, cp_i = scored[i]
        for j in range(n):
            if i == j:
                continue
            _, wr_j, cp_j = scored[j]
            if _dominates(wr_j, cp_j, wr_i, cp_i):
                layer[i] += 1  # i is dominated by j

    # Collect into layer groups
    layer_groups: dict[int, list[tuple[PatternRecord, float, float]]] = {}
    for i, (r, wr, cp) in enumerate(scored):
        lyr = layer[i]
        if lyr not in layer_groups:
            layer_groups[lyr] = []
        layer_groups[lyr].append((r, wr, cp))

    # Sort within each layer by win_rate * confidence_proxy desc
    result: list[PatternRecord] = []
    for lyr in sorted(layer_groups.keys()):
        group = sorted(layer_groups[lyr], key=lambda x: x[1] * x[2], reverse=True)
        result.extend(r for r, _, _ in group)

    return result

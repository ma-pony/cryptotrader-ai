"""spec 018 Pareto frontier unit tests — tests/test_pareto.py

SC-Z7: >= 6 use cases PASS.
"""

from __future__ import annotations

from cryptotrader.agents.skills.schema import PatternRecord, PnLTrack
from cryptotrader.learning.evolution.pareto import _confidence_proxy, _win_rate, rank_rules


def _make_rule(
    name: str = "rule",
    agent: str = "tech",
    maturity: str = "active",
    wins: int = 0,
    cases: int = 0,
    importance: float = 0.5,
) -> PatternRecord:
    pt = PnLTrack(cases=cases, wins=wins, win_rate=wins / cases if cases > 0 else 0.0)
    return PatternRecord(
        name=name,
        agent=agent,
        description="desc",
        body="body",
        maturity=maturity,  # type: ignore[arg-type]
        pnl_track=pt,
        importance=importance,
    )


# ── basic correctness ─────────────────────────────────────────────────────────


def test_single_rule_returned_unchanged():
    """T016(a): single rule is returned as-is."""
    r = _make_rule(wins=5, cases=10)
    result = rank_rules([r])
    assert len(result) == 1
    assert result[0] is r


def test_two_rules_dominating():
    """T016(b): one rule dominates the other -> dominant rule first."""
    # r_good: higher win_rate AND higher confidence -> dominates
    r_good = _make_rule("good", maturity="active", wins=8, cases=10, importance=0.9)
    r_bad = _make_rule("bad", maturity="observed", wins=2, cases=10, importance=0.3)

    result = rank_rules([r_bad, r_good])
    assert result[0].name == "good"
    assert result[1].name == "bad"


def test_two_rules_pareto_incomparable():
    """T016(c): two rules Pareto-incomparable -> same layer, sorted by product."""
    # r_a: high win_rate, low confidence
    r_a = _make_rule("a", maturity="observed", wins=9, cases=10, importance=0.9)
    # r_b: low win_rate, high confidence
    r_b = _make_rule("b", maturity="active", wins=3, cases=10, importance=0.9)

    result = rank_rules([r_a, r_b])
    # Both in layer 0 (neither dominates the other), sorted by wr*cp desc
    # r_a: wr=0.9, cp=0.9*0.3=0.27, product=0.243
    # r_b: wr=0.3, cp=0.9*1.0=0.9, product=0.27
    assert result[0].name == "b"  # product 0.27 > 0.243


def test_high_confidence_beats_high_winrate_observed():
    """T016(d): active rule with lower win_rate may beat observed rule with higher win_rate."""
    # active rule: wr=0.4, cp=0.5*1.0=0.5, product=0.2
    # observed rule: wr=1.0, cp=0.5*0.3=0.15, product=0.15
    r_active = _make_rule("active_rule", maturity="active", wins=4, cases=10, importance=0.5)
    r_obs = _make_rule("obs_rule", maturity="observed", wins=10, cases=10, importance=0.5)

    # r_obs dominates on win_rate axis but not confidence;
    # r_active dominates on confidence axis but not win_rate -> incomparable
    # In layer-0, sorted by product:
    result = rank_rules([r_obs, r_active])
    assert result[0].name == "active_rule"  # 0.2 > 0.15


def test_zero_trade_rule_default_winrate():
    """T016(e): rule with 0 trades has default win_rate=0.5."""
    r_no_trades = _make_rule("empty", wins=0, cases=0, importance=0.5, maturity="active")
    wr = _win_rate(r_no_trades)
    assert wr == 0.5


def test_five_rules_mixed_layers():
    """T016(f): 5 rules with mixed dominance produce correct ordering."""
    # Layer 0 (non-dominated): r1 (wr=0.9, cp=0.9)
    r1 = _make_rule("r1", maturity="active", wins=9, cases=10, importance=0.9)
    # Layer 1: r2 (wr=0.7, cp=0.7) dominated by r1
    r2 = _make_rule("r2", maturity="active", wins=7, cases=10, importance=0.7)
    # Layer 1: r3 (wr=0.8, cp=0.6) dominated by r1
    r3 = _make_rule("r3", maturity="active", wins=8, cases=10, importance=0.6)
    # Layer 2: r4 (wr=0.5, cp=0.5) dominated by r1, r2, r3
    r4 = _make_rule("r4", maturity="active", wins=5, cases=10, importance=0.5)
    # deprecated: cp=0 (always worst layer)
    r5 = _make_rule("r5", maturity="deprecated", wins=10, cases=10, importance=0.9)

    result = rank_rules([r5, r4, r3, r2, r1])
    assert result[0].name == "r1"
    # r5 with cp=0 should be last or very late
    assert result[-1].name in ("r5", "r4")


def test_empty_rules_returns_empty():
    """Empty list returns empty list."""
    assert rank_rules([]) == []


def test_confidence_proxy_values():
    """confidence_proxy = importance * maturity_weight."""
    r_active = _make_rule(maturity="active", importance=0.8)
    r_prob = _make_rule(maturity="probationary", importance=0.8)
    r_obs = _make_rule(maturity="observed", importance=0.8)
    r_dep = _make_rule(maturity="deprecated", importance=0.8)
    r_arch = _make_rule(maturity="archived", importance=0.8)

    assert abs(_confidence_proxy(r_active) - 0.8) < 1e-9
    assert abs(_confidence_proxy(r_prob) - 0.48) < 1e-9
    assert abs(_confidence_proxy(r_obs) - 0.24) < 1e-9
    assert _confidence_proxy(r_dep) == 0.0
    assert _confidence_proxy(r_arch) == 0.0

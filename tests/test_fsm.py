"""spec 018 FSM unit tests — tests/test_fsm.py

SC-Z6: >= 12 use cases PASS.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from cryptotrader.agents.skills.schema import PatternRecord, PnLTrack
from cryptotrader.learning.evolution.fsm import (
    _MAX_BODY_LINES,
    _OBSERVED_PROMO_SUCCESSES,
    _PROBATIONARY_PROMO_DAYS,
    evaluate_transitions,
)


def _make_rule(
    name: str = "test_rule",
    agent: str = "tech",
    maturity: str = "observed",
    wins: int = 0,
    cases: int = 0,
    fundamental_failure_streak: int = 0,
    manually_edited: bool = False,
    last_modified_at: datetime | None = None,
    body: str = "## Rule\nBody content",
    description: str = "test description",
) -> PatternRecord:
    """Build a PatternRecord fixture for FSM tests."""
    if last_modified_at is None:
        last_modified_at = datetime.now(UTC) - timedelta(days=10)
    pt = PnLTrack(cases=cases, wins=wins, win_rate=wins / cases if cases > 0 else 0.0)
    return PatternRecord(
        name=name,
        agent=agent,
        description=description,
        body=body,
        maturity=maturity,  # type: ignore[arg-type]
        pnl_track=pt,
        fundamental_failure_streak=fundamental_failure_streak,
        manually_edited=manually_edited,
        last_modified_at=last_modified_at,
        importance=0.5,
        access_count=0,
    )


# ── observed state tests ───────────────────────────────────────────────────────


def test_observed_no_change_insufficient_successes():
    """T014(a): observed + successes < 3 -> no change."""
    rule = _make_rule(maturity="observed", wins=2, cases=2)
    result = evaluate_transitions(rule)
    assert result is None


def test_observed_promotes_to_probationary():
    """T014(b): observed + successes >= 3 -> probationary."""
    rule = _make_rule(maturity="observed", wins=3, cases=5)
    result = evaluate_transitions(rule)
    assert result is not None
    assert result.maturity == "probationary"


def test_observed_exact_threshold():
    """observed + successes == 3 exactly promotes."""
    rule = _make_rule(maturity="observed", wins=_OBSERVED_PROMO_SUCCESSES, cases=3)
    result = evaluate_transitions(rule)
    assert result is not None
    assert result.maturity == "probationary"


# ── probationary state tests ───────────────────────────────────────────────────


def test_probationary_promotes_after_5_cycle_days():
    """T014(c): probationary + 5+ days without modification -> active."""
    old_time = datetime.now(UTC) - timedelta(days=_PROBATIONARY_PROMO_DAYS + 1)
    rule = _make_rule(maturity="probationary", last_modified_at=old_time)
    result = evaluate_transitions(rule)
    assert result is not None
    assert result.maturity == "active"


def test_probationary_promotes_after_3_days():
    """T014(d): probationary + exactly 3 days without modification -> active."""
    old_time = datetime.now(UTC) - timedelta(days=_PROBATIONARY_PROMO_DAYS)
    rule = _make_rule(maturity="probationary", last_modified_at=old_time)
    result = evaluate_transitions(rule)
    assert result is not None
    assert result.maturity == "active"


def test_probationary_no_change_recently_modified():
    """probationary + recently modified (<3 days) -> no change."""
    recent_time = datetime.now(UTC) - timedelta(days=1)
    rule = _make_rule(maturity="probationary", last_modified_at=recent_time)
    result = evaluate_transitions(rule)
    assert result is None


def test_probationary_no_change_empty_description():
    """T014(e): probationary + missing description (frontmatter incomplete) -> no change."""
    old_time = datetime.now(UTC) - timedelta(days=10)
    rule = _make_rule(maturity="probationary", last_modified_at=old_time, description="")
    result = evaluate_transitions(rule)
    assert result is None


def test_probationary_no_change_body_too_long():
    """T014(l): probationary + body > 300 lines -> no promotion to active."""
    old_time = datetime.now(UTC) - timedelta(days=10)
    long_body = "\n".join(f"line {i}" for i in range(_MAX_BODY_LINES + 1))
    rule = _make_rule(maturity="probationary", last_modified_at=old_time, body=long_body)
    result = evaluate_transitions(rule)
    assert result is None


def test_probationary_exactly_300_body_lines():
    """T014(f): probationary + body == 300 lines -> can promote."""
    old_time = datetime.now(UTC) - timedelta(days=10)
    exact_body = "\n".join(f"line {i}" for i in range(_MAX_BODY_LINES))
    rule = _make_rule(maturity="probationary", last_modified_at=old_time, body=exact_body)
    result = evaluate_transitions(rule)
    assert result is not None
    assert result.maturity == "active"


# ── active state tests ────────────────────────────────────────────────────────


def test_active_no_change_low_streak():
    """T014(g): active + fundamental_streak < 3 -> no change."""
    rule = _make_rule(maturity="active", fundamental_failure_streak=2)
    result = evaluate_transitions(rule)
    assert result is None


def test_active_archives_on_streak_3():
    """T014(h): active + fundamental_streak == 3 -> archived."""
    rule = _make_rule(maturity="active", fundamental_failure_streak=3)
    result = evaluate_transitions(rule)
    assert result is not None
    assert result.maturity == "archived"


def test_active_archives_on_streak_above_3():
    """active + fundamental_streak > 3 -> archived."""
    rule = _make_rule(maturity="active", fundamental_failure_streak=5)
    result = evaluate_transitions(rule)
    assert result is not None
    assert result.maturity == "archived"


def test_active_demotes_on_manually_edited():
    """T014(i): active + manually_edited=True -> probationary (demotion)."""
    rule = _make_rule(maturity="active", manually_edited=True)
    result = evaluate_transitions(rule)
    assert result is not None
    assert result.maturity == "probationary"
    assert result.manually_edited is False  # flag cleared after demotion


# ── terminal state tests ──────────────────────────────────────────────────────


def test_deprecated_is_terminal():
    """T014(j): deprecated -> no change (terminal state)."""
    rule = _make_rule(maturity="deprecated")
    result = evaluate_transitions(rule)
    assert result is None


def test_archived_is_terminal():
    """T014(k): archived -> no change (terminal state)."""
    rule = _make_rule(maturity="archived")
    result = evaluate_transitions(rule)
    assert result is None


# ── Transition dataclass ──────────────────────────────────────────────────────


def test_build_transition_records_correct_trigger():
    """build_transition generates correct triggered_by strings."""
    from cryptotrader.learning.evolution.fsm import build_transition

    old = _make_rule(maturity="observed")
    new = evaluate_transitions(_make_rule(maturity="observed", wins=3, cases=3))
    assert new is not None
    t = build_transition(old, new)
    assert t.old_state == "observed"
    assert t.new_state == "probationary"
    assert t.triggered_by == "pnl_threshold"

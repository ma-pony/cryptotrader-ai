"""spec 018 / 021 FSM unit tests — merged win_rate + 5-signal FSM.

Tests cover:
  - observed → probationary (cases ≥ 5 + win_rate ≥ 0.60)
  - probationary → active (time + quality + cases ≥ 15 + win_rate ≥ 0.65)
  - probationary → archived (PnL collapse: cases ≥ 10 + win_rate < 0.40)
  - active → archived (fundamental_streak OR PnL collapse)
  - active → probationary (manually_edited)
  - deprecated / archived terminal
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from cryptotrader.agents.skills.schema import PatternRecord, PnLTrack
from cryptotrader.learning.evolution.fsm import (
    _ARCHIVED_FUNDAMENTAL_STREAK,
    _ARCHIVED_MAX_WIN_RATE,
    _ARCHIVED_MIN_CASES,
    _MAX_BODY_LINES,
    _OBSERVED_MIN_CASES,
    _OBSERVED_MIN_WIN_RATE,
    _PROBATIONARY_MIN_CASES,
    _PROBATIONARY_MIN_WIN_RATE,
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


# ── observed → probationary ───────────────────────────────────────────────────


def test_observed_no_change_insufficient_cases():
    """observed + cases<5 → no change（win_rate=1.0 也不行）。"""
    rule = _make_rule(maturity="observed", wins=4, cases=4)
    assert evaluate_transitions(rule) is None


def test_observed_no_change_low_win_rate():
    """observed + cases≥5 但 win_rate<0.60 → no change。"""
    rule = _make_rule(maturity="observed", wins=2, cases=5)  # 0.40 win_rate
    assert evaluate_transitions(rule) is None


def test_observed_promotes_at_threshold():
    """observed + cases=5 + win_rate=0.60（边界）→ probationary。"""
    rule = _make_rule(maturity="observed", wins=3, cases=5)  # 0.60
    result = evaluate_transitions(rule)
    assert result is not None
    assert result.maturity == "probationary"


def test_observed_promotes_clearly():
    """observed + cases≥5 + win_rate≥0.60 → probationary。"""
    rule = _make_rule(maturity="observed", wins=5, cases=6)  # 0.83
    result = evaluate_transitions(rule)
    assert result is not None
    assert result.maturity == "probationary"


# ── probationary → active ─────────────────────────────────────────────────────


def _good_prob_rule(**overrides) -> PatternRecord:
    """Probationary rule meeting time + cases + win_rate (default well-formed)."""
    defaults: dict = {
        "maturity": "probationary",
        "wins": 10,
        "cases": _PROBATIONARY_MIN_CASES,  # 15
        "last_modified_at": datetime.now(UTC) - timedelta(days=_PROBATIONARY_PROMO_DAYS + 1),
    }
    defaults.update(overrides)
    # 15 cases @ default wins=10 ⇒ win_rate ≈ 0.67 ≥ 0.65
    return _make_rule(**defaults)


def test_probationary_promotes_when_all_signals_pass():
    """time≥3d + frontmatter + body≤300 + cases≥15 + win_rate≥0.65 → active。"""
    rule = _good_prob_rule()
    result = evaluate_transitions(rule)
    assert result is not None
    assert result.maturity == "active"


def test_probationary_no_change_recently_modified():
    """time<3d → 不晋升（即使 PnL 达标）。"""
    rule = _good_prob_rule(last_modified_at=datetime.now(UTC) - timedelta(days=1))
    assert evaluate_transitions(rule) is None


def test_probationary_no_change_empty_description():
    """frontmatter 缺字段 → 不晋升。"""
    rule = _good_prob_rule(description="")
    assert evaluate_transitions(rule) is None


def test_probationary_no_change_body_too_long():
    """body>300 行 → 不晋升。"""
    long_body = "\n".join(f"line {i}" for i in range(_MAX_BODY_LINES + 1))
    rule = _good_prob_rule(body=long_body)
    assert evaluate_transitions(rule) is None


def test_probationary_no_change_too_few_cases():
    """cases<15 → 不晋升（即使 win_rate 高）。"""
    rule = _good_prob_rule(wins=10, cases=14)  # win_rate=0.71 高，但 cases 不足
    assert evaluate_transitions(rule) is None


def test_probationary_no_change_low_win_rate():
    """win_rate<0.65 → 不晋升（即使 cases 足）。"""
    rule = _good_prob_rule(wins=9, cases=15)  # win_rate=0.60 不足
    assert evaluate_transitions(rule) is None


def test_probationary_exactly_300_body_lines_promotes():
    """body 恰好 300 行 + 其他都达标 → 可晋升。"""
    exact_body = "\n".join(f"line {i}" for i in range(_MAX_BODY_LINES))
    rule = _good_prob_rule(body=exact_body)
    result = evaluate_transitions(rule)
    assert result is not None
    assert result.maturity == "active"


# ── probationary → archived（PnL 坍塌）────────────────────────────────────────


def test_probationary_archives_on_pnl_collapse():
    """probationary + cases≥10 + win_rate<0.40 → archived（不再卡在 probationary）。"""
    rule = _good_prob_rule(
        wins=2,
        cases=_ARCHIVED_MIN_CASES,  # 10
    )  # win_rate=0.20 < 0.40
    result = evaluate_transitions(rule)
    assert result is not None
    assert result.maturity == "archived"


def test_probationary_no_archive_if_cases_below_threshold():
    """probationary + win_rate<0.40 但 cases<10 → 不归档（样本不足）。"""
    rule = _good_prob_rule(wins=2, cases=9, last_modified_at=datetime.now(UTC))
    # cases=9 → 既不满足升 active 的 cases≥15，也不触发归档 cases≥10
    assert evaluate_transitions(rule) is None


# ── active → archived / probationary ──────────────────────────────────────────


def test_active_no_change_when_healthy():
    """active + streak<3 + cases足但 win_rate 正常 → 不变。"""
    rule = _make_rule(maturity="active", wins=8, cases=10, fundamental_failure_streak=2)
    # win_rate=0.80 ≥ 0.40 不归档
    assert evaluate_transitions(rule) is None


def test_active_archives_on_fundamental_streak():
    """active + fundamental_streak≥3 → archived。"""
    rule = _make_rule(maturity="active", fundamental_failure_streak=_ARCHIVED_FUNDAMENTAL_STREAK)
    result = evaluate_transitions(rule)
    assert result is not None
    assert result.maturity == "archived"


def test_active_archives_on_pnl_collapse():
    """active + cases≥10 + win_rate<0.40 → archived（性能坍塌路径）。"""
    rule = _make_rule(maturity="active", wins=3, cases=_ARCHIVED_MIN_CASES)  # 0.30
    result = evaluate_transitions(rule)
    assert result is not None
    assert result.maturity == "archived"


def test_active_demotes_on_manually_edited():
    """active + manually_edited=True → probationary（需重过 prob 期）。"""
    rule = _make_rule(maturity="active", manually_edited=True, wins=8, cases=10)
    result = evaluate_transitions(rule)
    assert result is not None
    assert result.maturity == "probationary"
    assert result.manually_edited is False


# ── 终态 ─────────────────────────────────────────────────────────────────────


def test_deprecated_is_terminal():
    """deprecated 是终态（向后兼容），不再发生转移。"""
    rule = _make_rule(maturity="deprecated")
    assert evaluate_transitions(rule) is None


def test_archived_is_terminal():
    """archived 是终态。"""
    rule = _make_rule(maturity="archived")
    assert evaluate_transitions(rule) is None


# ── Transition build ─────────────────────────────────────────────────────────


def test_build_transition_records_correct_trigger():
    """build_transition: observed→probationary triggered_by 'pnl_threshold'。"""
    from cryptotrader.learning.evolution.fsm import build_transition

    old = _make_rule(maturity="observed", wins=5, cases=5)
    new = evaluate_transitions(old)
    assert new is not None
    t = build_transition(old, new)
    assert t.old_state == "observed"
    assert t.new_state == "probationary"
    assert t.triggered_by == "pnl_threshold"


def test_build_transition_active_to_archived_fundamental():
    """active → archived 由 fundamental_streak 触发。"""
    from cryptotrader.learning.evolution.fsm import build_transition

    old = _make_rule(maturity="active", fundamental_failure_streak=3)
    new = evaluate_transitions(old)
    assert new is not None
    t = build_transition(old, new)
    assert t.triggered_by == "fundamental_streak"


def test_build_transition_active_to_archived_pnl_collapse():
    """active → archived 由 PnL 坍塌触发。"""
    from cryptotrader.learning.evolution.fsm import build_transition

    old = _make_rule(maturity="active", wins=2, cases=10)  # 0.20
    new = evaluate_transitions(old)
    assert new is not None
    t = build_transition(old, new)
    assert t.triggered_by == "pnl_collapse"


# constants must be reachable from public-ish names (used by callers / tests)
def test_constants_exist():
    """常量导出存在性 sanity（保证下游能引用）。"""
    assert _OBSERVED_MIN_CASES == 5
    assert _OBSERVED_MIN_WIN_RATE == 0.60
    assert _PROBATIONARY_MIN_CASES == 15
    assert _PROBATIONARY_MIN_WIN_RATE == 0.65
    assert _ARCHIVED_MIN_CASES == 10
    assert _ARCHIVED_MAX_WIN_RATE == 0.40
    assert _ARCHIVED_FUNDAMENTAL_STREAK == 3

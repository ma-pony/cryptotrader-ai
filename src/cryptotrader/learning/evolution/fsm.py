"""spec 018 / 021 — 唯一 Maturity FSM（合并旧 win_rate 路径 + 5 信号路径）。

State transitions:
  observed     -> probationary  : cases >= 5 AND win_rate >= 0.60                       (signal 1: PnL gate)
  probationary -> active        : (now - last_modified_at) >= 3 days                    (signal 2: time gate)
                                  AND frontmatter filled AND body <= 300 lines          (signal 3: quality gate)
                                  AND cases >= 15 AND win_rate >= 0.65                  (signal 4: PnL gate)
  active       -> archived      : fundamental_failure_streak >= 3                       (signal 5a: IVE failure)
                                  OR (cases >= 10 AND win_rate < 0.40)                  (signal 5b: PnL collapse)
  active       -> probationary  : manually_edited == True                               (signal 6: human edit reset)
  deprecated / archived         : terminal states, no transition

设计原则（spec 021）：
- 唯一的 FSM 入口在此文件，删除 memory.py:_advance_maturity；
- 同时使用「绝对 win 数 + win_rate 门槛」（避免 3 wins / 100 losses 也升 active）；
- 同时使用「IVE fundamental 失败连续 3 次」+「win_rate 坍塌」双归档路径；
- 取消 `deprecated` 中间态，性能坍塌直接 `archived`。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cryptotrader.agents.skills.schema import PatternRecord

logger = logging.getLogger(__name__)

# ── PnL 门槛 ──
_OBSERVED_MIN_CASES = 5
_OBSERVED_MIN_WIN_RATE = 0.60
_PROBATIONARY_MIN_CASES = 15
_PROBATIONARY_MIN_WIN_RATE = 0.65
_ARCHIVED_MIN_CASES = 10
_ARCHIVED_MAX_WIN_RATE = 0.40  # 低于此 + cases ≥ 10 视为 PnL 坍塌

# ── 时间 / 质量门槛 ──
_PROBATIONARY_PROMO_DAYS = 3
_ARCHIVED_FUNDAMENTAL_STREAK = 3
_REQUIRED_FIELDS = ("name", "agent", "description", "maturity", "version")
_MAX_BODY_LINES = 300


@dataclass
class Transition:
    """Record of a state transition triggered by the FSM."""

    rule_id: str
    agent_id: str
    old_state: str
    new_state: str
    triggered_by: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


def _frontmatter_filled(rule: PatternRecord) -> bool:
    """Check all required fields are non-empty."""
    for f in _REQUIRED_FIELDS:
        val = getattr(rule, f, None)
        if val is None or val == "" or val == 0:
            return False
    return True


def _body_within_limit(rule: PatternRecord) -> bool:
    """Check body line count <= 300."""
    return len(rule.body.splitlines()) <= _MAX_BODY_LINES


def _time_since_modified(rule: PatternRecord) -> timedelta:
    """Calculate elapsed time since last_modified_at."""
    now = datetime.now(UTC)
    last_mod = rule.last_modified_at
    if last_mod.tzinfo is None:
        last_mod = last_mod.replace(tzinfo=UTC)
    return now - last_mod


def _pnl_stats(rule: PatternRecord) -> tuple[int, float]:
    """Return (cases, win_rate) tuple from PnLTrack, with 0/0 defaulting to (0, 0.0)."""
    track = rule.pnl_track
    cases = getattr(track, "cases", 0) or 0
    win_rate = getattr(track, "win_rate", 0.0) or 0.0
    return cases, float(win_rate)


def evaluate_transitions(rule: PatternRecord) -> PatternRecord | None:
    """Evaluate FSM transitions for a PatternRecord.

    Returns updated PatternRecord if state changed, or None if no transition.
    Does NOT persist changes — caller is responsible for writing back to file.
    """
    from dataclasses import replace

    maturity = rule.maturity

    # Terminal states
    if maturity in ("deprecated", "archived"):
        return None

    now = datetime.now(UTC)
    cases, win_rate = _pnl_stats(rule)

    if maturity == "observed":
        if cases >= _OBSERVED_MIN_CASES and win_rate >= _OBSERVED_MIN_WIN_RATE:
            logger.debug(
                "FSM: %s observed->probationary (cases=%d, win_rate=%.2f)",
                rule.name,
                cases,
                win_rate,
            )
            return replace(rule, maturity="probationary", last_modified_at=now)
        return None

    if maturity == "probationary":
        # 性能坍塌直接 archived（避免长期卡在 probationary 浪费 prompt 空间）
        if cases >= _ARCHIVED_MIN_CASES and win_rate < _ARCHIVED_MAX_WIN_RATE:
            logger.debug(
                "FSM: %s probationary->archived (PnL collapse: cases=%d, win_rate=%.2f)",
                rule.name,
                cases,
                win_rate,
            )
            return replace(rule, maturity="archived", last_modified_at=now)

        elapsed = _time_since_modified(rule)
        if (
            elapsed >= timedelta(days=_PROBATIONARY_PROMO_DAYS)
            and _frontmatter_filled(rule)
            and _body_within_limit(rule)
            and cases >= _PROBATIONARY_MIN_CASES
            and win_rate >= _PROBATIONARY_MIN_WIN_RATE
        ):
            logger.debug(
                "FSM: %s probationary->active (elapsed=%s, cases=%d, win_rate=%.2f)",
                rule.name,
                elapsed,
                cases,
                win_rate,
            )
            return replace(rule, maturity="active", last_modified_at=now)
        return None

    if maturity == "active":
        # Path 1: IVE fundamental failure streak
        if rule.fundamental_failure_streak >= _ARCHIVED_FUNDAMENTAL_STREAK:
            logger.debug(
                "FSM: %s active->archived (fundamental_streak=%d)",
                rule.name,
                rule.fundamental_failure_streak,
            )
            return replace(rule, maturity="archived", last_modified_at=now)
        # Path 2: PnL collapse
        if cases >= _ARCHIVED_MIN_CASES and win_rate < _ARCHIVED_MAX_WIN_RATE:
            logger.debug(
                "FSM: %s active->archived (PnL collapse: cases=%d, win_rate=%.2f)",
                rule.name,
                cases,
                win_rate,
            )
            return replace(rule, maturity="archived", last_modified_at=now)
        # Demote on human edit (re-validation needed)
        if rule.manually_edited:
            logger.debug("FSM: %s active->probationary (manually_edited)", rule.name)
            return replace(
                rule,
                maturity="probationary",
                manually_edited=False,
                last_modified_at=now,
            )
        return None

    return None


def build_transition(
    old_rule: PatternRecord,
    new_rule: PatternRecord,
) -> Transition:
    """Build a Transition record from old/new PatternRecord pair."""
    triggered_by: str
    if old_rule.maturity == "observed" and new_rule.maturity == "probationary":
        triggered_by = "pnl_threshold"
    elif old_rule.maturity == "probationary" and new_rule.maturity == "active":
        triggered_by = "time_elapsed_and_pnl"
    elif old_rule.maturity == "probationary" and new_rule.maturity == "archived":
        triggered_by = "pnl_collapse"
    elif old_rule.maturity == "active" and new_rule.maturity == "archived":
        # 调用方可通过 fundamental_failure_streak 区分子原因
        triggered_by = (
            "fundamental_streak"
            if old_rule.fundamental_failure_streak >= _ARCHIVED_FUNDAMENTAL_STREAK
            else "pnl_collapse"
        )
    elif old_rule.maturity == "active" and new_rule.maturity == "probationary":
        triggered_by = "reflect_modified"
    else:
        triggered_by = "unknown"

    return Transition(
        rule_id=f"{old_rule.agent}::{old_rule.name}",
        agent_id=old_rule.agent,
        old_state=old_rule.maturity,
        new_state=new_rule.maturity,
        triggered_by=triggered_by,
        timestamp=datetime.now(UTC),
    )

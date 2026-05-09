"""spec 018 — 5-signal Maturity FSM.

FR-Z11: evaluate_transitions(rule: PatternRecord) -> PatternRecord | None
State transitions:
  observed -> probationary  : pnl_track.successes >= 3
  probationary -> active    : (now - last_modified_at) >= 3 days AND frontmatter filled AND body <= 300 lines
  active -> archived        : fundamental_failure_streak >= 3
  active -> probationary    : manually_edited == True (reflect modified in active state)
  deprecated / archived     : terminal states, no transition
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cryptotrader.agents.skills.schema import PatternRecord

logger = logging.getLogger(__name__)

# Threshold constants (D-EV-03)
_OBSERVED_PROMO_SUCCESSES = 3  # signal 1: PnL success count
_PROBATIONARY_PROMO_DAYS = 3  # signal 2+3: time since last modification (3 days ≈ 5 cycle proxy)
_ARCHIVED_FUNDAMENTAL_STREAK = 3  # signal 5: consecutive fundamental failures

# Required frontmatter fields to consider "fully filled" (signal 3)
_REQUIRED_FIELDS = ("name", "agent", "description", "maturity", "version")
_MAX_BODY_LINES = 300


@dataclass
class Transition:
    """Record of a state transition triggered by the FSM."""

    rule_id: str
    agent_id: str
    old_state: str
    new_state: str
    triggered_by: str  # pnl_threshold / time_elapsed / fundamental_streak / reflect_modified
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
    # Ensure timezone-aware
    if last_mod.tzinfo is None:
        last_mod = last_mod.replace(tzinfo=UTC)
    return now - last_mod


def evaluate_transitions(rule: PatternRecord) -> PatternRecord | None:
    """Evaluate FSM transitions for a PatternRecord.

    Returns updated PatternRecord if state changed, or None if no transition.
    Does NOT persist changes — caller is responsible for writing back to file.

    FR-Z11: 5-signal FSM.
    FR-Z12: returns new PatternRecord or None.
    """
    from dataclasses import replace

    maturity = rule.maturity

    # Terminal states: no further transitions
    if maturity in ("deprecated", "archived"):
        return None

    now = datetime.now(UTC)

    if maturity == "observed":
        # Signal 1: enough PnL successes
        successes = rule.pnl_track.wins if hasattr(rule.pnl_track, "wins") else 0
        if successes >= _OBSERVED_PROMO_SUCCESSES:
            logger.debug("FSM: %s observed->probationary (successes=%d)", rule.name, successes)
            return replace(
                rule,
                maturity="probationary",
                last_modified_at=now,
            )
        return None

    if maturity == "probationary":
        # Signal 2+3: time elapsed without reflect modification + quality checks
        elapsed = _time_since_modified(rule)
        if (
            elapsed >= timedelta(days=_PROBATIONARY_PROMO_DAYS)
            and _frontmatter_filled(rule)
            and _body_within_limit(rule)
        ):
            logger.debug("FSM: %s probationary->active (elapsed=%s)", rule.name, elapsed)
            return replace(
                rule,
                maturity="active",
                last_modified_at=now,
            )
        return None

    if maturity == "active":
        # Signal 5: fundamental failure streak triggers archival
        if rule.fundamental_failure_streak >= _ARCHIVED_FUNDAMENTAL_STREAK:
            logger.debug(
                "FSM: %s active->archived (streak=%d)",
                rule.name,
                rule.fundamental_failure_streak,
            )
            return replace(
                rule,
                maturity="archived",
                last_modified_at=now,
            )
        # 撤销条件: reflect modified in active state -> demote to probationary
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
        triggered_by = "time_elapsed"
    elif old_rule.maturity == "active" and new_rule.maturity == "archived":
        triggered_by = "fundamental_streak"
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

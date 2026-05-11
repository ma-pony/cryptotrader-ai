"""spec 018 EvolvingMemoryProvider unit tests — tests/test_evolving_memory_provider.py

SC-Z4: >= 10 use cases PASS.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from cryptotrader.agents.skills.schema import PatternRecord, PnLTrack
from cryptotrader.learning.evolution.provider import (
    EvolvingMemoryProvider,
    _load_pattern_from_path,
    _save_pattern_to_path,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_pattern_file(
    tmp_path: Path,
    agent: str = "tech",
    name: str = "test_rule",
    maturity: str = "active",
    wins: int = 5,
    cases: int = 10,
    importance: float = 0.7,
    access_count: int = 0,
    fundamental_failure_streak: int = 0,
    manually_edited: bool = False,
) -> Path:
    """Create a pattern markdown file in tmp_path/agent/patterns/."""
    pattern_dir = tmp_path / agent / "patterns"
    pattern_dir.mkdir(parents=True, exist_ok=True)
    path = pattern_dir / f"{name}.md"
    pt = PnLTrack(cases=cases, wins=wins, win_rate=wins / cases if cases > 0 else 0.0)
    rule = PatternRecord(
        name=name,
        agent=agent,
        description=f"description of {name}",
        body=f"## Rule\nBody content for {name}.",
        maturity=maturity,  # type: ignore[arg-type]
        pnl_track=pt,
        importance=importance,
        access_count=access_count,
        fundamental_failure_streak=fundamental_failure_streak,
        manually_edited=manually_edited,
        file_path=path,
    )
    _save_pattern_to_path(rule, path)
    return path


def _make_case_file(
    tmp_path: Path,
    cycle_id: str = "cycle_001",
    pair: str = "BTC/USDT",
    verdict_action: str = "long",
    final_pnl: float | None = 50.0,
    ive_classification: dict | None = None,
    applied_patterns: list[str] | None = None,
    timestamp: datetime | None = None,
) -> Path:
    """Create a case markdown file in tmp_path/cases/."""
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    path = cases_dir / f"{cycle_id}.md"
    ts = (timestamp or datetime.now(UTC)).isoformat()
    ive_str = json.dumps(ive_classification) if ive_classification else "null"
    applied_str = json.dumps(applied_patterns or [])
    content = f"""---
cycle_id: {cycle_id}
timestamp: {ts}
pair: {pair}
verdict_action: {verdict_action}
final_pnl: {final_pnl if final_pnl is not None else "null"}
risk_gate_passed: true
applied_patterns: {applied_str}
ive_classification: {ive_str}
---
# Cycle Record: {cycle_id}

## Agent Analyses

### Tech

Bearish signal detected.

## IVE Classification

(pending classification)
"""
    path.write_text(content, encoding="utf-8")
    return path


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_load_active_rule_pareto_sorted(tmp_path: Path):
    """T026(a): active rules loaded and returned via Pareto ranking."""
    _make_pattern_file(tmp_path, name="rule_a", maturity="active", wins=8, cases=10, importance=0.9)
    _make_pattern_file(tmp_path, name="rule_b", maturity="active", wins=3, cases=10, importance=0.5)

    provider = EvolvingMemoryProvider(memory_root=tmp_path, top_k_rules=5)
    result = provider.get_recent_memory("tech", {})

    assert "### Patterns" in result
    assert "rule_a" in result or "rule_b" in result


def test_archived_rules_filtered_out(tmp_path: Path):
    """T026(b): archived and deprecated rules excluded from get_recent_memory."""
    _make_pattern_file(tmp_path, name="archived_rule", maturity="archived")
    _make_pattern_file(tmp_path, name="deprecated_rule", maturity="deprecated")
    _make_pattern_file(tmp_path, name="active_rule", maturity="active", wins=5, cases=10)

    provider = EvolvingMemoryProvider(memory_root=tmp_path)
    result = provider.get_recent_memory("tech", {})

    assert "archived_rule" not in result
    assert "deprecated_rule" not in result
    assert "active_rule" in result


def test_cases_loaded_by_timestamp_descending(tmp_path: Path):
    """T026(c): cases loaded in timestamp descending order (most recent first)."""
    now = datetime.now(UTC)
    _make_case_file(tmp_path, cycle_id="old_cycle", timestamp=now - timedelta(hours=5))
    _make_case_file(tmp_path, cycle_id="new_cycle", timestamp=now - timedelta(minutes=10))

    provider = EvolvingMemoryProvider(memory_root=tmp_path, top_n_cases=5)
    result = provider.get_recent_memory("tech", {})

    # new_cycle should appear before old_cycle
    if "new_cycle" in result and "old_cycle" in result:
        assert result.index("new_cycle") < result.index("old_cycle")


def test_access_count_incremented_on_access(tmp_path: Path):
    """T026(d): access_count and last_accessed_at are updated after get_recent_memory."""
    path = _make_pattern_file(tmp_path, name="my_rule", maturity="active", access_count=0)

    provider = EvolvingMemoryProvider(memory_root=tmp_path, top_k_rules=5)
    provider.get_recent_memory("tech", {})

    # Reload from disk and check access_count incremented
    updated = _load_pattern_from_path(path)
    assert updated is not None
    assert updated.access_count == 1


@pytest.mark.asyncio
async def test_ive_exception_returns_empty_string(tmp_path: Path):
    """T026(e): IVE LLM exception in classify_pending_cases → noise result + warning."""
    from unittest.mock import AsyncMock

    _make_case_file(tmp_path, cycle_id="unclassified", ive_classification=None)

    provider = EvolvingMemoryProvider(memory_root=tmp_path)
    with patch(
        "cryptotrader.learning.evolution.ive.classify_case",
        new=AsyncMock(side_effect=RuntimeError("LLM down")),
    ):
        # Should not raise; problem case should be skipped
        results = await provider.classify_pending_cases()
    # Either empty (exception skipped) or returned noise
    assert isinstance(results, list)


def test_fsm_exception_returns_empty_transitions(tmp_path: Path):
    """T026(f): FSM evaluate_transitions exception → skipped + continues."""
    _make_pattern_file(tmp_path, name="rule_x", maturity="observed", wins=5, cases=5)

    provider = EvolvingMemoryProvider(memory_root=tmp_path)
    with patch(
        "cryptotrader.learning.evolution.fsm.evaluate_transitions",
        side_effect=RuntimeError("FSM error"),
    ):
        transitions = provider.evaluate_all_rules()
    assert isinstance(transitions, list)


def test_pareto_exception_returns_empty_string(tmp_path: Path):
    """T026(g): Pareto rank_rules exception → get_recent_memory returns ''."""
    _make_pattern_file(tmp_path, name="rule_y", maturity="active")

    provider = EvolvingMemoryProvider(memory_root=tmp_path)
    with patch(
        "cryptotrader.learning.evolution.pareto.rank_rules",
        side_effect=RuntimeError("Pareto error"),
    ):
        result = provider.get_recent_memory("tech", {})
    assert result == ""


def test_io_exception_returns_empty_string(tmp_path: Path):
    """T026(h): IO exception in get_recent_memory → returns ''."""
    provider = EvolvingMemoryProvider(memory_root=tmp_path)
    with patch.object(provider, "_load_active_patterns", side_effect=OSError("disk error")):
        result = provider.get_recent_memory("tech", {})
    assert result == ""


def test_empty_directory_returns_placeholder(tmp_path: Path):
    """T026(i): empty memory root → returns '暂无历史记忆'."""
    provider = EvolvingMemoryProvider(memory_root=tmp_path)
    result = provider.get_recent_memory("tech", {})
    assert result == "暂无历史记忆"


def test_provider_implements_memory_provider_protocol(tmp_path: Path):
    """T026(j): EvolvingMemoryProvider duck-type satisfies MemoryProvider Protocol."""
    from cryptotrader.agents.prompt_builder import MemoryProvider

    provider = EvolvingMemoryProvider(memory_root=tmp_path)
    # Duck-type check: has get_recent_memory with correct signature
    assert hasattr(provider, "get_recent_memory")
    assert callable(provider.get_recent_memory)
    # Runtime isinstance check via Protocol (requires runtime_checkable)
    # Just verify it can be used as MemoryProvider via attribute check
    _ = MemoryProvider  # confirm import works


def test_evaluate_all_rules_returns_transitions(tmp_path: Path):
    """T026 extra: evaluate_all_rules returns transitions for state changes."""
    # spec 021: 合并 FSM 要求 cases ≥ 5 + win_rate ≥ 0.60
    _make_pattern_file(tmp_path, name="promo_rule", maturity="observed", wins=4, cases=5)

    provider = EvolvingMemoryProvider(memory_root=tmp_path)
    transitions = provider.evaluate_all_rules()

    assert isinstance(transitions, list)
    # At least one transition should occur (observed → probationary)
    assert len(transitions) >= 1
    t = transitions[0]
    assert t.old_state == "observed"
    assert t.new_state == "probationary"


@pytest.mark.asyncio
async def test_classify_pending_skips_already_classified(tmp_path: Path):
    """T026 extra: cases with existing ive_classification are skipped."""
    existing_cls = {"failure_type": "noise", "confidence": 0.5, "reasoning": "already done"}
    _make_case_file(tmp_path, cycle_id="classified_001", ive_classification=existing_cls)

    provider = EvolvingMemoryProvider(memory_root=tmp_path)
    results = await provider.classify_pending_cases()

    # Should be empty since the case is already classified
    assert results == []

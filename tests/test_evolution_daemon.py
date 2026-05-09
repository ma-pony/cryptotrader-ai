"""spec 022 -- Unit tests for EvolutionDaemon (T007-T009, T012-T013, T015-T017).

Tests cover:
- Pareto action: archives dominated rules (US1)
- Regime action: recalculates stale tags (US2)
- Skill proposal action: threshold-based trigger (US3)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cryptotrader.config import EvolutionDaemonConfig
from cryptotrader.ops.daemon import ActionResult, EvolutionDaemon

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def daemon_config():
    return EvolutionDaemonConfig(
        enabled=True,
        cron="0 0 * * *",
        actions=["pareto", "regime", "skill_proposal"],
        llm_model="",
        propose_threshold=10,
    )


@pytest.fixture
def daemon(daemon_config):
    return EvolutionDaemon(config=daemon_config)


def _make_pattern(name: str, agent: str, maturity: str, win_rate: float, importance: float) -> MagicMock:
    """Build a mock PatternRecord."""
    p = MagicMock()
    p.name = name
    p.agent = agent
    p.maturity = maturity
    p.importance = importance
    p.file_path = MagicMock(spec=Path)
    p.pnl_track = MagicMock()
    p.pnl_track.wins = int(win_rate * 100)
    p.pnl_track.cases = 100
    return p


# ---------------------------------------------------------------------------
# T007: test_pareto_action_archives_dominated_rules
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pareto_action_archives_dominated_rules(daemon, tmp_path):
    """50 active rules (10 low win_rate) -> >= 10 archived after pareto action."""
    patterns = [_make_pattern(f"high_{i}", "tech", "active", 0.8, 0.9) for i in range(40)]
    patterns.extend(_make_pattern(f"low_{i}", "tech", "active", 0.2, 0.1) for i in range(10))

    saved_patterns = []

    # Create fake patterns dir with 50 placeholder files
    patterns_dir = tmp_path / "tech" / "patterns"
    patterns_dir.mkdir(parents=True)
    file_paths = []
    for i in range(50):
        f = patterns_dir / f"p{i}.md"
        f.write_text(f"---\nname: p{i}\n---\n")
        file_paths.append(f)

    # Patch load to return our mock patterns; patch save to capture archived
    def fake_load(path):
        idx = int(path.stem[1:])
        return patterns[idx]

    with (
        patch("cryptotrader.learning.evolution._io._load_pattern_from_path", side_effect=fake_load),
        patch(
            "cryptotrader.learning.evolution._io._save_pattern_to_path",
            side_effect=lambda p, path: saved_patterns.append(p),
        ),
        patch("cryptotrader.agents.skills._constants.DEFAULT_AGENT_MEMORY_DIR", tmp_path),
        patch("cryptotrader.agents.skills._constants.VALID_AGENT_IDS", frozenset({"tech"})),
    ):
        result = await daemon._action_pareto()

    assert result.status == "PASS"
    archived = [p for p in saved_patterns if p.maturity == "archived"]
    assert len(archived) >= 10, f"Expected >= 10 archived, got {len(archived)}"


# ---------------------------------------------------------------------------
# T008: test_pareto_action_empty_rules
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pareto_action_empty_rules(daemon, tmp_path):
    """0 active rules -> PASS 0ms, no exception."""
    # tmp_path has no patterns dirs
    with (
        patch("cryptotrader.agents.skills._constants.DEFAULT_AGENT_MEMORY_DIR", tmp_path),
        patch("cryptotrader.agents.skills._constants.VALID_AGENT_IDS", frozenset({"tech"})),
    ):
        result = await daemon._action_pareto()

    assert result.status == "PASS"
    assert result.details["archived_count"] == 0
    assert result.details["processed_count"] == 0
    assert result.duration_ms >= 0


# ---------------------------------------------------------------------------
# T009: test_pareto_action_all_frontier
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pareto_action_all_frontier(daemon, tmp_path):
    """All rules on Pareto frontier -> 0 archived (idempotent)."""
    # 5 patterns each dominating in different objectives — no single rule dominates another
    patterns = [
        _make_pattern("p1", "tech", "active", 0.9, 0.1),  # high wr, low cp
        _make_pattern("p2", "tech", "active", 0.1, 0.9),  # low wr, high cp
        _make_pattern("p3", "tech", "active", 0.5, 0.5),
        _make_pattern("p4", "tech", "active", 0.8, 0.2),
        _make_pattern("p5", "tech", "active", 0.2, 0.8),
    ]

    patterns_dir = tmp_path / "tech" / "patterns"
    patterns_dir.mkdir(parents=True)
    file_paths = []
    for i in range(5):
        f = patterns_dir / f"p{i + 1}.md"
        f.write_text(f"---\nname: p{i + 1}\n---\n")
        file_paths.append(f)

    saved_patterns = []

    def fake_load(path):
        idx = int(path.stem[1:]) - 1
        return patterns[idx]

    with (
        patch("cryptotrader.learning.evolution._io._load_pattern_from_path", side_effect=fake_load),
        patch(
            "cryptotrader.learning.evolution._io._save_pattern_to_path",
            side_effect=lambda p, path: saved_patterns.append(p),
        ),
        patch("cryptotrader.agents.skills._constants.DEFAULT_AGENT_MEMORY_DIR", tmp_path),
        patch("cryptotrader.agents.skills._constants.VALID_AGENT_IDS", frozenset({"tech"})),
    ):
        result = await daemon._action_pareto()

    assert result.status == "PASS"
    archived = [p for p in saved_patterns if p.maturity == "archived"]
    assert len(archived) == 0, f"Expected 0 archived on frontier, got {len(archived)}"


# ---------------------------------------------------------------------------
# T012: test_regime_action_recalculates_stale_tags
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_regime_action_recalculates_stale_tags(daemon):
    """Regime action returns changed_count > 0 when stale tags differ from current."""
    with patch("cryptotrader.learning.memory.refilter_records_by_regime", return_value=35) as mock_rfr:
        result = await daemon._action_regime()

    assert result.status == "PASS"
    assert result.details["changed_count"] == 35
    mock_rfr.assert_called_once()


# ---------------------------------------------------------------------------
# T013: test_regime_action_idempotent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_regime_action_idempotent(daemon):
    """Regime action returns 0 changed when all cases already current."""
    with patch("cryptotrader.learning.memory.refilter_records_by_regime", return_value=0):
        result = await daemon._action_regime()

    assert result.status == "PASS"
    assert result.details["changed_count"] == 0


# ---------------------------------------------------------------------------
# T015: test_skill_proposal_threshold_met
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_skill_proposal_threshold_met(daemon, tmp_path):
    """12 active rules for agent:tech -> propose_new_skill called once, draft created."""
    active_patterns = [{"name": f"p{i}", "agent": "tech", "maturity": "active"} for i in range(12)]
    draft_path = tmp_path / "agent_skills" / "proposed_tech_skill" / "SKILL.md.draft"

    def fake_load(agent_id, memory_dir):
        return active_patterns if agent_id == "tech" else []

    with (
        patch("cryptotrader.learning.skill_proposal._load_active_patterns_for_agent", side_effect=fake_load),
        patch("cryptotrader.learning.skill_proposal.propose_new_skill", return_value=draft_path) as mock_propose,
        patch("cryptotrader.agents.skills._constants.DEFAULT_AGENT_MEMORY_DIR", tmp_path),
    ):
        result = await daemon._action_skill_proposal()

    assert result.status == "PASS"
    assert str(draft_path) in result.details["drafts_created"]
    mock_propose.assert_called_once_with(scope="agent:tech")


# ---------------------------------------------------------------------------
# T016: test_skill_proposal_threshold_not_met
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_skill_proposal_threshold_not_met(daemon, tmp_path):
    """8 active rules (< threshold 10) -> no .draft created, step PASS."""
    active_patterns = [{"name": f"p{i}", "agent": "tech"} for i in range(8)]

    with (
        patch(
            "cryptotrader.learning.skill_proposal._load_active_patterns_for_agent",
            return_value=active_patterns,
        ),
        patch("cryptotrader.learning.skill_proposal.propose_new_skill") as mock_propose,
        patch("cryptotrader.agents.skills._constants.DEFAULT_AGENT_MEMORY_DIR", tmp_path),
    ):
        result = await daemon._action_skill_proposal()

    assert result.status == "PASS"
    assert result.details["drafts_created"] == []
    mock_propose.assert_not_called()


# ---------------------------------------------------------------------------
# T017: test_skill_proposal_per_agent_independent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_skill_proposal_per_agent_independent(daemon, tmp_path):
    """tech=12, chain=8, news=15, macro=5 -> 2 .draft files (tech + news)."""
    agent_counts = {"tech": 12, "chain": 8, "news": 15, "macro": 5}
    draft_tech = tmp_path / "agent_skills" / "tech_skill" / "SKILL.md.draft"
    draft_news = tmp_path / "agent_skills" / "news_skill" / "SKILL.md.draft"

    def fake_load(agent_id, memory_dir):
        return [{"name": f"p{i}", "agent": agent_id} for i in range(agent_counts[agent_id])]

    def fake_propose(scope):
        if scope == "agent:tech":
            return draft_tech
        if scope == "agent:news":
            return draft_news
        return None

    with (
        patch("cryptotrader.learning.skill_proposal._load_active_patterns_for_agent", side_effect=fake_load),
        patch("cryptotrader.learning.skill_proposal.propose_new_skill", side_effect=fake_propose),
        patch("cryptotrader.agents.skills._constants.DEFAULT_AGENT_MEMORY_DIR", tmp_path),
    ):
        result = await daemon._action_skill_proposal()

    assert result.status == "PASS"
    assert len(result.details["drafts_created"]) == 2
    assert str(draft_tech) in result.details["drafts_created"]
    assert str(draft_news) in result.details["drafts_created"]


# ---------------------------------------------------------------------------
# T024: test_soft_degrade_llm_failure (moved here from C3 for cohesion)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_soft_degrade_llm_failure(daemon):
    """LLM error in skill_proposal -> exit 0, skill_proposal SKIP, pareto/regime PASS."""
    from openai import OpenAIError

    with (
        patch.object(daemon, "_action_pareto", return_value=ActionResult("pareto", "PASS", 10, {})),
        patch.object(daemon, "_action_regime", return_value=ActionResult("regime", "PASS", 5, {})),
        patch.object(daemon, "_action_skill_proposal", side_effect=OpenAIError("boom")),
    ):
        result = await daemon.run_once()

    assert result.exit_code == 0
    by_name = {a.name: a for a in result.actions_run}
    assert by_name["pareto"].status == "PASS"
    assert by_name["regime"].status == "PASS"
    assert by_name["skill_proposal"].status == "SKIP"


# ---------------------------------------------------------------------------
# T025: test_lock_timeout_skips_run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lock_timeout_skips_run(daemon, tmp_path):
    """If locks cannot be acquired -> exit 0, no actions run."""
    with patch("cryptotrader.ops.daemon._try_acquire_locks", return_value=(False, [])):
        result = await daemon.run_once()

    assert result.exit_code == 0
    assert result.actions_run == []


# ---------------------------------------------------------------------------
# run_once smoke test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_once_all_pass(daemon):
    """run_once with all mocked actions -> exit_code=0, 3 PASS results."""
    with (
        patch.object(daemon, "_action_pareto", return_value=ActionResult("pareto", "PASS", 10, {})),
        patch.object(daemon, "_action_regime", return_value=ActionResult("regime", "PASS", 5, {})),
        patch.object(
            daemon,
            "_action_skill_proposal",
            return_value=ActionResult("skill_proposal", "PASS", 20, {}),
        ),
    ):
        result = await daemon.run_once()

    assert result.exit_code == 0
    assert len(result.actions_run) == 3
    assert all(a.status == "PASS" for a in result.actions_run)

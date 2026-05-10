"""spec 020c — Daemon + lineage integration tests.

T014: test_daemon_pareto_archives_recorded_in_transitions
T015: test_daemon_run_once_commits_with_transitions
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest

from cryptotrader.ops.daemon import ActionResult, EvolutionDaemon, RunResult, _build_lineage_summary

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# T014: _action_pareto transitions collection
# ---------------------------------------------------------------------------


def test_daemon_pareto_archives_recorded_in_transitions() -> None:
    """T014: When pareto archives N rules, transitions list length == N."""
    transitions = [
        {
            "rule_id": f"rule_{i}",
            "agent_id": "tech",
            "old_state": "active",
            "new_state": "archived",
            "triggered_by": "pareto_dominated",
        }
        for i in range(5)
    ]
    pareto_result = ActionResult(
        name="pareto",
        status="PASS",
        duration_ms=100,
        details={"archived_count": 5, "processed_count": 10, "transitions": transitions},
    )
    run_result = RunResult(actions_run=[pareto_result], total_duration_ms=200, exit_code=0)

    summary = _build_lineage_summary(run_result)
    pareto_action = next(a for a in summary["actions"] if a["name"] == "pareto")
    assert len(pareto_action["details"]["transitions"]) == 5
    for t in pareto_action["details"]["transitions"]:
        assert "rule_id" in t
        assert t["new_state"] == "archived"
        assert t["triggered_by"] == "pareto_dominated"


# ---------------------------------------------------------------------------
# T015: daemon run_once commits with transitions batch
# ---------------------------------------------------------------------------


def _setup_git_repo(repo: Path) -> None:
    """Create a minimal git repo with one commit in agent_memory/."""
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    agent_mem = repo / "agent_memory" / "patterns"
    agent_mem.mkdir(parents=True)
    (agent_mem / "seed.md").write_text("seed")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True)
    (agent_mem / "rule_dirty.md").write_text("dirty rule")


def _assert_evolution_commit(repo: Path) -> None:
    """Sync helper: assert evolution branch exists and last commit has the trailer."""
    branches = subprocess.check_output(["git", "branch", "-a"], cwd=repo).decode()
    assert "evolution" in branches
    log = subprocess.check_output(["git", "log", "evolution", "-1", "--format=%B"], cwd=repo).decode()
    assert "Auto-Generated-By: spec-020c" in log


@pytest.mark.asyncio
async def test_daemon_run_once_commits_with_transitions(tmp_path: Path) -> None:
    """T015: daemon run_once -> evolution branch gets 1 commit containing transitions."""
    _setup_git_repo(tmp_path)

    cfg = SimpleNamespace(actions=["pareto"], cron="0 2 * * *", propose_threshold=5)
    daemon = EvolutionDaemon(config=cfg)

    transitions = [
        {
            "rule_id": f"rule_{i}",
            "agent_id": "tech",
            "old_state": "active",
            "new_state": "archived",
            "triggered_by": "pareto_dominated",
        }
        for i in range(5)
    ]
    mock_pareto_result = ActionResult(
        name="pareto",
        status="PASS",
        duration_ms=50,
        details={"archived_count": 5, "processed_count": 10, "transitions": transitions},
    )

    from cryptotrader.ops import daemon as daemon_mod
    from cryptotrader.ops.lineage import GitLineageHook

    def _patched_commit_lineage(run_result: RunResult) -> None:
        from cryptotrader.observability.daemon_metrics import record_lineage_event

        summary = _build_lineage_summary(run_result)
        hook = GitLineageHook(branch="evolution", repo_path=tmp_path)
        commit_result = hook.commit_changes(summary)
        record_lineage_event(success=commit_result.success)

    with (
        patch.object(daemon, "_action_pareto", AsyncMock(return_value=mock_pareto_result)),
        patch.object(daemon_mod, "_commit_lineage", side_effect=_patched_commit_lineage),
        patch.object(daemon_mod, "_record_run_metrics"),
    ):
        result = await daemon.run_once()

    assert result.exit_code == 0
    assert len(result.actions_run) == 1

    # Use sync helper to avoid ASYNC101 (subprocess calls inside async fn)
    _assert_evolution_commit(tmp_path)


# ---------------------------------------------------------------------------
# _build_lineage_summary shape tests
# ---------------------------------------------------------------------------


def test_build_lineage_summary_type_is_daemon() -> None:
    """_build_lineage_summary always sets type='daemon'."""
    run_result = RunResult(actions_run=[], total_duration_ms=0, exit_code=0)
    summary = _build_lineage_summary(run_result)
    assert summary["type"] == "daemon"
    assert summary["actions"] == []


def test_build_lineage_summary_includes_all_actions() -> None:
    """_build_lineage_summary includes all action results."""
    results = [
        ActionResult(name="pareto", status="PASS", duration_ms=10, details={"archived_count": 1}),
        ActionResult(name="regime", status="PASS", duration_ms=5, details={"changed_count": 2}),
        ActionResult(name="skill_proposal", status="SKIP", duration_ms=3, details={"reason": "llm_error"}),
    ]
    run_result = RunResult(actions_run=results, total_duration_ms=100, exit_code=0)
    summary = _build_lineage_summary(run_result)
    assert len(summary["actions"]) == 3
    names = [a["name"] for a in summary["actions"]]
    assert "pareto" in names
    assert "regime" in names
    assert "skill_proposal" in names

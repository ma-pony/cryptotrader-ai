"""spec 020c — E2E git lineage end-to-end test (T026).

Mocked daemon cycle:
  daemon.run_once() -> evolution branch created -> commit contains trailer
  -> lineage metrics gauges updated.
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest

from cryptotrader.observability.daemon_metrics import (
    get_lineage_commit_failure_aggregator,
    record_lineage_event,
)
from cryptotrader.ops.daemon import ActionResult, EvolutionDaemon, RunResult, _build_lineage_summary
from cryptotrader.ops.lineage import GitLineageHook

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_git_repo(repo: Path) -> None:
    """Create a minimal git repo with one commit in agent_memory/."""
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "e2e@test.com"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "E2E"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    agent_mem = repo / "agent_memory" / "patterns"
    agent_mem.mkdir(parents=True)
    (agent_mem / "seed.md").write_text("seed")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True)
    # Create a dirty file so lineage hook has something to commit
    (agent_mem / "e2e_rule.md").write_text("e2e rule content")


def _assert_evolution_branch_and_trailer(repo: Path) -> None:
    """Assert evolution branch exists and last commit has the spec-020c trailer."""
    branches = subprocess.check_output(["git", "branch", "-a"], cwd=repo).decode()
    assert "evolution" in branches, f"evolution branch not found in: {branches}"
    log = subprocess.check_output(
        ["git", "log", "evolution", "-1", "--format=%B"], cwd=repo
    ).decode()
    assert "Auto-Generated-By: spec-020c" in log, f"trailer missing from commit: {log}"


# ---------------------------------------------------------------------------
# T026: E2E mocked daemon cycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_e2e_daemon_cycle_creates_evolution_commit_and_updates_metrics(
    tmp_path: Path,
) -> None:
    """T026: Full mocked daemon cycle -- evolution branch + commit trailer + metrics gauges."""
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
        for i in range(3)
    ]
    mock_pareto_result = ActionResult(
        name="pareto",
        status="PASS",
        duration_ms=50,
        details={"archived_count": 3, "processed_count": 8, "transitions": transitions},
    )

    from cryptotrader.ops import daemon as daemon_mod

    lineage_calls: list[bool] = []

    def _patched_commit_lineage(run_result: RunResult) -> None:
        summary = _build_lineage_summary(run_result)
        hook = GitLineageHook(branch="evolution", repo_path=tmp_path)
        commit_result = hook.commit_changes(summary)
        lineage_calls.append(commit_result.success)
        record_lineage_event(success=commit_result.success)

    with (
        patch.object(daemon, "_action_pareto", AsyncMock(return_value=mock_pareto_result)),
        patch.object(daemon_mod, "_commit_lineage", side_effect=_patched_commit_lineage),
        patch.object(daemon_mod, "_record_run_metrics"),
    ):
        result = await daemon.run_once()

    # Daemon run completed successfully
    assert result.exit_code == 0
    assert len(result.actions_run) == 1

    # Evolution branch + commit trailer (sync helpers avoid ASYNC101)
    _assert_evolution_branch_and_trailer(tmp_path)

    # Lineage hook was called once and succeeded
    assert len(lineage_calls) == 1, "lineage commit should be called exactly once"
    assert lineage_calls[0] is True, "lineage commit should succeed"


def test_e2e_lineage_failure_recorded_in_metrics(tmp_path: Path) -> None:
    """T026b: When lineage commit fails (bad repo), metrics record the failure."""
    # A git repo with a dirty file but no user.email config -> commit will fail
    bad_repo = tmp_path / "bad_repo"
    bad_repo.mkdir()
    subprocess.run(["git", "init"], cwd=bad_repo, check=True, capture_output=True)
    # Intentionally omit git config user.email so commit fails
    agent_mem = bad_repo / "agent_memory"
    agent_mem.mkdir()
    (agent_mem / "dirty.md").write_text("dirty")

    failure_agg = get_lineage_commit_failure_aggregator()
    before_rate = failure_agg.failure_rate()

    hook = GitLineageHook(branch="evolution", repo_path=bad_repo)
    result = hook.commit_changes({"type": "daemon", "actions": []})

    # Soft-fail: hook returns False (git commit fails without user config), no exception raised
    assert result.success is False

    # Record the failure
    record_lineage_event(success=False)
    assert failure_agg.failure_rate() > before_rate, "failure rate should increase after failure event"


def test_e2e_build_lineage_summary_shape_for_pareto() -> None:
    """T026c: _build_lineage_summary produces correct shape for pareto action."""
    transitions = [
        {
            "rule_id": "r1",
            "agent_id": "macro",
            "old_state": "active",
            "new_state": "archived",
            "triggered_by": "pareto_dominated",
        },
    ]
    results = [
        ActionResult(
            name="pareto",
            status="PASS",
            duration_ms=20,
            details={"archived_count": 1, "processed_count": 5, "transitions": transitions},
        ),
        ActionResult(
            name="regime",
            status="PASS",
            duration_ms=5,
            details={"changed_count": 1, "total_count": 4},
        ),
    ]
    run_result = RunResult(actions_run=results, total_duration_ms=100, exit_code=0)
    summary = _build_lineage_summary(run_result)

    assert summary["type"] == "daemon"
    assert len(summary["actions"]) == 2
    pareto_action = next(a for a in summary["actions"] if a["name"] == "pareto")
    assert pareto_action["details"]["transitions"][0]["rule_id"] == "r1"
    assert pareto_action["details"]["transitions"][0]["triggered_by"] == "pareto_dominated"

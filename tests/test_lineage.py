"""spec 020c — GitLineageHook unit tests.

T008: test_commit_changes_creates_orphan_evolution_branch
T009: test_commit_changes_with_no_changes
T010: test_commit_changes_protects_dev_workspace
T011: test_commit_failure_soft_fail
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from cryptotrader.ops.lineage import CommitResult, GitLineageHook

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_repo(tmp_path: Path) -> Path:
    """Create a real git repo in a temp directory with an initial commit."""
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True, capture_output=True)
    # Create agent_memory dir with a file so there's something to commit
    agent_memory = tmp_path / "agent_memory" / "patterns"
    agent_memory.mkdir(parents=True)
    (agent_memory / "rule1.md").write_text("rule content")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True, capture_output=True)
    return tmp_path


@pytest.fixture
def dirty_repo(temp_repo: Path) -> Path:
    """Add an uncommitted change to agent_memory/."""
    (temp_repo / "agent_memory" / "patterns" / "rule2.md").write_text("new rule")
    return temp_repo


# ---------------------------------------------------------------------------
# T008: orphan evolution branch creation
# ---------------------------------------------------------------------------


def test_commit_changes_creates_orphan_evolution_branch(dirty_repo: Path) -> None:
    """T008: First-time commit creates evolution branch (orphan, no main history)."""
    hook = GitLineageHook(branch="evolution", repo_path=dirty_repo)
    summary = {
        "type": "daemon",
        "actions": [{"name": "pareto", "status": "PASS", "details": {"archived_count": 1, "processed_count": 5}}],
    }
    result = hook.commit_changes(summary)

    assert result.success is True
    assert result.commit_sha is not None
    assert result.error is None
    assert result.duration_ms >= 0

    # evolution branch must exist
    branches = subprocess.check_output(["git", "branch", "-a"], cwd=dirty_repo).decode()
    assert "evolution" in branches

    # commit message must contain trailer
    log = subprocess.check_output(["git", "log", "evolution", "-1", "--format=%B"], cwd=dirty_repo).decode()
    assert "Auto-Generated-By: spec-020c" in log
    assert "archived=1" in log

    # back on original branch (main/master)
    current = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=dirty_repo).decode().strip()
    assert current != "evolution"


# ---------------------------------------------------------------------------
# T009: no changes → no empty commit
# ---------------------------------------------------------------------------


def test_commit_changes_with_no_changes(temp_repo: Path) -> None:
    """T009: When agent_memory/ + agent_skills/ are clean, no commit is created."""
    hook = GitLineageHook(branch="evolution", repo_path=temp_repo)
    summary = {"type": "daemon", "actions": []}

    result = hook.commit_changes(summary)

    assert result.success is True
    assert result.commit_sha is None  # no commit created

    # evolution branch should NOT be created
    branches = subprocess.check_output(["git", "branch", "-a"], cwd=temp_repo).decode()
    assert "evolution" not in branches


# ---------------------------------------------------------------------------
# T010: dev workspace stash protection
# ---------------------------------------------------------------------------


def test_commit_changes_protects_dev_workspace(dirty_repo: Path) -> None:
    """T010: Dev changes on main branch are stashed and restored after commit."""
    # Create a file outside agent_memory that simulates dev work
    dev_file = dirty_repo / "my_dev_file.py"
    dev_file.write_text("# dev work in progress")

    # Also have a tracked change in agent_memory (dirty)
    (dirty_repo / "agent_memory" / "patterns" / "rule_extra.md").write_text("extra rule")

    hook = GitLineageHook(branch="evolution", repo_path=dirty_repo)
    summary = {
        "type": "daemon",
        "actions": [{"name": "pareto", "status": "PASS", "details": {"archived_count": 0, "processed_count": 3}}],
    }
    result = hook.commit_changes(summary)

    assert result.success is True

    # After commit, we should be back on original branch
    current = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=dirty_repo).decode().strip()
    assert current != "evolution"

    # The dev file should still exist (stash was popped)
    assert dev_file.exists()


# ---------------------------------------------------------------------------
# T011: subprocess failure → soft fail, CommitResult(success=False)
# ---------------------------------------------------------------------------


def test_commit_failure_soft_fail(tmp_path: Path) -> None:
    """T011: CalledProcessError from git → CommitResult(success=False), no exception raised."""
    hook = GitLineageHook(branch="evolution", repo_path=tmp_path)

    # _has_changes returns True so we proceed, but _current_branch raises
    with (
        patch.object(hook, "_has_changes", return_value=True),
        patch.object(
            hook,
            "_current_branch",
            side_effect=subprocess.CalledProcessError(1, "git", stderr="fatal: not a repo"),
        ),
    ):
        summary = {"type": "daemon", "actions": []}
        result = hook.commit_changes(summary)

    assert result.success is False
    assert result.error is not None
    assert "fatal: not a repo" in result.error or "CalledProcessError" in result.error or result.error != ""
    assert result.commit_sha is None
    assert result.duration_ms >= 0


# ---------------------------------------------------------------------------
# Additional: _build_message produces correct templates
# ---------------------------------------------------------------------------


def test_build_daemon_message_contains_trailer() -> None:
    """Daemon summary message must end with Auto-Generated-By trailer."""
    hook = GitLineageHook()
    summary = {
        "type": "daemon",
        "actions": [
            {"name": "pareto", "status": "PASS", "details": {"archived_count": 3, "processed_count": 10}},
            {"name": "regime", "status": "PASS", "details": {"changed_count": 5, "total_count": 42}},
            {
                "name": "skill_proposal",
                "status": "PASS",
                "details": {"drafts_created": ["p1.md"], "agents_checked": 4},
            },
        ],
    }
    msg = hook._build_message(summary)
    assert "Auto-Generated-By: spec-020c" in msg
    assert "archived=3" in msg
    assert "processed=10" in msg
    assert "changed=5" in msg
    assert "drafts_created=1" in msg


def test_build_transitions_message_contains_rule_ids() -> None:
    """Transitions summary message must list each transition + trailer."""
    hook = GitLineageHook()
    summary = {
        "type": "transitions",
        "transitions": [
            {
                "rule_id": "foo",
                "agent_id": "tech",
                "old_state": "active",
                "new_state": "archived",
                "triggered_by": "pareto_dominated",
            },
            {
                "rule_id": "bar",
                "agent_id": "macro",
                "old_state": "probationary",
                "new_state": "active",
                "triggered_by": "pnl_threshold",
            },
        ],
    }
    msg = hook._build_message(summary)
    assert "Auto-Generated-By: spec-020c" in msg
    assert "rule_id=foo" in msg
    assert "rule_id=bar" in msg
    assert "active→archived" in msg
    assert "2 maturity transitions" in msg


def test_commit_result_success_false_when_no_git_repo(tmp_path: Path) -> None:
    """No git repo → _git raises → soft fail CommitResult."""
    # tmp_path has no git repo
    hook = GitLineageHook(branch="evolution", repo_path=tmp_path)
    # Make hook think there are changes by patching _has_changes
    with patch.object(hook, "_has_changes", return_value=True):
        result = hook.commit_changes({"type": "daemon", "actions": []})
    # Should soft-fail, not raise
    assert result.success is False
    assert result.commit_sha is None


def test_mock_commit_failure_returns_soft_result() -> None:
    """CommitResult dataclass fields are correct on failure path (SC-L6)."""
    result = CommitResult(success=False, commit_sha=None, error="some git error", duration_ms=42)
    assert result.success is False
    assert result.commit_sha is None
    assert result.error == "some git error"
    assert result.duration_ms == 42

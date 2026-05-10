"""spec 020c — Git Lineage Hook.

FR-L1: GitLineageHook 类 (commit_changes / helpers)
FR-L2: subprocess git CLI (no gitpython dependency)
FR-L3: stash 保护 dev workspace + evolution orphan branch + soft fail
FR-L9: OTel error span on failure; return CommitResult(success=False)
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from time import time

logger = logging.getLogger(__name__)


@dataclass
class CommitResult:
    """Result of a single lineage commit attempt."""

    success: bool
    commit_sha: str | None
    error: str | None
    duration_ms: int


class GitLineageHook:
    """Auto-commit agent_memory/ + agent_skills/ changes to the evolution branch.

    Usage::

        hook = GitLineageHook(branch="evolution")
        result = hook.commit_changes({"type": "daemon", "actions": [...]})

    The hook is soft-fail: any git error is caught, OTel span is recorded,
    and CommitResult(success=False) is returned — the daemon continues normally.
    """

    def __init__(self, branch: str = "evolution", repo_path: Path | None = None) -> None:
        self.branch = branch
        self.repo = repo_path or Path.cwd()
        self._stash_marker = "spec-020c-lineage-stash"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def commit_changes(self, message_summary: dict) -> CommitResult:
        """Commit dirty agent_memory/ + agent_skills/ to the evolution branch.

        Strategy (FR-L3):
        - Untracked files survive branch switches in git, so no stash is needed
          in the normal case.  We stash only if checkout fails due to a conflict.
        - Path A (branch exists):   git checkout evolution  (fallback: stash+checkout)
        - Path B (first run):       git checkout --orphan evolution + rm --cached .
        - In both paths only the paths that exist on disk are staged, avoiding the
          "fatal: pathspec did not match" error when agent_skills/ is absent.
        - FR-L9: any failure → best-effort restore + OTel error span +
          CommitResult(success=False); daemon continues normally (exit 0).
        """
        span_ctx = _get_span_ctx("evolution.lineage.commit")
        with span_ctx as span:
            start = time()
            original_branch: str | None = None
            stash_pushed = False
            try:
                # (a) early-exit when nothing in the evolution paths is dirty
                if not self._has_changes():
                    logger.debug("[lineage] no dirty files in evolution paths — skipping commit")
                    return CommitResult(success=True, commit_sha=None, error=None, duration_ms=0)

                original_branch = self._current_branch()

                if self._branch_exists():
                    # Path A: checkout existing evolution branch.
                    # Untracked files are preserved through branch switches so no
                    # stash is needed. If checkout fails (conflicting tracked file),
                    # fall back to stash-then-checkout.
                    try:
                        self._git("checkout", self.branch)
                    except subprocess.CalledProcessError:
                        stash_out = self._git("stash", "push", "--include-untracked", "-m", self._stash_marker)
                        stash_pushed = "No local changes" not in stash_out
                        self._git("checkout", self.branch)
                else:
                    # Path B: first run — orphan creation.
                    # git preserves working tree through --orphan so the
                    # agent_memory/ files remain available after the switch.
                    self._git("checkout", "--orphan", self.branch)
                    from contextlib import suppress

                    with suppress(subprocess.CalledProcessError):
                        self._git("rm", "-rf", "--cached", ".")

                # (d) stage only paths that exist (agent_skills/ may be absent)
                self._add_evolution_paths()

                # (e) commit
                msg = self._build_message(message_summary)
                self._git("commit", "-m", msg)
                sha = self._git("rev-parse", "HEAD").strip()

                # (f) restore original branch
                self._git("checkout", original_branch)

                # (g) restore dev workspace when stash was used
                if stash_pushed:
                    self._restore_stash()
                    stash_pushed = False

                duration_ms = int((time() - start) * 1000)
                logger.info("[lineage] committed to %s: %s (duration=%dms)", self.branch, sha, duration_ms)
                _set_span_attr(span, "evolution.lineage.commit_sha", sha)
                return CommitResult(success=True, commit_sha=sha, error=None, duration_ms=duration_ms)

            except Exception as exc:
                duration_ms = int((time() - start) * 1000)
                logger.warning("[lineage] commit failed (soft-fail): %s", exc, exc_info=True)
                _record_span_exc(span, exc)
                from contextlib import suppress

                # Restore original branch first so subsequent reset acts on it.
                with suppress(Exception):
                    if original_branch:
                        self._git("checkout", original_branch)
                # Clear any agent_memory/agent_skills paths that were staged by
                # _add_evolution_paths() before the commit failed — leaving them
                # staged would pollute the original branch's index.
                with suppress(Exception):
                    self._git("reset", "HEAD", "--", "agent_memory/", "agent_skills/")
                with suppress(Exception):
                    if stash_pushed:
                        self._restore_stash()
                return CommitResult(success=False, commit_sha=None, error=str(exc), duration_ms=duration_ms)

    # ------------------------------------------------------------------
    # Private git helpers
    # ------------------------------------------------------------------

    def _git(self, *args: str) -> str:
        """Run a git command in self.repo; raise CalledProcessError on failure."""
        result = subprocess.run(
            ["git", *args],
            cwd=self.repo,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout

    def _has_changes(self) -> bool:
        """Return True if agent_memory/ or agent_skills/ has uncommitted changes."""
        try:
            out = self._git("status", "--porcelain", "agent_memory/", "agent_skills/")
            return bool(out.strip())
        except subprocess.CalledProcessError:
            # If git status fails (e.g. not a repo), treat as no changes
            return False

    def _current_branch(self) -> str:
        """Return the current branch name (or HEAD if detached)."""
        return self._git("rev-parse", "--abbrev-ref", "HEAD").strip()

    def _add_evolution_paths(self) -> None:
        """Stage agent_memory/ and agent_skills/ — skipping paths that don't exist.

        git add -A with a non-existent path argument exits 128 ("pathspec did not
        match any files"), so we only pass paths that are present on disk.
        """
        paths = [p for p in ("agent_memory/", "agent_skills/") if (self.repo / p.rstrip("/")).exists()]
        if paths:
            self._git("add", "-A", *paths)

    def _branch_exists(self) -> bool:
        """Return True if self.branch already exists in the local repo."""
        try:
            out = self._git("branch", "--list", self.branch)
            return bool(out.strip())
        except subprocess.CalledProcessError:
            return False

    def _restore_stash(self) -> None:
        """Pop the stash with our marker if it exists.

        Checks stash list for the marker before popping to avoid popping
        an unrelated stash entry.
        """
        try:
            stash_list = self._git("stash", "list")
            if self._stash_marker in stash_list:
                self._git("stash", "pop")
        except subprocess.CalledProcessError as exc:
            logger.warning("[lineage] stash pop failed: %s", exc)

    # ------------------------------------------------------------------
    # Message builder (FR-L5 / FR-L6)
    # ------------------------------------------------------------------

    def _build_message(self, summary: dict) -> str:
        """Build a git commit message from a summary dict.

        Supports two shapes:
        - type="daemon"      → daemon run summary (FR-L5)
        - type="transitions" → FSM transitions batch (FR-L6)
        """
        summary_type = summary.get("type", "daemon")

        if summary_type == "transitions":
            return self._build_transitions_message(summary)
        return self._build_daemon_message(summary)

    def _build_daemon_message(self, summary: dict) -> str:
        """Build daemon run summary commit message (FR-L5)."""
        actions = summary.get("actions", [])

        # Extract per-action details
        pareto_archived = 0
        pareto_processed = 0
        regime_changed = 0
        regime_total = 0
        drafts_created = 0
        agents_checked = 0
        all_transitions: list[dict] = []

        for action in actions:
            name = action.get("name", "")
            details = action.get("details", {})
            if name == "pareto":
                pareto_archived = details.get("archived_count", 0)
                pareto_processed = details.get("processed_count", 0)
                all_transitions.extend(details.get("transitions", []))
            elif name == "regime":
                regime_changed = details.get("changed_count", 0)
                regime_total = details.get("total_count", 0)
            elif name == "skill_proposal":
                drafts = details.get("drafts_created", [])
                drafts_created = len(drafts) if isinstance(drafts, list) else drafts
                agents_checked = details.get("agents_checked", 4)

        lines = [
            "evolution: daemon run summary",
            "",
            f"Pareto: archived={pareto_archived} processed={pareto_processed}",
            f"Regime: changed={regime_changed} total={regime_total}",
            f"Skill proposal: drafts_created={drafts_created} agents_checked={agents_checked}",
        ]

        # Append transitions if any
        if all_transitions:
            lines.append("")
            lines.append(f"Transitions ({len(all_transitions)}):")
            for t in all_transitions:
                rule_id = t.get("rule_id", "?")
                agent = t.get("agent_id", "?")
                old_s = t.get("old_state", "?")
                new_s = t.get("new_state", "?")
                triggered = t.get("triggered_by", "?")
                lines.append(f"- rule_id={rule_id} agent={agent} {old_s}→{new_s} (triggered_by={triggered})")

        lines.extend(["", "Auto-Generated-By: spec-020c"])
        return "\n".join(lines)

    def _build_transitions_message(self, summary: dict) -> str:
        """Build FSM transitions batch commit message (FR-L6)."""
        transitions = summary.get("transitions", [])
        n = len(transitions)

        lines = [
            f"evolution: {n} maturity transition{'s' if n != 1 else ''}",
            "",
        ]

        for t in transitions:
            rule_id = t.get("rule_id", "?")
            agent = t.get("agent_id", "?")
            old_s = t.get("old_state", "?")
            new_s = t.get("new_state", "?")
            triggered = t.get("triggered_by", "?")
            lines.append(f"- rule_id={rule_id} agent={agent} {old_s}→{new_s} (triggered_by={triggered})")

        lines.extend(["", "Auto-Generated-By: spec-020c"])
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# OTel helpers (graceful no-op when OTel not available)
# ---------------------------------------------------------------------------


def _get_span_ctx(span_name: str) -> object:
    """Return an OTel span context manager, or nullcontext if unavailable."""
    from contextlib import nullcontext

    try:
        from opentelemetry import trace as _otel_trace

        return _otel_trace.get_tracer(__name__).start_as_current_span(span_name)
    except Exception:
        return nullcontext()


def _set_span_attr(span: object, key: str, value: object) -> None:
    """Set a span attribute; silently skip if span is unavailable."""
    if span is None:
        return
    from contextlib import suppress

    with suppress(Exception):
        span.set_attribute(key, value)  # type: ignore[union-attr]


def _record_span_exc(span: object, exc: Exception) -> None:
    """Record exception on span + set error status; silently skip if unavailable."""
    if span is None:
        return
    from contextlib import suppress

    with suppress(Exception):
        span.record_exception(exc)  # type: ignore[union-attr]
        span.set_attribute("step.status", "FAIL")  # type: ignore[union-attr]

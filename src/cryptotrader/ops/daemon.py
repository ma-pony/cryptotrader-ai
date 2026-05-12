"""spec 022 -- Evolution reflect daemon.

FR-D1: EvolutionDaemon (run_once / run_forever)
FR-D3: APScheduler AsyncIOScheduler + CronTrigger
FR-D6: Pareto rerank action
FR-D7: Regime filter action (via refilter_records_by_regime wrapper)
FR-D8: Skill proposal auto-trigger (per-agent independent)
FR-D10: Soft degrade -- LLM failures -> SKIP, algorithm steps continue
FR-D11: OTel span evolution.daemon.run + 3 child spans
FR-D12: fcntl.flock 5s timeout, alphabetical order (cases -> patterns)
"""

from __future__ import annotations

import asyncio
import fcntl
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from cryptotrader.config import EvolutionDaemonConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ActionResult:
    """Result of one reflect action (pareto / regime / skill_proposal)."""

    name: str
    status: Literal["PASS", "SKIP", "FAIL"]
    duration_ms: int
    details: dict = field(default_factory=dict)


@dataclass
class RunResult:
    """Result of a full daemon run_once() invocation."""

    actions_run: list[ActionResult]
    total_duration_ms: int
    exit_code: int


# ---------------------------------------------------------------------------
# Lock constants
# ---------------------------------------------------------------------------

_LOCK_TIMEOUT_S: float = 5.0
_LOCK_PATHS_ALPHABETICAL: list[str] = [
    "agent_memory/cases/.lock",
    "agent_memory/patterns/.lock",
]


# ---------------------------------------------------------------------------
# EvolutionDaemon
# ---------------------------------------------------------------------------


class EvolutionDaemon:
    """Daily reflect daemon: Pareto rerank + regime re-tag + skill proposal.

    Usage:
        daemon = EvolutionDaemon(config=cfg.evolution_daemon)
        result = await daemon.run_once()          # single run (--once)
        await daemon.run_forever()                # blocks on APScheduler cron
    """

    def __init__(self, config: EvolutionDaemonConfig) -> None:
        self.config = config
        self._scheduler = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_once(self) -> RunResult:
        """Execute all configured reflect actions once; return RunResult.

        FR-D11: parent span evolution.daemon.run
        FR-D12: acquire file locks before running actions
        FR-D10: SKIP steps that raise LLM exceptions; always exit 0
        """
        t_start = time.monotonic()
        with _get_span_ctx("evolution.daemon.run"):
            try:
                acquired, lock_fds = await _try_acquire_locks(_LOCK_PATHS_ALPHABETICAL, _LOCK_TIMEOUT_S)
            except Exception:
                logger.warning("[evolution-daemon] unexpected error acquiring locks", exc_info=True)
                acquired, lock_fds = False, []

            if not acquired:
                logger.warning(
                    "[evolution-daemon] could not acquire file locks within %.1fs — skipping this run",
                    _LOCK_TIMEOUT_S,
                )
                total_ms = int((time.monotonic() - t_start) * 1000)
                return RunResult(actions_run=[], total_duration_ms=total_ms, exit_code=0)

            try:
                results: list[ActionResult] = []
                for action_name in self.config.actions:
                    result = await self._run_action(action_name)
                    results.append(result)
            finally:
                _release_locks(lock_fds)

            total_ms = int((time.monotonic() - t_start) * 1000)
            # soft degrade: SKIP is not failure; only explicit FAIL would set exit_code=1
            exit_code = 0
            for r in results:
                if r.status == "FAIL":
                    exit_code = 1
                    break

            run_result = RunResult(
                actions_run=results,
                total_duration_ms=total_ms,
                exit_code=exit_code,
            )
            # FR-D13/FR-D14: record metrics events to Redis for Prometheus gauges
            _record_run_metrics(run_result)

            # spec 020c FR-L4: auto-commit changed agent_memory/ + agent_skills/ to evolution branch
            _commit_lineage(run_result)

            return run_result

    async def run_forever(self) -> None:
        """Start APScheduler CronTrigger loop; blocks until SIGTERM/SIGINT.

        FR-D3: AsyncIOScheduler + CronTrigger from crontab string.
        spec 020c FR-L11: SIGTERM/SIGINT graceful shutdown via loop.add_signal_handler.
        Waits for the current run_once() to finish before shutting down (wait=True).
        """
        import signal

        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.cron import CronTrigger

        self._scheduler = AsyncIOScheduler()
        self._scheduler.add_job(
            self.run_once,
            CronTrigger.from_crontab(self.config.cron, timezone="UTC"),
        )
        self._scheduler.start()
        logger.info("[evolution-daemon] scheduler started; cron=%s", self.config.cron)

        # spec 020c: graceful shutdown flag + signal handlers
        self._shutdown_flag = asyncio.Event()
        loop = asyncio.get_running_loop()

        def _on_signal() -> None:
            logger.info("[evolution-daemon] shutdown signal received, waiting for current run_once to finish")
            self._shutdown_flag.set()

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, _on_signal)

        await self._shutdown_flag.wait()

        # Graceful shutdown: wait=True lets the current run_once job finish (≤ 30s)
        logger.info("[evolution-daemon] scheduler shutdown (wait=True) ...")
        self._scheduler.shutdown(wait=True)
        logger.info("[evolution-daemon] scheduler shutdown complete")

        # Close Redis connection if available
        try:
            from cryptotrader.observability.daemon_metrics import _get_redis

            rc = _get_redis()
            if rc is not None:
                rc.close()
                logger.info("[evolution-daemon] redis closed")
        except Exception:
            logger.info("[evolution-daemon] redis close skipped", exc_info=True)

        # Flush OTel provider if available
        try:
            from opentelemetry import trace as _otel_trace

            provider = _otel_trace.get_tracer_provider()
            if hasattr(provider, "shutdown"):
                provider.shutdown()
                logger.info("[evolution-daemon] OTel flush complete, exit 0")
        except Exception:
            logger.info("[evolution-daemon] OTel flush skipped", exc_info=True)

    # ------------------------------------------------------------------
    # Internal: action dispatch + soft degrade wrapper
    # ------------------------------------------------------------------

    async def _run_action(self, name: str) -> ActionResult:
        """Run one named action; catch LLM/network errors and return SKIP.

        FR-D10: soft degrade -- OpenAI / TimeoutError / network -> SKIP
        FR-D11: child span evolution.daemon.<name>
        """
        span_ctx = _get_span_ctx(f"evolution.daemon.{name}")
        with span_ctx as span:
            t0 = time.monotonic()
            try:
                result = await self._dispatch_action(name, t0)
                _set_span_attrs(span, result.status, result.duration_ms)
                return result
            except Exception as exc:
                duration_ms = int((time.monotonic() - t0) * 1000)
                return self._handle_action_exc(exc, name, duration_ms, span)

    async def _dispatch_action(self, name: str, t0: float) -> ActionResult:
        """Dispatch to the appropriate action method by name."""
        if name == "pareto":
            return await self._action_pareto()
        if name == "regime":
            return await self._action_regime()
        if name == "skill_proposal":
            return await self._action_skill_proposal()
        if name == "pattern_extraction":
            return await self._action_pattern_extraction()
        logger.warning("[evolution-daemon] unknown action '%s' -- skipping", name)
        duration_ms = int((time.monotonic() - t0) * 1000)
        return ActionResult(
            name=name,
            status="SKIP",
            duration_ms=duration_ms,
            details={"reason": "unknown action"},
        )

    def _handle_action_exc(self, exc: Exception, name: str, duration_ms: int, span: object) -> ActionResult:
        """Handle exception from an action: soft-degrade -> SKIP, else FAIL."""
        skip_reason = _classify_soft_degrade(exc)
        if skip_reason:
            logger.warning(
                "[evolution-daemon] action '%s' soft-degrade (%s): %s",
                name,
                skip_reason,
                exc,
            )
            _record_span_exc(span, exc, "SKIP")
            return ActionResult(
                name=name,
                status="SKIP",
                duration_ms=duration_ms,
                details={"reason": skip_reason, "error": str(exc)},
            )
        logger.error("[evolution-daemon] action '%s' FAIL: %s", name, exc, exc_info=True)
        _record_span_exc(span, exc, "FAIL")
        return ActionResult(
            name=name,
            status="FAIL",
            duration_ms=duration_ms,
            details={"error": str(exc)},
        )

    # ------------------------------------------------------------------
    # FR-D6: Pareto rerank action
    # ------------------------------------------------------------------

    async def _action_pareto(self) -> ActionResult:
        """Pareto frontier rerank; dominated rules → maturity=archived.

        FR-D6 clarify Q1: only non-frontier members (dominated by at least
        one frontier rule) are archived; frontier members stay active.
        """
        from cryptotrader.agents.skills._constants import DEFAULT_AGENT_MEMORY_DIR, VALID_AGENT_IDS
        from cryptotrader.learning.evolution._io import _load_pattern_from_path, _save_pattern_to_path

        t0 = time.monotonic()
        archived_count = 0
        processed_count = 0
        transitions: list[dict] = []

        for agent_id in sorted(VALID_AGENT_IDS):
            patterns_dir = DEFAULT_AGENT_MEMORY_DIR / agent_id / "patterns"
            if not patterns_dir.exists():
                continue

            pattern_files = [f for f in sorted(patterns_dir.glob("*.md")) if not f.name.startswith(".")]
            active_rules = []
            for f in pattern_files:
                p = _load_pattern_from_path(f)
                if p is not None and p.maturity == "active":
                    active_rules.append(p)

            if not active_rules:
                continue

            # Compute frontier: non-dominated members stay active; dominated -> archived
            frontier_ids = _compute_frontier_ids(active_rules)

            for rule in active_rules:
                processed_count += 1
                if id(rule) not in frontier_ids:
                    old_maturity = rule.maturity
                    rule.maturity = "archived"
                    if rule.file_path is not None:
                        try:
                            _save_pattern_to_path(rule, rule.file_path)
                            archived_count += 1
                            logger.info("[pareto] archived rule: %s/%s", agent_id, rule.name)
                            # spec 020c T012: collect transition for lineage batch commit
                            transitions.append(
                                {
                                    "rule_id": rule.name,
                                    "agent_id": agent_id,
                                    "old_state": old_maturity,
                                    "new_state": "archived",
                                    "triggered_by": "pareto_dominated",
                                }
                            )
                        except Exception:
                            logger.warning("[pareto] failed to save archived rule %s", rule.name, exc_info=True)

        duration_ms = int((time.monotonic() - t0) * 1000)
        logger.info(
            "[pareto] PASS — processed=%d archived=%d duration_ms=%d", processed_count, archived_count, duration_ms
        )
        return ActionResult(
            name="pareto",
            status="PASS",
            duration_ms=duration_ms,
            details={
                "archived_count": archived_count,
                "processed_count": processed_count,
                "transitions": transitions,
            },
        )

    # ------------------------------------------------------------------
    # FR-D7: Regime filter action
    # ------------------------------------------------------------------

    async def _action_regime(self) -> ActionResult:
        """Re-tag all cases with current regime; return changed count.

        FR-D7: calls refilter_records_by_regime() thin public wrapper.
        """
        from cryptotrader.learning.memory import refilter_records_by_regime

        t0 = time.monotonic()
        changed_count = refilter_records_by_regime()
        duration_ms = int((time.monotonic() - t0) * 1000)
        logger.info("[regime] PASS — changed=%d duration_ms=%d", changed_count, duration_ms)
        return ActionResult(
            name="regime",
            status="PASS",
            duration_ms=duration_ms,
            details={"changed_count": changed_count},
        )

    # ------------------------------------------------------------------
    # FR-D8: Skill proposal action
    # ------------------------------------------------------------------

    async def _action_skill_proposal(self) -> ActionResult:
        """Per-agent skill proposal; triggers only when active_rules >= threshold.

        FR-D8 clarify Q2: 4 agents independent; single daemon can trigger 0-4 calls.
        FR-D9: propose_new_skill writes .draft only (no auto-save).
        """
        from cryptotrader.agents.skills._constants import DEFAULT_AGENT_MEMORY_DIR
        from cryptotrader.learning.skill_proposal import _load_active_patterns_for_agent, propose_new_skill

        t0 = time.monotonic()
        drafts_created: list[str] = []
        agents_checked = 0
        threshold = self.config.propose_threshold

        for agent_id in ["tech", "chain", "news", "macro"]:
            agents_checked += 1
            active_patterns = _load_active_patterns_for_agent(agent_id, DEFAULT_AGENT_MEMORY_DIR)
            if len(active_patterns) >= threshold:
                logger.info(
                    "[skill_proposal] agent=%s active_rules=%d >= threshold=%d — proposing",
                    agent_id,
                    len(active_patterns),
                    threshold,
                )
                try:
                    draft_path = propose_new_skill(scope=f"agent:{agent_id}")
                    if draft_path is not None:
                        drafts_created.append(str(draft_path))
                        logger.info("[skill_proposal] draft created: %s", draft_path)
                except Exception:
                    # LLM failure inside propose_new_skill bubbles up to _run_action soft degrade
                    raise
            else:
                logger.debug(
                    "[skill_proposal] agent=%s active_rules=%d < threshold=%d — skipping",
                    agent_id,
                    len(active_patterns),
                    threshold,
                )

        duration_ms = int((time.monotonic() - t0) * 1000)
        logger.info(
            "[skill_proposal] PASS — agents_checked=%d drafts_created=%d duration_ms=%d",
            agents_checked,
            len(drafts_created),
            duration_ms,
        )
        return ActionResult(
            name="skill_proposal",
            status="PASS",
            duration_ms=duration_ms,
            details={"drafts_created": drafts_created, "agents_checked": agents_checked},
        )

    # ------------------------------------------------------------------
    # spec 021: Pattern extraction action
    # ------------------------------------------------------------------

    async def _action_pattern_extraction(self) -> ActionResult:
        """从 cases 蒸馏 patterns（cold-start + maturity FSM 更新）。

        spec 021 FR-P8: 调 distill_patterns(cycles_window=cfg.experience.lookback_commits)
        FR-P11: 异常时 ActionResult(status=SKIP)（soft degrade，与 FR-D10 一致）
        """
        from cryptotrader.config import load_config
        from cryptotrader.learning.memory import distill_patterns

        t0 = time.monotonic()
        try:
            cfg = load_config()
            run = distill_patterns(cycles_window=cfg.experience.lookback_commits)
            duration_ms = int((time.monotonic() - t0) * 1000)
            logger.info(
                "[pattern_extraction] PASS — new=%d updated=%d archived=%d cases=%d duration_ms=%d",
                run.patterns_created,
                run.patterns_updated,
                run.patterns_archived,
                run.cases_processed,
                duration_ms,
            )
            return ActionResult(
                name="pattern_extraction",
                status="PASS",
                duration_ms=duration_ms,
                details={
                    "new_count": run.patterns_created,
                    "updated_count": run.patterns_updated,
                    "archived_count": run.patterns_archived,
                    "cases_processed": run.cases_processed,
                },
            )
        except Exception as exc:
            duration_ms = int((time.monotonic() - t0) * 1000)
            logger.warning("[pattern_extraction] SKIP due to exception: %s", exc, exc_info=True)
            return ActionResult(
                name="pattern_extraction",
                status="SKIP",
                duration_ms=duration_ms,
                details={"reason": "exception", "error": str(exc)},
            )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _classify_soft_degrade(exc: BaseException) -> str | None:
    """Return a soft-degrade reason string if exc is an LLM/network error, else None.

    FR-D10: OpenAIAPIError / TimeoutError / network errors → SKIP (soft degrade).
    """
    exc_type_name = type(exc).__name__
    module = type(exc).__module__ or ""

    # openai errors
    if "openai" in module.lower() or "OpenAI" in exc_type_name:
        return "openai_api_error"

    # asyncio / concurrent / builtin TimeoutError (Py 3.11+ unifies them)
    if isinstance(exc, TimeoutError):
        return "timeout"

    # httpx / network errors
    if "httpx" in module.lower() and "NetworkError" in exc_type_name:
        return "network_error"
    if exc_type_name in ("NetworkError", "ConnectError", "RemoteProtocolError"):
        return "network_error"

    # LangChain / LangOpenAI wrapper
    if "langchain" in module.lower() and "error" in exc_type_name.lower():
        return "langchain_llm_error"

    return None


def _compute_frontier_ids(rules: list) -> set[int]:
    """Return set of id() for rules on the Pareto frontier (layer 0 = not dominated).

    Uses the same dominance logic as pareto.py _dominates().
    A rule is on the frontier iff no other rule dominates it.
    """
    from cryptotrader.learning.evolution.pareto import (  # type: ignore[attr-defined]
        _confidence_proxy,
        _dominates,
        _win_rate,
    )

    if not rules:
        return set()

    scored = [(r, _win_rate(r), _confidence_proxy(r)) for r in rules]
    frontier: set[int] = set()

    for i, (r_i, wr_i, cp_i) in enumerate(scored):
        dominated = False
        for j, (_, wr_j, cp_j) in enumerate(scored):
            if i == j:
                continue
            if _dominates(wr_j, cp_j, wr_i, cp_i):
                dominated = True
                break
        if not dominated:
            frontier.add(id(r_i))

    return frontier


def _prepare_lock_file(lp_str: str):  # type: ignore[return]
    """Ensure lock file exists and return an open writable file object.

    Synchronous helper — called from async _try_acquire_locks to avoid
    ASYNC230/ASYNC240 violations (pathlib.Path methods must not be used
    directly inside async functions per ruff ASYNC230/ASYNC240 rules).
    Returns an open IO object, or raises OSError on failure.
    """
    lp = Path(lp_str)
    lp.parent.mkdir(parents=True, exist_ok=True)
    if not lp.exists():
        lp.touch()
    return lp.open("w")


async def _try_acquire_locks(lock_paths: list[str], timeout_s: float) -> tuple[bool, list]:
    """Try to acquire fcntl.flock exclusive locks in order; 5s timeout each.

    FR-D12: alphabetical order (cases → patterns); timeout → skip run.
    spec 020c FR-L10: async def + await asyncio.sleep (replaces sync time.sleep).
    Returns (acquired: bool, open_fds: list[IO]).
    """
    import errno

    fds = []
    for lp_str in sorted(lock_paths):  # alphabetical order enforced
        try:
            fd = _prepare_lock_file(lp_str)
            fds.append(fd)
        except OSError as exc:
            logger.warning("[evolution-daemon] cannot open lock file %s: %s", lp_str, exc)
            _release_locks(fds)
            return False, []

        # Poll-based flock with timeout (fcntl.LOCK_NB + await asyncio.sleep)
        deadline = time.monotonic() + timeout_s
        acquired_this = False
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired_this = True
                break
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK):
                    logger.warning("[evolution-daemon] flock error on %s: %s", lp_str, exc)
                    _release_locks(fds)
                    return False, []
                if time.monotonic() >= deadline:
                    logger.warning("[evolution-daemon] lock timeout on %s after %.1fs", lp_str, timeout_s)
                    _release_locks(fds)
                    return False, []
                await asyncio.sleep(0.1)

        if not acquired_this:
            _release_locks(fds)
            return False, []

    return True, fds


def _release_locks(fds: list) -> None:
    """Release all fcntl locks and close file descriptors."""
    from contextlib import suppress

    for fd in fds:
        with suppress(Exception):
            fcntl.flock(fd, fcntl.LOCK_UN)
        with suppress(Exception):
            fd.close()


# ---------------------------------------------------------------------------
# OTel helpers (gracefully no-op when OTel not available)
# ---------------------------------------------------------------------------


def _get_span_ctx(span_name: str) -> object:
    """Return an OTel span context manager, or nullcontext if OTel unavailable."""
    from contextlib import nullcontext

    try:
        from opentelemetry import trace as _otel_trace

        return _otel_trace.get_tracer(__name__).start_as_current_span(span_name)
    except Exception:
        return nullcontext()


def _set_span_attrs(span: object, status: str, duration_ms: int) -> None:
    """Set step.status and step.duration_ms on span; silently skip if unavailable."""
    if span is None:
        return
    from contextlib import suppress

    with suppress(Exception):
        span.set_attribute("step.status", status)  # type: ignore[union-attr]
        span.set_attribute("step.duration_ms", duration_ms)  # type: ignore[union-attr]


def _record_span_exc(span: object, exc: Exception, status: str) -> None:
    """Record exception and set step.status on span; silently skip if unavailable."""
    if span is None:
        return
    from contextlib import suppress

    with suppress(Exception):
        span.record_exception(exc)  # type: ignore[union-attr]
        span.set_attribute("step.status", status)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# FR-D13/FR-D14: post-run Redis metrics recording
# ---------------------------------------------------------------------------


def _record_run_metrics(run_result: RunResult) -> None:
    """Record daemon run events to Redis for Prometheus gauge lazy-update.

    FR-D13: DaemonRunCountAggregator / DaemonLLMFailureAggregator /
            SkillProposalDraftAggregator via Redis sorted sets.
    Silently no-ops when Redis is unavailable (all helpers guard internally).
    """
    from contextlib import suppress

    with suppress(Exception):
        from cryptotrader.observability.daemon_metrics import (
            record_draft_event,
            record_llm_failure_event,
            record_run_event,
        )

        record_run_event()

        has_llm_skip = any(a.status == "SKIP" for a in run_result.actions_run)
        record_llm_failure_event(failed=has_llm_skip)

        for action in run_result.actions_run:
            if action.name == "skill_proposal":
                for _ in action.details.get("drafts_created", []):
                    record_draft_event()


# ---------------------------------------------------------------------------
# spec 020c FR-L4: lineage auto-commit after each run_once()
# ---------------------------------------------------------------------------


def _build_lineage_summary(run_result: RunResult) -> dict:
    """Build the summary dict for GitLineageHook.commit_changes().

    T007: constructs type="daemon" summary with actions list including
    per-action details; _action_pareto transitions are included in details.
    """
    actions = [{"name": a.name, "status": a.status, "details": a.details} for a in run_result.actions_run]
    return {"type": "daemon", "actions": actions}


def _commit_lineage(run_result: RunResult) -> None:
    """No-op: spec 020c lineage auto-commit was disabled 2026-05-12.

    GitLineageHook.commit_changes() repeatedly left the working tree stuck on
    the `evolution` orphan branch when its `git commit` failed (pre-commit
    hooks, merge conflicts on auto-bumped SKILL.md telemetry, etc.) and the
    soft-fail path did not restore the original branch. Operator instruction
    is to keep all relevant code on main and stop branch-bouncing. Telemetry
    (access_count, last_accessed_at) stays as in-place file edits on main;
    commit them manually if/when the diff is meaningful.
    """
    # Still record a "skipped" lineage event so the dashboard gauge ticks.
    from contextlib import suppress

    with suppress(Exception):
        from cryptotrader.observability.daemon_metrics import record_lineage_event

        record_lineage_event(success=True)

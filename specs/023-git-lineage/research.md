# Phase 0：研究与决策

**关联 spec**：[spec.md](spec.md)
**关联 brainstorm**：[brainstorm/08-spec-020c-git-lineage.md](../../brainstorm/08-spec-020c-git-lineage.md)
**Date**: 2026-05-09

## 概述

5 项 brainstorm 决策 + 3 项 clarify + 5 项 spot-check 已消除全部 ambiguity。本文档记录最终决定 + 实施细节。

## 5 项关键决策（来自 brainstorm）

| # | 决策 | 来源 |
|---|---|---|
| Q1 触发范围 | B daemon + maturity transitions | 噪声 vs 可追溯平衡 |
| Q2 branch 模型 | B 独立 evolution branch（不 merge 回 main） | 干净 audit log |
| Q3 commit 作者 | C author=current user + trailer | 不引入新 identity |
| Q4 P2 advisory 收尾 | B 3 项高 ROI 子集（asyncio.sleep / SIGTERM / a11y） | 与 lineage 主题耦合 |
| Q5 失败处理 | B Soft fail + 改动保留 | 与 spec 020a/020b 一致 |

## 3 项 clarify 决策

| # | Question | Answer |
|---|---|---|
| C1 | dev 在 main 工作时 lineage 切 evolution branch 路径 | 用 git stash --include-untracked --keep-index 保护 dev workspace |
| C2 | SIGTERM 中途 run_once 处理 | 等当前 action 完成（≤30s 延迟），不中途 abort |
| C3 | 单 cycle 多 transitions 提交粒度 | batch 1 commit（与 daemon 单 commit 模式一致） |

## 5 项 spot-check 结果（2026-05-09）

| # | 检查项 | 结果 |
|---|---|---|
| 1 | spec 018 FSM transitions 写入路径 | ✓ src/cryptotrader/learning/evolution/fsm.py 4 transitions（promotions / archived） |
| 2 | daemon `time.sleep(0.1)` 现状 | ✓ daemon.py:458 sync 调用 |
| 3 | run_forever signal handler 现状 | ✓ daemon.py:146 `asyncio.Event().wait()` 无 SIGTERM |
| 4 | SkillsGrid badge a11y 现状 | ⚠ 0 个 aria-label |
| 5 | git config user | ✓ 已配 |

## 实施细节决策

### Decision 1：GitLineageHook 类骨架

```python
# src/cryptotrader/ops/lineage.py
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Literal

from opentelemetry import trace as _otel_trace

_tracer = _otel_trace.get_tracer(__name__)


@dataclass
class CommitResult:
    success: bool
    commit_sha: str | None
    error: str | None
    duration_ms: int


class GitLineageHook:
    def __init__(self, branch: str = "evolution", repo_path: Path | None = None) -> None:
        self.branch = branch
        self.repo = repo_path or Path.cwd()
        self._stash_marker = "spec-020c-lineage-stash"

    def commit_changes(self, message_summary: dict) -> CommitResult:
        with _tracer.start_as_current_span("evolution.lineage.commit") as span:
            start = time()
            try:
                # (a) 检测 dirty
                if not self._has_changes():
                    return CommitResult(success=True, commit_sha=None, error=None, duration_ms=0)

                original_branch = self._current_branch()
                # (b) stash dev 改动
                self._git("stash", "push", "--include-untracked", "--keep-index", "-m", self._stash_marker)
                # (c) checkout evolution branch（orphan 创建若不存在）
                self._ensure_branch()
                # (d-e) add + commit
                self._git("add", "-A", "agent_memory/", "agent_skills/")
                msg = self._build_message(message_summary)
                self._git("commit", "-m", msg)
                sha = self._git("rev-parse", "HEAD").strip()
                # (f) 切回原 branch
                self._git("checkout", original_branch)
                # (g) stash pop（即便没 stash 也不抛错；用 list + drop 检查）
                self._restore_stash()

                span.set_attribute("evolution.lineage.commit_sha", sha)
                return CommitResult(success=True, commit_sha=sha, error=None, duration_ms=int((time()-start)*1000))
            except Exception as exc:
                span.record_exception(exc)
                # 尽力恢复
                try:
                    self._git("checkout", original_branch)
                    self._restore_stash()
                except Exception:
                    pass
                return CommitResult(success=False, commit_sha=None, error=str(exc), duration_ms=int((time()-start)*1000))

    def _git(self, *args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=self.repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    def _has_changes(self) -> bool:
        return bool(self._git("status", "--porcelain", "agent_memory/", "agent_skills/").strip())

    def _current_branch(self) -> str:
        return self._git("rev-parse", "--abbrev-ref", "HEAD").strip()

    def _ensure_branch(self) -> None:
        try:
            self._git("checkout", self.branch)
        except subprocess.CalledProcessError:
            self._git("checkout", "--orphan", self.branch)
            self._git("rm", "-rf", "--cached", ".")  # 清空 orphan 索引

    def _restore_stash(self) -> None:
        result = self._git("stash", "list").strip()
        if self._stash_marker in result:
            self._git("stash", "pop")

    def _build_message(self, summary: dict) -> str:
        # 见 Decision 2
        ...
```

**Rationale**：
- subprocess 路径不引入 gitpython，与项目已有 dependency 风格一致
- 异常路径覆盖 stash 失败 / branch 切换失败 / commit 失败，全部走 soft fail
- `_has_changes()` 限定 agent_memory/ + agent_skills/ 路径，避免误 commit 其他 working tree 改动

### Decision 2：Commit Message 模板

**daemon run summary**：
```
evolution: daemon run summary

Pareto: archived=12 processed=50
Regime: changed=33 total=142
Skill proposal: drafts_created=2 agents_checked=4

Auto-Generated-By: spec-020c
```

**FSM transitions batch**：
```
evolution: 5 maturity transitions

- rule_id=foo agent=tech active→archived (triggered_by=fundamental_streak)
- rule_id=bar agent=macro probationary→active (triggered_by=pnl_threshold)
- rule_id=baz agent=chain observed→probationary (triggered_by=initial_pattern)
- rule_id=qux agent=news active→archived (triggered_by=fundamental_streak)
- rule_id=zap agent=tech probationary→archived (triggered_by=loss_streak)

Auto-Generated-By: spec-020c
```

**Rationale**：trailer `Auto-Generated-By: spec-020c` 在 message 末尾（git 标准 trailer 位置）；首行短 summary 便于 `git log --oneline`；body 列具体 details 便于 `git log --format=%B` 详查。

### Decision 3：Daemon 集成

```python
# src/cryptotrader/ops/daemon.py（既有 modify）
async def run_once(self) -> RunResult:
    # ... 原有 actions 执行 ...
    result = RunResult(actions_run=results, total_duration_ms=..., exit_code=...)

    # spec 020c 新增：lineage commit
    from cryptotrader.ops.lineage import GitLineageHook
    from cryptotrader.observability.daemon_metrics import record_lineage_event

    summary = {
        "type": "daemon",
        "actions": [{"name": a.name, "status": a.status, "details": a.details} for a in results],
    }
    commit_result = GitLineageHook(branch="evolution").commit_changes(summary)
    record_lineage_event(success=commit_result.success)

    return result

async def _try_acquire_locks(self) -> ...:
    # ... 重试循环 ...
    while time.time() - start < timeout:
        try:
            ...
        except BlockingIOError:
            await asyncio.sleep(0.1)  # spec 020c 修复（原 time.sleep）

async def run_forever(self) -> None:
    self._scheduler = AsyncIOScheduler()
    self._scheduler.add_job(...)
    self._scheduler.start()

    # spec 020c 新增：SIGTERM/SIGINT handler
    self._shutdown_flag = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: self._shutdown_flag.set())

    await self._shutdown_flag.wait()
    # graceful shutdown
    self._scheduler.shutdown(wait=True)  # wait 当前 run_once 完成
    # close redis / flush OTel ...
```

**Rationale**：
- `loop.add_signal_handler` 是 asyncio 推荐路径（vs `signal.signal`），与 asyncio loop 集成更好
- `_scheduler.shutdown(wait=True)` 等当前 job 完成（满足 SIGTERM 不中途 abort 约束）

### Decision 4：FSM Transitions Batch 触发点

spec 018 fsm.py transitions 由调用方收集（典型在 `evaluate_node` / daemon `_action_pareto`）。本 spec **不**在 fsm.py 内嵌 lineage hook（避免 fsm 模块依赖 ops），而是在调用方 batch 收集后触发：

```python
# src/cryptotrader/learning/evolution/fsm.py（既有 modify — 仅加可选返回字段，不改算法）
# 实际无需修改 fsm.py；transition 收集在 daemon._action_pareto 中。

# src/cryptotrader/ops/daemon.py
async def _action_pareto(self) -> ActionResult:
    # ... rank_rules + archive 逻辑 ...
    transitions = []
    for rule in active:
        if rule.id not in frontier:
            old_maturity = rule.maturity
            rule.maturity = "archived"
            transitions.append({
                "rule_id": rule.id,
                "agent_id": rule.agent_id,
                "old_state": old_maturity,
                "new_state": "archived",
                "triggered_by": "pareto_dominated",
            })
            save_pattern(rule)

    # transitions list 通过 ActionResult.details 透传到 lineage hook
    return ActionResult(name="pareto", status="PASS", details={"transitions": transitions, "archived_count": len(transitions)})
```

**Rationale**：
- 不修改 spec 018 fsm.py 内部算法，仅扩展调用方 transitions 收集
- 收集到 daemon `RunResult.actions_run` 后，单 commit 包含 daemon 触发的所有 transitions（Q3 batch 模式）
- 非 daemon 路径触发的 transitions（如 trader manual CLI）不在本 spec 范围（保持 daemon-only）

### Decision 5：SkillsGrid aria-label

```tsx
// web/src/pages/memory/components/SkillsGrid.tsx（既有 modify）
{skill.regime_tags.map((tag) => (
  <Badge key={tag} variant="default" aria-label={`Regime: ${tag}`}>
    {tag}
  </Badge>
))}

{skill.triggers_keywords.slice(0, 5).map((kw) => (
  <Badge key={kw} variant="outline" aria-label={`Trigger keyword: ${kw}`}>
    {kw}
  </Badge>
))}

{skill.inference_failed && (
  <Badge variant="destructive" aria-label="Inference failed during proposal">
    Inference Failed
  </Badge>
)}
```

**Rationale**：3 类 badge 各自独立 aria-label；屏幕阅读器朗读模板 `"Regime: high_funding"` 比纯 tag value 更清晰。

### Decision 6：2 Lineage Aggregator

```python
# src/cryptotrader/observability/daemon_metrics.py（既有 modify）

class LineageCommitCountAggregator:
    """24h sliding window count of evolution commits."""
    # 复用 spec 020a CacheMetricsAggregator 模式

class LineageCommitFailureAggregator:
    """24h sliding window failure rate of lineage commits."""

LINEAGE_COMMIT_AGG = LineageCommitCountAggregator()
LINEAGE_FAILURE_AGG = LineageCommitFailureAggregator()


def record_lineage_event(*, success: bool) -> None:
    LINEAGE_COMMIT_AGG.record()
    LINEAGE_FAILURE_AGG.record(failed=not success)
```

```python
# src/api/routes/metrics.py（既有 modify）
EVOLUTION_COMMIT_COUNT_GAUGE = Gauge(
    "evolution_commit_count_24h",
    "Evolution branch commit count in last 24h",
)
EVOLUTION_COMMIT_FAILURE_RATE_GAUGE = Gauge(
    "evolution_commit_failure_rate_24h",
    "Evolution lineage commit failure rate in last 24h",
)
```

**Rationale**：与 spec 020a/020b aggregator 模式一致（deque + Lock + sliding window）；不引入新 dependency。

## Phase 0 检查项

- [x] 所有 NEEDS CLARIFICATION 已解决
- [x] 所有 dependency 已识别 best practice
- [x] 所有 integration 已找到 pattern

Phase 0 输出完成。

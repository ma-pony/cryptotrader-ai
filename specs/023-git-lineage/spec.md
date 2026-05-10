# Feature Specification: Spec 020c — Git Lineage（trilogy 终段）

**Feature Branch**: `023-git-lineage`
**Created**: 2026-05-09
**Status**: Draft
**Input**: User description: "spec-020c-git-lineage — trilogy 终段。落地 D-ENG-02 git lineage：daemon + maturity FSM transitions auto-commit 到独立 evolution branch，commit message 含 Auto-Generated-By trailer，soft fail。同时收尾 3 项跨 spec P2 advisory（asyncio.sleep / SIGTERM handler / SkillsGrid a11y aria-label）。"

## Background

trilogy 收尾终段。trilogy 已合并 main：
- spec 017a/b（PromptBuilder + 4 agent 集成）
- spec 018（Memory Evolution + FSM transitions）
- spec 019（Skill Evolution + SkillsGrid）
- spec 020a（Trilogy Ops — cache 观测 + 部署）
- spec 020b（Evolution Daemon — 独立 docker-compose service）

spec 020b daemon 跑会修改 `agent_memory/patterns/`、`agent_memory/cases/`、`agent_skills/<name>/SKILL.md.draft` 文件，但**目前没有审计追溯**：daemon 跑完文件就变了，git 不会 auto-commit。如 daemon 误 archive rules 或 regime cluster 计算错误，无法回查"什么时候、哪个 commit、archive 了哪条 rule"。

spec 016 D-ENG-02 git lineage 决策：daemon + maturity transitions auto-commit 到**独立 `evolution` branch**（不污染 main）。本 spec 落地该决策。

同时本 spec 收尾 3 项跨 spec P2 advisory（asyncio.sleep / SIGTERM / a11y），均为**有生产 ROI**的修复（性能 / 数据完整 / 合规）。

直接删旧不留 fallback。Markdown 简体中文。

## Clarifications

### Session 2026-05-09

- Q: lineage hook 切换 evolution branch 时如果 dev 在 main 工作（patterns 文件被 IDE 打开 / 改动），如何处理？→ A: 用 `git stash --include-untracked --keep-index` 保护 dev 改动，commit 完后 `git stash pop` 恢复；避免污染 dev workspace
- Q: SIGTERM 在 daemon `run_once()` 中途收到（如 pareto 完 regime 跑一半）时如何处理？→ A: 等当前 action 跑完再 graceful shutdown（最坏 30s 延迟），不中途 abort 改动
- Q: FSM transitions 单 cycle 多个时（如 daemon Pareto rerank 多 rule archived）per-transition 1 commit 还是 batch 1 commit？→ A: batch 1 commit（与 spec 020b daemon 单 commit 模式一致），message 列每个 transition 详情

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Daemon Auto-Commit（Priority: P1）

作为 SRE，daemon 每次跑完应该 auto-commit 改动到独立 `evolution` branch，commit message 列出 actions_run summary（archived rules 数量 / 改 regime tags 数量 / 创建 .draft 数量），让我能用 `git log evolution` 回溯所有 daemon 行为。

**Why this priority**：spec 020b daemon 已落地但无审计追溯，事故时无法回查。git lineage 是 trader-grade 审计 SLA 前置。

**Independent Test**：跑 `arena evolution-daemon --once` → `git log evolution -1` 有新 commit + message 含 `Auto-Generated-By: spec-020c` trailer + 含 actions_run details。

**Acceptance Scenarios**：

1. **Given** daemon `--once` 跑完含 archived rules，**When** 检查 `git log evolution -1`，**Then** commit message 含 "archived_count: N" 行 + trailer
2. **Given** evolution branch 不存在，**When** daemon 首次 commit，**Then** branch orphan 创建（不继承 main 历史）
3. **Given** daemon 跑完无任何文件改动，**When** 触发 lineage hook，**Then** 不创建空 commit（idempotent）

---

### User Story 2 - FSM Transitions Lineage（Priority: P1）

作为审计员，maturity FSM 状态变化（observed→probationary / probationary→active / active→archived / probationary→archived）应该单独 commit 到 evolution branch，让我能精确定位"rule X 在什么时间点从 active 转 archived"。

**Why this priority**：spec 018 既有 transitions 是已识别 evolution 事件；以 transition 粒度提交便于精确审计；与 daemon batch commit 互补。

**Independent Test**：mock 1 个 transition（rule_id=foo, active→archived）→ 触发 lineage hook → `git log evolution --grep="rule_id: foo"` ≥ 1 hit。

**Acceptance Scenarios**：

1. **Given** rule maturity 从 active → archived，**When** transition 触发 lineage hook，**Then** evolution branch 有新 commit message 含 rule_id + old_state + new_state + triggered_by
2. **Given** 单 cycle 内多个 transitions（如 daemon Pareto rerank 多 rule archived），**When** lineage hook 触发，**Then** 全部 transitions 合并为 1 commit（batch 模式）
3. **Given** transition 触发 commit 失败（merge conflict），**When** 检查状态，**Then** 改动保留 + OTel error span + daemon exit 0（soft fail）

---

### User Story 3 - Daemon Asyncio + SIGTERM（Priority: P1）

作为 daemon 维护者，spec 020b 留下的 P2 advisory：(a) `_try_acquire_locks` 用 sync `time.sleep(0.1)` 阻塞 event loop；(b) `run_forever()` 无 SIGTERM handler，docker stop 时 daemon 直接被 kill 数据可能丢失。本 spec 修复。

**Why this priority**：(a) 影响 daemon 与 trading cycle 共部署时的 event loop 响应；(b) 影响 daemon graceful shutdown，docker 重启时 OTel span / redis events 可能丢失。

**Independent Test**：(a) `grep "time.sleep" src/cryptotrader/ops/daemon.py` 0 hits；(b) 启动 daemon 后发 SIGTERM → daemon 在 ≤ 5s 内 graceful exit + OTel span flushed。

**Acceptance Scenarios**：

1. **Given** daemon `_try_acquire_locks` lock 被占用，**When** 重试等待，**Then** 用 `await asyncio.sleep(0.1)` 不阻塞 event loop
2. **Given** daemon `run_forever()` 跑中，**When** 进程收 SIGTERM，**Then** ≤ 5s 内 graceful shutdown：APScheduler.shutdown() + redis.close() + OTel flush
3. **Given** SIGINT（Ctrl+C），**When** dev 模式停止 daemon，**Then** 同样 graceful exit

---

### User Story 4 - SkillsGrid a11y Aria-Label（Priority: P2）

作为前端用户（含视障用户），SkillsGrid 当前 3 类 badge（regime / triggers / inference_failed）缺 aria-label，屏幕阅读器无法朗读 badge 含义。本 spec 加 aria-label。

**Why this priority**：a11y 合规（WCAG 2.1）；spec 019 P2-02 advisory 既已标注。

**Independent Test**：`grep "aria-label" web/src/pages/memory/components/SkillsGrid.tsx` ≥ 3 hits + Vitest 单测验证 badge 含 `aria-label` 属性。

**Acceptance Scenarios**：

1. **Given** skill 含 regime_tags badges，**When** SkillsGrid 渲染，**Then** 每个 regime badge 含 `aria-label="Regime: <tag-value>"`
2. **Given** skill 含 triggers_keywords badges，**When** 渲染，**Then** 每个 triggers badge 含 `aria-label="Trigger: <keyword>"`
3. **Given** skill `inference_failed=true`，**When** 渲染，**Then** inference_failed indicator 含 `aria-label="Inference failed during proposal"`

---

### Edge Cases

- evolution branch 已存在 → daemon 不重新创建，正常 commit
- evolution branch 在 git checkout 切换失败 → daemon 写 OTel error + soft fail 不阻塞 actions
- daemon 改动 0 文件（all PASS but nothing changed）→ 不创建空 commit
- transition 写盘 + lineage hook 之间被 interrupt → 改动保留，下次 daemon 触发时 lineage hook 检测 dirty 仍 commit
- daemon 跑时 dev 在 main branch 工作 → lineage hook 用 `git stash` + `git checkout evolution` + commit + `git checkout -` + `git stash pop` 路径不影响 dev
- SIGTERM 在 daemon `run_once()` 中途收到 → 等当前 action 跑完再 graceful shutdown（不中途 abort 改动）
- `evolution` branch 远程 push 路径 → 本 spec **不**自动 push（用户 manual push）
- daemon commit 跨多 cycle 累积改动 → 单 commit 含所有累积，不强制每 cycle 1 commit

## Requirements *(mandatory)*

### Functional Requirements

#### Lineage Module

- **FR-L1**：`src/cryptotrader/ops/lineage.py` 新模块 MUST 存在，含 `GitLineageHook` 类（`__init__(branch="evolution", repo_path=None)` + `commit_changes(message_summary: dict) -> CommitResult`）
- **FR-L2**：`GitLineageHook` MUST 用 `subprocess.run(["git", ...])` 调用 git；不引入 gitpython 依赖
- **FR-L3**：`commit_changes()` MUST：(a) 检测 working tree dirty（`git status --porcelain`）；(b) `git stash --include-untracked --keep-index` 保护 dev 改动；(c) 切换到 evolution branch（不存在时 orphan 创建）；(d) `git add -A agent_memory/ agent_skills/`（仅 evolution 数据路径）；(e) `git commit -m "<message>"`；(f) 切回原 branch；(g) `git stash pop` 恢复 dev 改动；(h) 返回 CommitResult

#### Daemon 集成

- **FR-L4**：`src/cryptotrader/ops/daemon.py:run_once()` 末尾（actions 全部跑完后，记录 metrics 后）MUST 调 `GitLineageHook(branch="evolution").commit_changes(actions_summary)`
- **FR-L5**：commit message 模板：
  ```
  evolution: daemon run summary

  Pareto: archived=N processed=M
  Regime: changed=N total=M
  Skill proposal: drafts_created=N agents_checked=4

  Auto-Generated-By: spec-020c
  ```

#### FSM Transitions 集成

- **FR-L6**：`src/cryptotrader/learning/evolution/fsm.py` 4 transition 写入路径（promotions / archived）调用方 MUST 在 transition 完成后触发 `GitLineageHook.commit_changes(transition_summary)`；**单 cycle 内多个 transitions batch 1 commit**（与 spec 020b daemon 单 commit 模式一致）；transition 触发 commit message 模板：
  ```
  evolution: rule_id=<id> <old_state>→<new_state>

  Triggered by: <triggered_by>
  Agent: <agent_id>

  Auto-Generated-By: spec-020c
  ```

#### Branch + Author

- **FR-L7**：commit MUST 到 `evolution` branch；branch 不存在时用 `git checkout --orphan evolution` 创建（不继承 main 历史；干净 audit log）
- **FR-L8**：commit author 继承 `git config user.{name,email}`（不改写）；trailer `Auto-Generated-By: spec-020c` 唯一标识来源

#### 失败处理

- **FR-L9**：commit 失败（merge conflict / hook reject / permission）时 MUST：(a) `git checkout -` 恢复原 branch；(b) 改动保留在 working tree；(c) 写 OTel `evolution.lineage.commit_failed` error span；(d) 返回 `CommitResult(success=False, error=...)` 让 daemon 继续 exit 0

#### Daemon P2 修复

- **FR-L10**：`src/cryptotrader/ops/daemon.py:_try_acquire_locks` 中 `time.sleep(0.1)` MUST 改为 `await asyncio.sleep(0.1)`；`_try_acquire_locks` 改 `async def`
- **FR-L11**：`run_forever()` MUST 加 SIGTERM + SIGINT signal handler：(a) `signal.signal(SIGTERM, ...)`；(b) handler 设置 shutdown flag；(c) **等当前 `run_once()` 调用完成**（最坏 30s 延迟，与 daemon `run_once` ≤ 30s 性能约束一致），不中途 abort 改动；(d) 当前 action 跑完后调用 `_scheduler.shutdown(wait=False)` + `redis.close()` + `_otel_provider.shutdown()`；(e) ≤ 30s 内 exit

#### SkillsGrid a11y

- **FR-L12**：`web/src/pages/memory/components/SkillsGrid.tsx` 3 类 badge MUST 加 `aria-label`：
  - regime badge：`aria-label={`Regime: ${tag}`}`
  - triggers badge：`aria-label={`Trigger: ${keyword}`}`
  - inference_failed indicator：`aria-label="Inference failed during proposal"`

#### Lineage Monitoring

- **FR-L13**：`src/cryptotrader/observability/daemon_metrics.py` MUST 加 2 个 sliding window aggregator：`LineageCommitCountAggregator` (24h) / `LineageCommitFailureAggregator` (24h)
- **FR-L14**：`src/api/routes/metrics.py` MUST 注册 2 个新 Prometheus Gauge：`evolution_commit_count_24h` / `evolution_commit_failure_rate_24h`

### Key Entities

- **CommitResult**：dataclass（`success: bool` / `commit_sha: str | None` / `error: str | None` / `duration_ms: int`）
- **LineageMessageSummary**：dict 含 actions_run 详情（archived_count / changed_count / drafts_created）或 transition 详情（rule_id / old_state / new_state / triggered_by / agent_id）

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-L1**：`uv run arena evolution-daemon --once` 跑完后 `git log evolution -1 --format=%B` 输出含 "Auto-Generated-By: spec-020c"
- **SC-L2**：`git log evolution --grep="Auto-Generated-By" --oneline` 输出 ≥ 1 行
- **SC-L3**：`grep -n "time.sleep" src/cryptotrader/ops/daemon.py` 返回空（仅 `await asyncio.sleep` 存在）
- **SC-L4**：daemon `run_forever()` 收 SIGTERM 后 ≤ 30s graceful shutdown（等当前 run_once 完成；pytest 验证 OTel span flushed + redis closed + scheduler shutdown）
- **SC-L5**：`grep "aria-label" web/src/pages/memory/components/SkillsGrid.tsx` ≥ 3 hits
- **SC-L6**：commit 失败时 daemon exit 0 + working tree 改动保留（pytest 验证）
- **SC-L7**：spec 014 / 015 / 17a / 17b / 18 / 19 / 20a / 20b 既有测试不回归（baseline 2439 → ≥ 2439 pass / 0 fail）
- **SC-L8**：`/spex:review-spec` 无 P0 / P1 issues
- **SC-L9**：`/spex:review-plan` 任务覆盖完整 + REVIEW-PLAN.md 生成
- **SC-L10**：本 spec 单 PR ≤ 4 commit（C1 文档 + C2 lineage + daemon fix + C3 frontend + C4 E2E gate）

## Assumptions

- spec 020b daemon 已合并 main 且 `EvolutionDaemon.run_once()` 入口稳定
- spec 018 FSM transitions 路径稳定（fsm.py 4 transitions 不重写）
- git config user.{name,email} 已配置（dev 机 + docker 容器都有）
- evolution branch 命名约定：本 spec 固定 `evolution`（不参数化）
- 用户接受不自动 push 远程（manual periodic push）
- daemon 在 docker 容器内的工作路径与 host repo path 一致（mount 路径）
- subprocess `git` 调用在 daemon 进程内可用（docker base image 含 git）
- pytest-asyncio 在 dev deps（spec 018 既有）
- spec 020a observability aggregator 模式（deque + Lock + sliding window）可复用
- SkillsGrid 既有 React 组件结构（spec 019 / 020a）稳定，仅加 props 不重构

## Dependencies

**Upstream**：
- spec 010（OpenTelemetry tracing）
- spec 015（metrics endpoint）
- spec 018（FSM transitions）
- spec 019（SkillsGrid 组件）
- spec 020a（observability aggregator pattern）
- spec 020b（EvolutionDaemon）

**Downstream**：无（trilogy 终段）

**External tooling**：无新 runtime 依赖（subprocess / signal / git CLI 已存在）

## Out of Scope

- ❌ 全 9 项 P2 advisory 收尾（仅做 3 项 ROI 子集；其余独立 fix PR）
- ❌ Per-cycle evaluate_node auto-commit（brainstorm Q1 B 决定）
- ❌ Skill `.draft` save 触发 commit（保留 human gate）
- ❌ `evolution` branch 自动 merge / push 远程（manual operation）
- ❌ GPG sign / multi-author signing（author=current git user）
- ❌ Lineage 告警 alertmanager（仅 dashboard）
- ❌ Git LFS / submodule（数据文件 < 1MB）
- ❌ gitpython 依赖（用 subprocess git CLI）
- ❌ Branch 命名参数化（固定 `evolution`）
- ❌ Lineage 历史可视化前端（CLI `git log evolution` 已足够）

## Reversibility

本 spec 落地后可通过 git revert 单 PR 回退（无 schema 变更，无数据迁移）。回退后：
- `src/cryptotrader/ops/lineage.py` 删除（不影响生产 daemon 运行）
- `daemon.py:run_once()` 移除 lineage hook 调用（不影响 reflect actions）
- `daemon.py:_try_acquire_locks` 恢复 sync `time.sleep`（功能不变，恢复 event loop 阻塞 advisory 状态）
- `run_forever()` 移除 signal handler（恢复 docker stop kill 风险）
- SkillsGrid 移除 aria-label（恢复 a11y advisory 状态）
- `evolution` branch 历史保留（不删除，作为只读 audit log）
- 2 lineage Prometheus Gauge 移除（不影响 spec 020a/020b 既有 gauge）

## Implementation Outline

### 单 PR 切 4 commit（与 spec 019 / 020a / 020b 同 pattern）

**C1 — 基础设施 + 文档**：
- `src/cryptotrader/ops/lineage.py`（新 GitLineageHook 类骨架）
- `src/cryptotrader/observability/daemon_metrics.py`（修改 — 加 2 lineage aggregator）
- `src/api/routes/metrics.py`（修改 — 注册 2 lineage Gauge）

**C2 — Lineage 集成 + daemon P2 修复**：
- `src/cryptotrader/ops/daemon.py`（修改 — `run_once()` 末尾调 lineage hook + `_try_acquire_locks` 改 async + `run_forever()` 加 signal handler）
- `src/cryptotrader/learning/evolution/fsm.py`（修改 — 调用方加 lineage hook trigger）
- 单测（pytest-asyncio）

**C3 — 前端 a11y**：
- `web/src/pages/memory/components/SkillsGrid.tsx`（修改 — 3 类 badge 加 aria-label）
- `web/src/pages/memory/components/SkillsGrid.test.tsx`（修改 — 加 a11y 断言）

**C4 — E2E + 最终 Gate**：
- `tests/test_e2e_git_lineage.py`（mocked daemon cycle 验证 evolution branch commit + SIGTERM graceful + soft fail）
- `pyproject.toml`（per-file-ignores 如需）
- grep / pytest / ruff 全部 gate

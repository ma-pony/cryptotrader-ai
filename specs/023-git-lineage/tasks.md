# Tasks: Spec 020c — Git Lineage

**Branch**: `023-git-lineage` | **Spec**: [spec.md](spec.md) | **Plan**: [plan.md](plan.md)

## Phase 1: Setup（无新依赖）

复用 stdlib subprocess / signal / 既有 OTel + Prometheus；无新 dependency。

## Phase 2: Foundational

- [x] T001 [P] 创建 `src/cryptotrader/ops/lineage.py`：`GitLineageHook` 类骨架（`__init__` + `commit_changes` + 私有 helper `_git` / `_has_changes` / `_current_branch` / `_ensure_branch` / `_restore_stash`，按 research.md Decision 1）
- [x] T002 [P] 在 `src/cryptotrader/ops/lineage.py` 实现 `_build_message(summary: dict) -> str` 模板（按 research.md Decision 2，支持 type=daemon / type=transitions）
- [x] T003 [US4] 在 `src/cryptotrader/observability/daemon_metrics.py` 加 `LineageCommitCountAggregator` (24h) + `LineageCommitFailureAggregator` (24h)（复用 spec 020a 模式）
- [x] T004 [US4] 在 `src/cryptotrader/observability/daemon_metrics.py` 加 `record_lineage_event(*, success: bool)` 入口函数 + module-level singletons
- [x] T005 [US4] 修改 `src/api/routes/metrics.py`：注册 `EVOLUTION_COMMIT_COUNT_GAUGE` + `EVOLUTION_COMMIT_FAILURE_RATE_GAUGE`；`/metrics` endpoint lazy update from aggregator

---

## Phase 3: User Story 1 — Daemon Auto-Commit（P1）

**Goal**：daemon `run_once()` 末尾 auto-commit 到 evolution branch，message 含 actions_run summary + trailer。

**Independent Test**：跑 `arena evolution-daemon --once` → `git log evolution -1` 含 trailer + actions details。

- [x] T006 [US1] 修改 `src/cryptotrader/ops/daemon.py:run_once()`：actions 跑完后调 `GitLineageHook(branch="evolution").commit_changes(summary)`；调 `record_lineage_event(success=...)` 写 metrics
- [x] T007 [US1] 在 `daemon.py:run_once()` 构造 daemon summary dict（type="daemon" + actions list）传给 lineage hook
- [x] T008 [P] [US1] 创建 `tests/test_lineage.py`：`test_commit_changes_creates_orphan_evolution_branch` 用例（首次跑 orphan 创建 + commit 含 trailer）
- [x] T009 [P] [US1] 在 `tests/test_lineage.py` 加 `test_commit_changes_with_no_changes` 用例（无 dirty → 不创建空 commit）
- [x] T010 [P] [US1] 在 `tests/test_lineage.py` 加 `test_commit_changes_protects_dev_workspace` 用例（main branch dev 改动 stash + 恢复路径）
- [x] T011 [P] [US1] 在 `tests/test_lineage.py` 加 `test_commit_failure_soft_fail` 用例（mock subprocess CalledProcessError → result.success=False + working tree 改动保留）

---

## Phase 4: User Story 2 — FSM Transitions Lineage（P1）

**Goal**：daemon `_action_pareto()` 收集 transitions，batch commit 到 evolution branch；message 列每条 transition。

**Independent Test**：mock pareto rerank 5 transitions → daemon 跑完 → `git log evolution --grep="rule_id"` 出现 5 行。

- [x] T012 [US2] 修改 `src/cryptotrader/ops/daemon.py:_action_pareto()`：archive 时收集 transition dict（rule_id / agent_id / old_state / new_state / triggered_by="pareto_dominated"）到 `ActionResult.details["transitions"]`
- [x] T013 [US2] 在 `daemon.py:run_once()` 处理 ActionResult.details["transitions"] 累积到 daemon summary（同 commit 含 transitions list）
- [x] T014 [P] [US2] 创建 `tests/test_daemon_lineage_integration.py`：`test_daemon_pareto_archives_recorded_in_transitions` 用例（5 rules archived → transitions list 长度 5）
- [x] T015 [US2] 在 `tests/test_daemon_lineage_integration.py` 加 `test_daemon_run_once_commits_with_transitions` 用例（daemon 跑完 → evolution branch 含 1 commit + message 含 5 transitions）

---

## Phase 5: User Story 3 — Daemon Asyncio + SIGTERM（P1）

**Goal**：`_try_acquire_locks` 改 async；`run_forever()` 加 SIGTERM/SIGINT signal handler。

**Independent Test**：`grep "time.sleep" daemon.py` 0 hits；mock SIGTERM → daemon 在 ≤ 30s 内 graceful exit。

- [x] T016 [US3] 修改 `src/cryptotrader/ops/daemon.py:_try_acquire_locks`：`def` → `async def`，`time.sleep(0.1)` → `await asyncio.sleep(0.1)`
- [x] T017 [US3] 修改 `daemon.py` 所有 `_try_acquire_locks()` 调用方加 `await`
- [x] T018 [US3] 修改 `daemon.py:run_forever()`：用 `loop.add_signal_handler(SIGTERM/SIGINT, ...)` 设置 shutdown flag；handler 调用 `_scheduler.shutdown(wait=True)` 等当前 run_once 完成；`redis.close()` + OTel flush
- [x] T019 [P] [US3] 创建 `tests/test_daemon_signal_handler.py`：`test_run_forever_sigterm_graceful_shutdown` 用例（asyncio task + 模拟 SIGTERM + 验证 30s 内 exit）
- [x] T020 [US3] 在 `tests/test_daemon_signal_handler.py` 加 `test_sigterm_during_run_once_waits_for_completion` 用例（SIGTERM 在 run_once 中途 → 等当前 action 完成）
- [x] T021 [US3] 跑 `grep -n "time.sleep" src/cryptotrader/ops/daemon.py` 校验返回空（仅 await asyncio.sleep）

---

## Phase 6: User Story 4 — SkillsGrid a11y（P2）

**Goal**：3 类 badge 加 aria-label。

**Independent Test**：grep aria-label ≥ 3 hits + Vitest 测试 PASS。

- [ ] T022 [P] [US4] 修改 `web/src/pages/memory/components/SkillsGrid.tsx`：regime badge 加 `aria-label={\`Regime: ${tag}\`}`
- [ ] T023 [P] [US4] 在 SkillsGrid.tsx triggers badge 加 `aria-label={\`Trigger keyword: ${kw}\`}`
- [ ] T024 [P] [US4] 在 SkillsGrid.tsx inference_failed indicator 加 `aria-label="Inference failed during proposal"`
- [ ] T025 [US4] 修改 `web/src/pages/memory/components/SkillsGrid.test.tsx`：加 `describe("a11y")` block 含 3 个用例（验证 regime / triggers / inference_failed badge 各含 aria-label 属性）

---

## Phase 7: Polish & Cross-Cutting

- [ ] T026 [P] 创建 `tests/test_e2e_git_lineage.py`：mocked daemon cycle 端到端测试（daemon → evolution branch 创建 → commit 含 trailer → metrics gauge update）
- [ ] T027 跑 `uv run python -m pytest tests/ --no-cov 2>&1 | tail -3` 验证 ≥ 2439 passed / 0 failed（SC-L7）
- [ ] T028 跑 `uv run ruff check src/cryptotrader/ops/lineage.py src/cryptotrader/observability/daemon_metrics.py src/cryptotrader/ops/daemon.py tests/test_lineage.py tests/test_daemon_signal_handler.py tests/test_daemon_lineage_integration.py tests/test_e2e_git_lineage.py` clean
- [ ] T029 跑 SC-L1：`uv run arena evolution-daemon --once` 后 `git log evolution -1 --format=%B` 含 "Auto-Generated-By: spec-020c"
- [ ] T030 跑 SC-L2：`git log evolution --grep="Auto-Generated-By" --oneline` ≥ 1
- [ ] T031 跑 SC-L3：`grep -n "time.sleep" src/cryptotrader/ops/daemon.py` 返回空
- [ ] T032 跑 SC-L5：`grep -c "aria-label" web/src/pages/memory/components/SkillsGrid.tsx` ≥ 3
- [ ] T033 跑 `git log --oneline 023-git-lineage..main | wc -l` ≤ 4 commit 校验（SC-L10）

---

## Dependencies

```
Phase 2 Foundational (T001-T005: lineage module + aggregator + Gauge)
   ↓
US1 (Daemon commit)         T006-T011 (依赖 T001-T005)
US2 (Transitions batch)     T012-T015 (依赖 T006 daemon 集成)
US3 (Async + SIGTERM)       T016-T021 (依赖 T005 daemon 既有)
US4 (a11y)                  T022-T025 (无 backend 依赖)
   ↓
Phase 7 Polish              T026-T033
```

T001 (`GitLineageHook`) 是 US1 + US2 的前置依赖。
T006 (daemon 集成) 必须在 T012 (transitions 收集) 前。
T016 (async lock) 必须在其他 daemon-touching task 前完成（避免 merge conflict）。

US4（前端）独立，可并行启动。

## Parallel Execution

US1 内部：T008 / T009 / T010 / T011 可并行（独立 test 用例）
US4 内部：T022 / T023 / T024 可并行（同文件不同行）

跨 US：US1 + US4 完全独立可并行；US2 / US3 与 daemon.py 耦合需顺序。

## Implementation Strategy

### MVP scope

**最小可发布**：T001-T011 (lineage 基建 + daemon commit + transitions 占位)。**不**含 US3（SIGTERM）+ US4（a11y）也可独立发布。建议同 PR 全交付（trilogy 终段一气收尾）。

### 4-commit 切分

| commit | 涵盖 task | 说明 |
|---|---|---|
| C1 | T001-T005 | lineage 模块 + aggregator + Gauge（纯新增）|
| C2 | T006-T021 | daemon 集成（lineage commit + transitions batch + async lock + SIGTERM）|
| C3 | T022-T025 | 前端 a11y（独立改动）|
| C4 | T026-T033 | E2E + 最终 gate |

### 增量交付

落地顺序建议：
1. C1 优先（纯新增无回归风险）
2. C2 次（daemon 集成 + 3 P2 修复）
3. C3 再次（前端独立）
4. C4 最后（gate + 全套回归）

## Validation

任务总数：33
- Foundational: 5 task
- US1: 6 task
- US2: 4 task
- US3: 6 task
- US4: 4 task
- Polish: 8 task

每 user story 含 independent test；每 task 含具体路径；checklist format 全部合规。

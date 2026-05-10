# Plan Review: Spec 020c — Git Lineage

**Spec dir**: specs/023-git-lineage/
**Date**: 2026-05-09
**Reviewer**: Claude (spex:review-plan)

## Overall Assessment

**Status**: ✅ SOUND

**Summary**：plan + tasks 结构合规，33 task 覆盖 14 FR + 10 SC，无 P0/P1 issues。MVP 切分清晰（4 commit），4 user story 间通过 lineage 模块串联（US4 完全独立）。可进入 implement。

## 0. Scope Check

4 user story 均围绕 trilogy 收尾终段：
- US1+US2 lineage 主体（daemon + transitions）
- US3 daemon P2 fix（asyncio.sleep + SIGTERM）— 与 lineage 主题相关（SIGTERM 影响 lineage 完整性）
- US4 frontend a11y（spec 019 advisory 收尾）— 独立但与 trilogy 主题对齐

不跨独立 subsystem。US4 前端独立可并行。

## 1. Task Quality Enforcement

### 检查项

- ✓ Actionable：33 task 全部含明确动作动词
- ✓ Testable：所有 task 含具体 file path + acceptance check
- ✓ Atomic：单 task 单产出
- ✓ Ordered：T001 lineage 类骨架前置；T006 集成在 T012 前；T016 async lock 在其他 daemon task 前
- ✓ 文件路径具体
- ✓ Phase 顺序合理

### 文件结构映射

plan.md `Source Code` 段列出 5 个文件改动 + 4 个新文件，每个文件单一责任：
- `lineage.py` — GitLineageHook 类
- `daemon_metrics.py` — 2 lineage aggregator
- `metrics.py` — 2 Prometheus Gauge
- `daemon.py` — run_once 集成 + async lock + SIGTERM
- `SkillsGrid.tsx` — 3 类 badge aria-label

无 vague utils / helpers 文件。

## 2. Coverage Matrix

### Functional Requirements

| FR | Story | Tasks | Status |
|---|---|---|---|
| FR-L1（lineage.py + GitLineageHook 类） | US1 | T001 | ✓ |
| FR-L2（subprocess git 不引 gitpython） | US1 | T001 | ✓ |
| FR-L3（commit_changes 8 步路径含 stash 保护） | US1 | T001 | ✓ |
| FR-L4（daemon run_once 末尾调 lineage） | US1 | T006 / T007 | ✓ |
| FR-L5（commit message 模板 + trailer） | US1 | T002 | ✓ |
| FR-L6（FSM transitions batch + message 模板） | US2 | T012 / T013 | ✓ |
| FR-L7（evolution branch orphan 创建） | US1 | T001（_ensure_branch）| ✓ |
| FR-L8（author=current user + trailer） | US1 | T002（trailer in template）| ✓ |
| FR-L9（commit 失败 soft fail） | US1 | T011 | ✓ |
| FR-L10（_try_acquire_locks await asyncio.sleep） | US3 | T016 | ✓ |
| FR-L11（run_forever SIGTERM handler） | US3 | T018 | ✓ |
| FR-L12（SkillsGrid 3 类 aria-label） | US4 | T022 / T023 / T024 | ✓ |
| FR-L13（2 lineage aggregator） | Foundational | T003 / T004 | ✓ |
| FR-L14（2 Prometheus Gauge） | Foundational | T005 | ✓ |

### Success Criteria

| SC | Tasks | Verification | Status |
|---|---|---|---|
| SC-L1（git log evolution 含 trailer） | T029 | shell exec | ✓ |
| SC-L2（git log --grep 输出 ≥ 1） | T030 | shell exec | ✓ |
| SC-L3（grep time.sleep 空） | T021 / T031 | shell grep | ✓ |
| SC-L4（SIGTERM ≤ 30s graceful） | T019 / T020 | pytest | ✓ |
| SC-L5（aria-label ≥ 3 hits） | T032 | shell grep | ✓ |
| SC-L6（commit fail soft fail） | T011 | pytest | ✓ |
| SC-L7（≥ 2439 passed） | T027 | pytest | ✓ |
| SC-L8（review-spec 无 P0/P1） | （已 PASS） | REVIEW-SPEC.md | ✓ |
| SC-L9（review-plan + REVIEW-PLAN.md） | （本文档） | this | ✓ |
| SC-L10（≤ 4 commit）| T033 | git log | ✓ |

### Edge cases coverage

| Edge case | Task | Status |
|---|---|---|
| evolution branch 已存在 | T001（_ensure_branch try/except）| ✓ |
| 0 改动空 commit | T009 | ✓ |
| dev workspace 保护 | T010 | ✓ |
| commit 失败 soft fail | T011 | ✓ |
| SIGTERM 中途等当前 action | T020 | ✓ |
| `evolution` branch 不 push 远程 | （Out of Scope 显式声明）| ✓ |

全部 14 FR + 10 SC + 6 edge case 100% 覆盖。

## 3. Red Flag Scanning

- ✓ 无 vague task
- ✓ 无 monster task
- ✓ 无 missing file paths
- ✓ Phase 顺序合理
- ✓ 无跨 spec 依赖泄漏（不修改 spec 018 fsm.py 内部算法，仅扩展调用方收集）
- ✓ NFR 已落到 task（commit ≤ 500ms / SIGTERM ≤ 30s / a11y 无 render 开销）

## 4. NFR Validation

- ✓ 性能：commit_changes ≤ 500ms（subprocess git overhead）；SIGTERM ≤ 30s
- ✓ 内存：sliding window deque 自管理 + redis sorted set evict
- ✓ 并发：subprocess 阻塞但 daemon 独立 service 不影响 trading cycle
- ✓ 可观测性：OTel + Prometheus（2 新 Gauge）+ structlog 三层
- ✓ 可回滚：spec.md Reversibility 段显式覆盖

## 5. Recommendations

### Critical (Must Fix Before Implementation)
无

### Important (Should Fix)
无

### Optional (Nice to Have)
- [ ] T001 `_restore_stash()` 实现细节（如何检测 stash 存在）可在 implement 时细化（research.md Decision 1 已提供 list + grep marker 路径）
- [ ] T018 SIGTERM handler 与 asyncio loop 集成的具体形式（add_signal_handler vs signal.signal）已在 research.md Decision 3 提供推荐路径

## 6. MVP & Incremental Strategy

- ✓ MVP scope：US1 + US3 是 trilogy 终段刚需；US2（transitions batch）+ US4（a11y）是增量
- ✓ 4 commit 切分（C1 基建 / C2 daemon 集成 / C3 前端 / C4 gate）— 与 spec 019/020a/020b 一致
- ✓ US4 完全独立可并行实施

## Conclusion

plan + tasks 结构完整、覆盖率 100%、无 red flag、MVP 切分清晰。可进入 `/speckit-implement`。

**Ready for implementation**: Yes

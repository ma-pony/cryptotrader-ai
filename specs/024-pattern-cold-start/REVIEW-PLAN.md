# Plan Review: Spec 021 — Pattern Cold-Start

**Spec dir**: specs/024-pattern-cold-start/
**Date**: 2026-05-11
**Reviewer**: Claude (spex:review-plan)

## Overall Assessment

**Status**: ✅ SOUND

**Summary**：plan + tasks 结构合规，23 task 覆盖 13 FR + 10 SC，无 P0/P1 issues。MVP 切分清晰（4 commit）。3 US 完全独立可并行。Ready for implement。

## 0. Scope Check

3 US 围绕同一 trilogy 数据链 cold-start gap 的 3 个切面（distill / daemon / CLI）。共享底层 helper（`_make_pattern_slug` + `_create_pattern_from_cases`）。不跨独立 subsystem。

## 1. Task Quality

- ✓ Actionable：23 task 全部明确动作动词
- ✓ Testable：所有 task 含具体 file path
- ✓ Atomic：单 task 单产出
- ✓ Ordered：helper T003/T004 在 distill T005 前；daemon action T009 在 dispatch T010 前；CLI T013 在 T014 前
- ✓ 文件路径具体
- ✓ Phase 顺序合理

### 文件结构映射

plan.md `Source Code` 段列出 5 个文件改动 + 5 个新测试，每文件单一责任：
- `memory.py` — distill + helpers
- `daemon.py` — _action_pattern_extraction
- `config.py` — ExperienceConfig
- `default.toml` — config defaults
- `cli/main.py` — typer command

无 vague utils。

## 2. Coverage Matrix

### Functional Requirements

| FR | Story | Tasks | Status |
|---|---|---|---|
| FR-P1（distill cold-start 路径）| US1 | T005 | ✓ |
| FR-P2（PatternRecord 字段填充）| US1 | T004 | ✓ |
| FR-P3（向后兼容已有 patterns）| US1 | T005（cold-start 路径独立）| ✓ |
| FR-P4（失败 isolated）| US1 | T005（try/except 每 pattern）| ✓ |
| FR-P5（OTel span 写入）| US1 | T006 | ✓ |
| FR-P6（ExperienceConfig 加字段）| Foundational | T001 | ✓ |
| FR-P7（TOML 加默认值）| Foundational | T002 | ✓ |
| FR-P8（daemon action 方法）| US2 | T009 | ✓ |
| FR-P9（actions list 加 entry）| US2 | T011 | ✓ |
| FR-P10（dispatch 分支）| US2 | T010 | ✓ |
| FR-P11（soft degrade）| US2 | T009（既有 try/except in _run_action）| ✓ |
| FR-P12（CLI typer command）| US3 | T013 / T014 | ✓ |
| FR-P13（异常 exit 1）| US3 | T013（typer.Exit(1)）| ✓ |

### Success Criteria

| SC | Tasks | Verification | Status |
|---|---|---|---|
| SC-P1（CLI distill exit 0 + ≥1 patterns）| T019 | shell + count | ✓ |
| SC-P2（≥ 3 patterns 文件）| T020 | shell find + wc | ✓ |
| SC-P3（API /memory/rules total > 0）| T021 | curl + jq | ✓ |
| SC-P4（daemon --once 4 actions PASS）| T022 | shell | ✓ |
| SC-P5（5 distill 单测 PASS）| T008 + T017 | pytest | ✓ |
| SC-P6（e2e PASS）| T016 + T017 | pytest | ✓ |
| SC-P7（≥ 2458 pass）| T017 | pytest | ✓ |
| SC-P8（review-spec 无 P0/P1）| （已 PASS）| REVIEW-SPEC.md | ✓ |
| SC-P9（review-plan + REVIEW-PLAN.md）| （本文档）| this | ✓ |
| SC-P10（≤ 4 commit）| T023 | git log | ✓ |

### Edge cases coverage

| Edge case | Task | Status |
|---|---|---|
| cases 目录空 → 不抛异常 | T008（empty cases 用例）| ✓ |
| 单 case PnL=None | T008（pnl all None 用例）| ✓ |
| 同名 pattern 已存在 → 跳过 | T007（collision 用例 + 实际 distill 路径检查 file exists）| ✓ |
| daemon SKIP soft degrade | T012（异常 SKIP 用例）| ✓ |
| CLI + daemon 文件锁不冲突 | spec 020c FR-L12 既有（无新 task）| ✓ |
| 跨 agent 同名 applied_pattern | T005（per-agent 路径，无冲突）| ✓ |
| LLM 不可用 | spec 设计纯算法（无 LLM 调用）| ✓ |

全部 13 FR + 10 SC + 7 edge case 100% 覆盖。

## 3. Red Flag Scanning

- ✓ 无 vague task
- ✓ 无 monster task
- ✓ 无 missing file paths
- ✓ Phase 顺序合理
- ✓ 无跨 spec 依赖泄漏

## 4. NFR Validation

- ✓ 性能：distill cold-start 200 cases ≤ 2s（纯文本 parse + 字典聚合 + 文件写入；spec text 显式声明）
- ✓ 内存：cases list 一次性 load，预估 < 50MB
- ✓ 并发：daemon + CLI 共用 fcntl.flock（spec 020c FR-L12 既有）
- ✓ 可观测性：OTel span + structlog（FR-P5）
- ✓ 可回滚：spec.md Reversibility 段显式

## 5. Recommendations

### Critical / Important
无

### Optional
- [ ] T008 用例可加 OTel span 验证（attr 写入 `learning.distill.cold_start`），plan 阶段可保留
- [ ] T016 e2e 验证 dashboard /memory 页前端渲染（OOS 但可 backlog）

## 6. MVP

- ✓ MVP scope：T001-T008（Phase 2 + US1）— distill cold-start 单独可发布
- ✓ 4 commit 切分清晰（C1 cold-start / C2 daemon+CLI / C3 e2e / C4 gate）
- ✓ US1/US2/US3 完全独立可并行

## Conclusion

plan + tasks 完整覆盖、无 red flag、MVP 切分清晰。可进入 `/speckit-implement`。

**Ready for implementation**: Yes

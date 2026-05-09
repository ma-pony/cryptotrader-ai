# Plan Review: Spec 020a — Trilogy Ops

**Spec dir**: specs/021-trilogy-ops/
**Date**: 2026-05-09
**Reviewer**: Claude (spex:review-plan)

## Overall Assessment

**Status**: ✅ SOUND

**Summary**：plan.md + tasks.md 结构合规，40 task 覆盖 20 FR + 11 SC，无 P0/P1 issues。MVP 切分清晰（4 commit），5 user story 间无依赖可并行。可进入 implement 阶段。

## 0. Scope Check

5 个 user story 覆盖**同一 trilogy ops 子域**的不同切面（脚本 / 文档 / 后端 telemetry / async 化 / 前端 + 失败标志），共享既有基建（OTel / Prometheus / pytest / Vitest）。**不**跨独立 subsystem，无需拆分 spec。

US 之间相互独立可并行 → 增加并行度（已在 tasks.md "Parallel Execution" 段说明），但不构成"应拆 spec"信号。

## 1. Task Quality Enforcement

### 检查项

- ✓ Actionable：40 task 全部含明确动作动词（创建 / 修改 / 跑 / 加 / 验证）
- ✓ Testable：所有 task 含具体 file path + 可执行 acceptance check
- ✓ Atomic：单 task 单产出（无 "T001 创建 + 修改 + 测试" 拼盘）
- ✓ Ordered：依赖在 `Dependencies` 段标注（T017 依赖 T015，T022 → T024，etc.）
- ✓ 文件路径具体：无 "somewhere" / "TBD" / 通用 utils 文件
- ✓ Phase 顺序合理（Setup→Foundational→US1-5→Polish）
- ✓ 无 task 重复（grep 检查无 duplicate file path / action 组合）

### 文件结构映射

plan.md `Source Code` 段列出 12 个文件改动 + 6 个新文件，每个文件单一责任：
- `base.py` — log_llm_usage cache attr
- `ive.py` — classify_case async
- `skill_metadata_inference.py` — failure flag
- `metrics.py` — Prometheus gauge
- `cache_metrics.py` / `ive_metrics.py` — sliding window aggregator（专用职责）
- `staging_validate.py` — smoke check 入口
- `rollback-trilogy.md` — runbook 文档
- `SkillsGrid.tsx` — triggers badges
- `metrics/index.tsx` — panel 渲染

无 vague utils / helpers 文件。

## 2. Coverage Matrix

### Functional Requirements

| FR | Story | Tasks | Status |
|---|---|---|---|
| FR-Z1（staging_validate.py 存在 + dry-run flag） | US1 | T001 / T007 | ✓ |
| FR-Z2（脚本 5 step 顺序） | US1 | T002-T006 | ✓ |
| FR-Z3（stdout 格式） | US1 | T001 / T007 | ✓ |
| FR-Z4（rollback-trilogy.md 含 3 spec） | US2 | T009 / T011-T013 | ✓ |
| FR-Z5（每段 git/DB/验证 step） | US2 | T010-T013 | ✓ |
| FR-Z6（known data loss） | US2 | T010-T013 | ✓ |
| FR-Z7（cache_creation_input_tokens 提取） | US3 | T017 | ✓ |
| FR-Z8（OTel span attr 3 字段 + 0/0/0 边界） | US3 | T017 / T019 | ✓ |
| FR-Z9（4 agent + verdict 全覆盖） | US3 | T017 | ✓ |
| FR-Z10（classify_case async） | US4 | T022 | ✓ |
| FR-Z11（调用方 await） | US4 | T023 / T024 / T026 | ✓ |
| FR-Z12（pytest-asyncio fixture） | US4 | T025 | ✓ |
| FR-Z13（triggers_keywords badges 5+more） | US5 | T027 | ✓ |
| FR-Z14（muted 色，regime_tags 下方） | US5 | T027 | ✓ |
| FR-Z15（空 list 不渲染） | US5 | T027 / T028 | ✓ |
| FR-Z16（skill_metadata_inference inference_failed） | US5 | T029 | ✓ |
| FR-Z17（propose_new_skill 写 frontmatter） | US5 | T030 | ✓ |
| FR-Z18（2 Prometheus gauge） | US3 | T015 / T016 / T018 | ✓ |
| FR-Z19（仅 dashboard panel 不告警） | US3 | T021 | ✓ |
| FR-Z20（无 schema 变更） | N/A | （本 spec NOOP） | ✓ |

### Success Criteria

| SC | Tasks | Verification | Status |
|---|---|---|---|
| SC-Z1（staging_validate exit 0） | T036 | shell exec | ✓ |
| SC-Z2（rollback runbook 3 spec ≥ 3 step） | T009-T013 | grep `## Spec` | ✓ |
| SC-Z3（OTel cache attr ≥ 4 agent） | T032 | pytest e2e | ✓ |
| SC-Z4（grep "llm.invoke" 空） | T037 | shell grep | ✓ |
| SC-Z5（grep "triggers_keywords" hit + Vitest） | T028 / T038 | grep + vitest | ✓ |
| SC-Z6（test_llm_failure_writes_flag PASS） | T031 | pytest | ✓ |
| SC-Z7（dashboard 2 panel manual smoke） | T039 | manual | ✓ |
| SC-Z8（≥ 2339 test pass） | T033 | pytest | ✓ |
| SC-Z9（review-spec 无 P0/P1） | （已 PASS） | REVIEW-SPEC.md | ✓ |
| SC-Z10（review-plan + REVIEW-PLAN.md） | （本文档） | this | ✓ |
| SC-Z11（≤ 4 commit） | T040 | git log | ✓ |

### Edge cases coverage

| Edge case | Task | Status |
|---|---|---|
| migrate 失败 exit ≠ 0 | T002 / T003 / T007 | ✓ |
| cache_creation 缺字段写 0 | T017 / T019 | ✓ |
| IVE await 超时走 fallback | T022 | ✓（既有 fallback 路径不变） |
| triggers_keywords 空不渲染 | T027 / T028 | ✓ |
| LLM 失败 frontmatter 默认 metadata | T029 / T031 | ✓ |
| OTel SDK 未初始化兜底 | T019（is_recording guard） | ✓ |
| --dry-run 默认 True | T007 | ✓ |

全部 20 FR + 11 SC + 7 edge case 100% 覆盖。

## 3. Red Flag Scanning

- ✓ 无 vague task（无 "investigate" / "figure out" / "research"）
- ✓ 无 monster task（最大 task 单文件单职责）
- ✓ 无 missing file paths
- ✓ Phase 顺序合理（Polish 必须最后）
- ✓ 无跨 spec 依赖泄漏（spec 020a 不破坏既有 trilogy API）
- ✓ 无 hidden NFR（性能 / 内存 / 并发约束已在 plan.md Technical Context 列出）

## 4. NFR Validation

- ✓ 性能：staging_validate ≤ 60s（明确）
- ✓ 内存：sliding window deque 上限 ~120 entries（research.md Decision 2）
- ✓ 并发：CacheMetricsAggregator 用 `threading.Lock` 保护
- ✓ 可观测性：OTel + structlog + Prometheus 三层覆盖
- ✓ 可回滚：rollback-trilogy.md 显式覆盖

## 5. Recommendations

### Critical (Must Fix Before Implementation)
无

### Important (Should Fix)
无

### Optional (Nice to Have)
- [ ] T024 (`evaluate_node` 改 await) 在 T022 后跑 — 可在 task description 中显式标 "depends on T022"（目前在 Dependencies 段已标）
- [ ] T021 / T028 / T039 前端 manual smoke 可补 1 个 e2e Playwright 测试（OOS for 020a，可 backlog）

## 6. MVP & Incremental Strategy

- ✓ MVP scope：US1 + US3 + US4（3 个 P1 中最关键的）
- ✓ 4 commit 切分（C1 文档 / C2 后端 / C3 前端 / C4 gate）— 与 spec 019 一致
- ✓ 5 US 间无依赖，可分配 5 个并行 implementer

## Conclusion

plan + tasks 结构完整、覆盖率 100%、无 red flag、MVP 切分清晰。可进入 `/speckit-implement` 阶段。

**Ready for implementation**: Yes

**Next steps**：进入 implement 阶段（spex:ship 自动 fork subagent 执行 /speckit-implement）。

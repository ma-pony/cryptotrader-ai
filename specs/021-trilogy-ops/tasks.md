# Tasks: Spec 020a — Trilogy Ops

**Branch**: `021-trilogy-ops` | **Spec**: [spec.md](spec.md) | **Plan**: [plan.md](plan.md)

## Phase 1: Setup（无新依赖；本 spec 不需 setup task）

本 spec 复用既有 Python / TypeScript / OTel / Prometheus 基建，无新 dependency 安装。

## Phase 2: Foundational（无 — 5 user story 之间相互独立）

5 个 user story 全部 surgical 改动既有文件 + 新增独立文件，无共享 foundation。

---

## Phase 3: User Story 1 — Staging 验证脚本（P1）

**Goal**：1 命令跑完整 staging smoke check（migrate dry-run + cycle smoke + telemetry 校验 + retrieval 校验）。

**Independent Test**：`python scripts/staging_validate.py --dry-run` exit 0 + stdout ≥ 6 个 PASS 行。

- [ ] T001 [P] [US1] 创建 `scripts/staging_validate.py`，含 `StepResult` dataclass + `run_step()` 工具函数（按 research.md Decision 3）
- [ ] T002 [US1] 在 `scripts/staging_validate.py` 实现 step 1（migrate_017_to_018 dry-run）：subprocess 调 `python scripts/migrate_017_to_018.py --dry-run`，失败返回 stderr
- [ ] T003 [US1] 在 `scripts/staging_validate.py` 实现 step 2（migrate_018_to_019 dry-run）：同 step 1 模式
- [ ] T004 [US1] 在 `scripts/staging_validate.py` 实现 step 3（single cycle smoke）：mock `langchain_openai.ChatOpenAI.ainvoke` + 调 `cryptotrader.scheduler.run_one_cycle()` 单次触发
- [ ] T005 [US1] 在 `scripts/staging_validate.py` 实现 step 4（OTel 字段校验）：使用 `InMemorySpanExporter`，断言 4 agent span 各含 spec 017a FR-X18 8 字段 + 本 spec 3 cache 字段
- [ ] T006 [US1] 在 `scripts/staging_validate.py` 实现 step 5（retrieval 校验）：实例化 `EvolvingSkillProvider` + 调 `get_available_skills(agent_id="tech", snapshot={...})`，断言返回 ≥ 1 skill
- [ ] T007 [US1] 在 `scripts/staging_validate.py` 实现 main() + argparse + exit code 逻辑（任一 FAIL exit 1）
- [ ] T008 [P] [US1] 创建 `tests/test_staging_validate.py`：单测 `run_step` 成功路径 / 失败路径 / 输出格式

---

## Phase 4: User Story 2 — Rollback Runbook（P1）

**Goal**：trilogy 3 spec 的 rollback runbook 含可执行 step + known data loss。

**Independent Test**：`docs/rollback-trilogy.md` 存在且 grep 验证含 ≥ 4 个 `## Spec` 段（020a + 019 + 018 + 017b），每段含 git revert / DB / 验证 step + known data loss 段落。

- [ ] T009 [P] [US2] 创建 `docs/rollback-trilogy.md` 含适用范围 + 紧急联系信息（按 research.md Decision 4 模板）
- [ ] T010 [US2] 在 `docs/rollback-trilogy.md` 加 "Spec 020a 回退" 段（git revert + 验证 + known data loss = 无）
- [ ] T011 [US2] 在 `docs/rollback-trilogy.md` 加 "Spec 019 回退" 段（git revert 3fbf941 + rm .draft + pytest test_e2e_skill_evolution + known data loss）
- [ ] T012 [US2] 在 `docs/rollback-trilogy.md` 加 "Spec 018 回退" 段（git revert 458a0f2 + 14afc50 + 1c0302d + DB drop archived + pytest test_e2e_memory_evolution + known data loss）
- [ ] T013 [US2] 在 `docs/rollback-trilogy.md` 加 "Spec 017b 回退" 段（git revert 5b65a4a + 18e231e + git checkout config/agents + pytest test_e2e_prompt_externalization + known data loss）

---

## Phase 5: User Story 3 — Cache Hit Rate 可视化（P1）

**Goal**：`log_llm_usage()` 写 OTel span attr + Prometheus 2 metric exposed via `/metrics`。

**Independent Test**：1 mocked cycle 后 OTel trace 中 ≥ 4 agent LLM span 各含 3 cache 字段；`curl /metrics | grep llm_cache_hit_rate` 返回 gauge。

- [ ] T014 [P] [US3] 创建 `src/cryptotrader/observability/__init__.py`（空文件，标记包）
- [ ] T015 [P] [US3] 创建 `src/cryptotrader/observability/cache_metrics.py`：`CacheMetricsAggregator` 类（24h sliding window deque + Lock）
- [ ] T016 [P] [US3] 创建 `src/cryptotrader/observability/ive_metrics.py`：`IveMetricsAggregator` 类（1h sliding window deque + Lock）
- [ ] T017 [US3] 修改 `src/cryptotrader/agents/base.py:log_llm_usage()`：加 `cache_creation_input_tokens` 提取 + 写 OTel span attr 3 字段（按 research.md Decision 1）+ 调 `CacheMetricsAggregator.record(hit_rate)`
- [ ] T018 [US3] 修改 `src/api/routes/metrics.py`：注册 `LLM_CACHE_HIT_RATE_GAUGE` + `IVE_CLASSIFY_FAILURE_RATE_GAUGE` 到 prometheus REGISTRY；`prometheus_metrics()` endpoint 触发前先 `gauge.set(aggregator.average())`
- [ ] T019 [P] [US3] 创建 `tests/test_llm_usage_cache_attr.py`：使用 `InMemorySpanExporter` 验证 3 cache 字段写入；含 read=0+creation=0 边界测试 + OTel SDK 未初始化兜底测试
- [ ] T020 [P] [US3] 创建 `tests/test_metrics_endpoint_cache.py`：mock cache aggregator + 调 `/metrics` 验证 gauge 输出
- [ ] T021 [P] [US3] 修改 `web/src/pages/metrics/index.tsx`：加 2 个 panel（cache hit rate / IVE failure rate）从 prometheus output 解析

---

## Phase 6: User Story 4 — IVE Async 化（P1）

**Goal**：`classify_case` 改 async + 所有调用方改 await。

**Independent Test**：`grep -n "llm.invoke" src/cryptotrader/learning/evolution/ive.py` 返回空；`pytest tests/test_ive_async.py` PASS。

- [ ] T022 [US4] 修改 `src/cryptotrader/learning/evolution/ive.py:classify_case()`：`def` → `async def`，`llm.invoke(messages)` → `await llm.ainvoke(messages)`，加 `IveMetricsAggregator.record(success=...)` 调用（在 try / except 路径）
- [ ] T023 [US4] grep 全 repo 找所有 `classify_case` 调用方（预计 `nodes/evaluate.py` + tests）
- [ ] T024 [US4] 修改 `src/cryptotrader/nodes/evaluate.py:evaluate_node()`：classify_case 调用改 await
- [ ] T025 [US4] 修改 `tests/test_ive.py` → `tests/test_ive_async.py`：所有用例改 `@pytest.mark.asyncio` + await
- [ ] T026 [US4] 跑 `grep -rn "classify_case" src/ tests/` 校验全部调用方已改 await

---

## Phase 7: User Story 5 — SkillsGrid Triggers + Failure Flag（P2）

**Goal**：SkillsGrid 加 triggers_keywords badges + propose_new_skill `.draft` frontmatter 写 `inference_failed`。

**Independent Test**：`grep "triggers_keywords" web/src/pages/memory/components/SkillsGrid.tsx` ≥ 1 hit；`pytest tests/test_skill_proposal_metadata_inference.py::test_llm_failure_writes_flag` PASS。

- [ ] T027 [P] [US5] 修改 `web/src/pages/memory/components/SkillsGrid.tsx`：加 `triggers_keywords` badge row（最多 5 + "+N more"，muted 色，空 list 不渲染）
- [ ] T028 [P] [US5] 创建 `web/src/pages/memory/components/SkillsGrid.test.tsx`：Vitest 验证 8 keywords 显示 5 + "+3 more" / 空 list 不渲染 / regime + triggers 双 row
- [ ] T029 [US5] 修改 `src/cryptotrader/learning/evolution/skill_metadata_inference.py`：在 LLM call except 路径写 `inference_failed: True` 到返回 metadata；正常路径 `inference_failed: False`
- [ ] T030 [US5] 修改 `src/cryptotrader/learning/skill_proposal.py:propose_new_skill()`：把 metadata 中的 `inference_failed` 字段透传到 `.draft` frontmatter
- [ ] T031 [US5] 修改 `tests/test_skill_proposal_metadata_inference.py`：加 `test_llm_failure_writes_flag` 用例（mock LLM 抛 OpenAI 异常 → 验证 .draft 含 `inference_failed: true`）

---

## Phase 8: Polish & Cross-Cutting

- [ ] T032 [P] 创建 `tests/test_e2e_trilogy_ops.py`：mocked 单 cycle 跑完后断言 OTel trace 含 ≥ 4 agent LLM span 各 3 cache 字段，retrieval ≥ 1 hit
- [ ] T033 跑 `uv run python -m pytest tests/ --no-cov 2>&1 | tail -3` 验证 ≥ 2339 passed / 0 failed（SC-Z8）
- [ ] T034 跑 `uv run ruff check src/ scripts/ tests/` 修复任何新增 lint warning（如需 per-file-ignores 加到 pyproject.toml）
- [ ] T035 跑 `cd web && pnpm lint` 修复前端 lint warning
- [ ] T036 跑 `python scripts/staging_validate.py --dry-run` 自检全 PASS（SC-Z1）
- [ ] T037 跑 SC-Z4 grep 校验：`grep -n "llm.invoke" src/cryptotrader/learning/evolution/ive.py` 返回空
- [ ] T038 跑 SC-Z5 grep 校验：`grep "triggers_keywords" web/src/pages/memory/components/SkillsGrid.tsx` ≥ 1 hit
- [ ] T039 [P] 跑前端 manual smoke：启动 `cd web && pnpm dev` → 访问 `/memory` 看 SkillsGrid triggers badges → 访问 `/metrics` 看 2 个新 panel（SC-Z7）
- [ ] T040 跑 `git log --oneline 021-trilogy-ops..main | wc -l` ≤ 4 commit 校验（SC-Z11）

---

## Dependencies

```
US1 (Staging script)        ──────┐
US2 (Rollback runbook)      ──────┤
                                  ├──> Phase 8 Polish
US3 (Cache observability)   ──────┤    (T033-T040 跑全 SC 校验)
US4 (IVE async)             ──────┤
US5 (SkillsGrid + flag)     ──────┘
```

5 个 US 之间完全独立（不同文件 / 不同模块），可任意顺序执行 / 并行。

Phase 8（Polish）必须在 5 个 US 全部完成后才能跑。

T022 (`classify_case` async) 是 US4 内的关键变更；T024 (`evaluate.py`) 必须在 T022 后跑（同 US 内顺序）。

T017 (`log_llm_usage` 改造) 必须在 T015 (`CacheMetricsAggregator`) 后跑（依赖新模块）。

T018 (`metrics.py` Gauge) 必须在 T015 + T016 后跑（依赖 aggregator）。

---

## Parallel Execution

### 内部并行（同 US 不同文件）

US1：T001 / T008 可并行（不同文件）
US3：T014 / T015 / T016 / T019 / T020 / T021 可并行（不同文件，T017/T018 串行）
US4：T022 → T024 顺序 + T025 可并行（独立 test 文件）
US5：T027 / T028 / T029 可并行（前后端不同文件）；T030 在 T029 后

### 跨 US 并行

5 个 US 之间无依赖，可分配 5 个并行 implementer：
- worker A：US1（staging_validate）
- worker B：US2（rollback runbook，纯文档）
- worker C：US3（cache 观测）— 工作量最大
- worker D：US4（IVE async）— 改动最少
- worker E：US5（SkillsGrid + failure flag）

---

## Implementation Strategy

### MVP scope

**最小可发布版本**：US1 + US3 + US4（3 个 P1）= cache 观测 + IVE 解阻塞 + 部署脚本，覆盖核心运维收益。

US2 / US5 可拆出独立 PR（但建议同 PR 减少 review 开销）。

### 4-commit 切分（与 spec 019 同 pattern）

| commit | 涵盖 task | 说明 |
|---|---|---|
| C1 | T001-T013 | 文档 + 脚本（US1 + US2，纯新增） |
| C2 | T014-T026 | 后端 algorithm + telemetry（US3 + US4） |
| C3 | T027-T031 | 前端 + dashboard（US5 + US3 部分前端） |
| C4 | T032-T040 | E2E + 最终 gate |

### 增量交付

落地顺序建议：
1. C1 优先（纯新增无回归风险）
2. C2 次（含调用图修改，需 e2e 回归）
3. C3 再次（独立前端）
4. C4 最后（gate + 全套回归）

---

## Validation

任务总数：40
- US1：8 task
- US2：5 task
- US3：8 task
- US4：5 task
- US5：5 task
- Polish：9 task

每个 user story 含 independent test 标准；每个 task 含具体文件路径；checklist format 全部合规。

# Tasks：Memory Evolution（spec 018）

**输入**：[plan.md](plan.md) / [spec.md](spec.md) / [data-model.md](data-model.md) / [contracts/](contracts/) / [research.md](research.md)
**Tests**：spec 显式要求测试（SC-Z3..Z16）；所有阶段含对应测试
**Commit 序列**：4 commit 单 PR（C1 / C2 / C3 / C4）

## 格式：`[ID] [P?] [Story] Description`

- **[P]**：可与同 phase 其他任务并行
- **[Story]**：US1..US5 对应 spec user stories
- 路径为 repo root 相对路径

---

## Phase 1: Setup（无）

`agent_memory/` 既有 spec 014 目录结构无需新建。

---

## Phase 2: Foundational — C1 commit（迁移工具 + schema 字段）

**目的**：扩展 spec 014 既有 dataclass + 提供数据迁移脚本。无 behavior 变化。

**Checkpoint**：C1 后所有现有测试 PASS（schema 新字段都有 default 兼容旧实例）；新增迁移单测 PASS。

- [x] T001 [US4] 修改 `src/cryptotrader/agents/skills/schema.py:Maturity` Literal 加 `archived` 终态：`Maturity = Literal["observed", "probationary", "active", "deprecated", "archived"]`
- [x] T002 [US4] 修改 `src/cryptotrader/agents/skills/schema.py:PatternRecord` dataclass 加 5 字段：`importance: float = 0.5` / `access_count: int = 0` / `last_accessed_at: datetime = factory(now)` / `last_modified_at: datetime = factory(now)` / `fundamental_failure_streak: int = 0`
- [x] T003 [US4] 修改 `src/cryptotrader/agents/skills/schema.py:CaseRecord` dataclass 加 3 字段：`trade_execution: dict | None = None` / `causal_chain: dict | None = None` / `ive_classification: dict | None = None`
- [x] T004 [US4] 创建 `scripts/migrate_017_to_018.py`：(a) 扫 `agent_memory/cases/*.md` 加 3 段（Trade Execution / Causal Chain / IVE Classification 默认空字段）；(b) 扫 `agent_memory/<agent>/patterns/*.md` 加 5 字段（默认值见 research.md Decision 4，使用 file mtime 做 last_accessed_at / last_modified_at fallback）
- [x] T005 [US4] 迁移脚本支持 `--dry-run` 模式（输出 diff 预览不修改文件）
- [x] T006 [US4] 迁移脚本支持幂等性（重复跑不损坏）；启动时 print 备份建议
- [x] T007 [P] [US4] 创建 `tests/fixtures/memory_old_format/`：1 个旧 case（无新段）+ 2 个旧 patterns（不同 maturity 状态）
- [x] T008 [US4] 创建 `tests/test_migrate_017_to_018.py` ≥ 8 用例：(a) 旧 case 加 3 个新段；(b) 旧 patterns 加 5 字段；(c) 幂等性（重跑 2 次结果一致）；(d) `--dry-run` 不修改文件；(e) 损坏 frontmatter 跳过 + warning log；(f) 备份提示输出；(g) 已迁移 case 跳过；(h) 已迁移 pattern 跳过
- [x] T009 [US4] 运行 `uv run python -m pytest tests/test_migrate_017_to_018.py -v --no-cov` 确认全 PASS
- [x] T010 [US4] 运行完整回归 `uv run python -m pytest tests/ --no-cov 2>&1 | tail -5` 确认无回归（schema 新字段都有 default，spec 014/15/17a/17b 测试不受影响）
- [x] T011 **Commit C1**：`git add scripts/ src/cryptotrader/agents/skills/schema.py tests/test_migrate_017_to_018.py tests/fixtures/memory_old_format/` + commit message `feat(spec-018/c1): migration tool + schema field extensions (no behavior change)`

**Checkpoint C1**：迁移工具 + schema 字段就绪；main 路径无 behavior 变化。

---

## Phase 3: 算法层 — C2 commit（FSM + Pareto + IVE）

**目的**：实现 EvolvingMemoryProvider 内部用到的 3 个独立算法模块。无 cycle 集成，可独立单测。

**Goal**：US-Z2 / US-Z3 算法部分准备就绪。

**Checkpoint**：C2 后 `tests/test_fsm.py` / `test_pareto.py` / `test_ive.py` 全 PASS。

- [x] T012 [US2] 创建 `src/cryptotrader/learning/evolution/__init__.py`（空模块）
- [x] T013 [US2] 创建 `src/cryptotrader/learning/evolution/fsm.py`：
  - `Transition` dataclass（rule_id / agent_id / old_state / new_state / triggered_by / timestamp）
  - `evaluate_transitions(rule: PatternRecord) -> PatternRecord | None` 实现状态转换：
    - `observed → probationary`：`pnl_track.successes >= 3`
    - `probationary → active`：`(now - last_modified_at) >= 5 cycle 或 3 day` 且 frontmatter 全填 + body ≤ 300 行
    - `active → archived`：`fundamental_failure_streak >= 3`
    - `active → probationary`：rule 在 active 被 reflect 修改（用 manually_edited 或 last_modified_at 更新检测）
    - `deprecated` / `archived`：终态，return None
- [x] T014 [P] [US2] 创建 `tests/test_fsm.py` ≥ 12 用例：(a) observed + successes<3 → 不变；(b) observed + successes=3 → probationary；(c) probationary + 5 cycle 无修改 → active；(d) probationary + 3 days 无修改 → active；(e) active + frontmatter 不全 → 不变；(f) active + body=300 行 → 不变（已经是 active 不再升）；(g) active + fundamental_streak<3 → 不变；(h) active + fundamental_streak=3 → archived；(i) active + manually_edited=true → probationary（撤销）；(j) deprecated → 不变；(k) archived → 不变；(l) probationary + body>300 行 → 不升
- [x] T015 [US2] 创建 `src/cryptotrader/learning/evolution/pareto.py`：`rank_rules(rules: list[PatternRecord]) -> list[PatternRecord]` 实现双目标 Pareto frontier：
  - `win_rate = pnl_track.successes / max(1, pnl_track.successes + pnl_track.losses)`（0 trade → 0.5）
  - `confidence_proxy = importance × maturity_weight`（active=1.0 / probationary=0.6 / observed=0.3 / deprecated=0.0 / archived=0.0）
  - 输出：先按 Pareto 非支配前沿分层，再在层内按 `win_rate * confidence_proxy` 倒序
- [x] T016 [P] [US2] 创建 `tests/test_pareto.py` ≥ 6 用例：(a) 单 rule 返回 unchanged；(b) 2 rule 一支配一被支配 → 支配在前；(c) 2 rule Pareto 互不支配 → 同层按乘积排序；(d) win_rate=0 + active 排在 win_rate=1.0 + observed 之后（confidence weight）；(e) 0 trade rule 默认 win_rate=0.5；(f) 5 rule 混合层级正确
- [x] T017 [US3] 创建 `src/cryptotrader/learning/evolution/ive.py`：
  - `FailureClassification` dataclass（case_id / failure_type / reasoning / confidence / diagnostic_answers）
  - `classify_case(case: CaseRecord, llm_callable: Callable | None = None) -> FailureClassification` 实现 5 诊断问题 LLM 调用：
    - prompt 模板见 research.md Decision 3
    - 输出 JSON 解析（同 spec 014 既有 json_retry 兜底逻辑）
    - LLM 失败时 return `FailureClassification(failure_type="noise", confidence=0.0, ...)`
- [x] T018 [US3] 创建 `tests/test_ive.py` ≥ 8 用例：(a) mock LLM 返回 implementation；(b) mock LLM 返回 fundamental；(c) mock LLM 返回 noise；(d) LLM 调用失败 → 返回 noise + warning log；(e) LLM 输出非合法 JSON → 重试 1 次后返回 noise；(f) 5 诊断问题 prompt 含正确 case context（含 trade_execution 字段）；(g) 同 regime 下 case 在 prompt 中作 context；(h) Empty trade_execution 时 prompt 仍 well-formed
- [x] T019 [US2/US3] 运行 `uv run python -m pytest tests/test_fsm.py tests/test_pareto.py tests/test_ive.py -v --no-cov` 全 PASS
- [x] T020 [US2/US3] 运行完整回归 `uv run python -m pytest tests/ --no-cov 2>&1 | tail -5` 确认无回归
- [x] T021 **Commit C2**：`git add src/cryptotrader/learning/evolution/ tests/test_fsm.py tests/test_pareto.py tests/test_ive.py` + commit message `feat(spec-018/c2): FSM + Pareto + IVE algorithm modules`

**Checkpoint C2**：算法层 3 模块独立可用，单测全 PASS。

---

## Phase 4: Provider + nodes 集成 — C3 commit（atomic 切换）

**目的**：把 spec 017a 的 DefaultMemoryProvider 替换为 EvolvingMemoryProvider；新增 evaluate_node 节点；改 journal/execution/agents 写新 schema。

**Goal**：US-Z1 / US-Z2 / US-Z3 / US-Z4 全链路接通。

**⚠️ Atomic**：T022-T039 必须**全部完成**才能跑测试 / 提交 C3。中间状态会让 4 agent 拿不到 memory。

- [x] T022 [US1] 创建 `src/cryptotrader/learning/evolution/provider.py`：
  - `EvolvingMemoryProvider` class（实现 spec 017a `MemoryProvider` Protocol）
  - 构造：`__init__(memory_root: Path = Path("agent_memory"), top_k_rules: int = 5, top_n_cases: int = 5)`
  - `get_recent_memory()` 6 步流程见 contracts/evolving-memory-provider.md FR-Z8
  - `evaluate_all_rules() -> list[Transition]` 见 FR-Z13
  - `classify_pending_cases() -> list[FailureClassification]` 见 FR-Z16
  - 全局 try/except 容错（FR-Z9）：异常时返回空 + warning log
- [x] T023 [US3] 在 `provider.py` 实现归档逻辑：rule.fundamental_failure_streak >= 3 时 → 调 fsm.evaluate_transitions 升 archived → 移文件到 `<agent>/patterns/.archived/<rule_name>.md` + 更新 frontmatter maturity
- [x] T024 [US1] 修改 `src/cryptotrader/agents/prompt_builder.py`：删除 `class DefaultMemoryProvider`（spec 017a 路径错代码）；保留 `MemoryProvider` Protocol + `DefaultSkillProvider`（仍由 spec 019 处理）
- [x] T025 [US1] 修改 `src/cryptotrader/nodes/agents.py:_get_or_build_pb`：把 `DefaultMemoryProvider` import 改为 `EvolvingMemoryProvider`，初始化处替换
- [x] T026 [US1] 创建 `tests/test_evolving_memory_provider.py` ≥ 10 用例（见 spec.md SC-Z4 + contract 单测要求）
- [x] T027 [US1] 运行 `uv run python -m pytest tests/test_evolving_memory_provider.py -v --no-cov` 确认 PASS
- [x] T028 [US3] 创建 `src/cryptotrader/nodes/evolution.py`：
  - `async def evaluate_node(state: ArenaState) -> dict` 节点
  - 内部从 nodes/agents.py 取 module-level `_memory_provider`（现在是 EvolvingMemoryProvider）
  - 调 `provider.evaluate_all_rules()` 拿 transitions
  - 调 `provider.classify_pending_cases()` 拿 classifications
  - 写 OpenTelemetry 6 attribute（FR-Z30）
  - 异常 catch + log warning + return `{}`（不修改 state）
- [x] T029 [US3] 创建 `tests/test_evolution_node.py` ≥ 4 用例：(a) evaluate_node 在 fixture state 下跑通；(b) 调 provider.evaluate_all_rules + classify_pending_cases；(c) 异常时返回空 dict + warning；(d) telemetry 6 字段写入
- [x] T030 [US3] 修改 `src/cryptotrader/graph.py:_build_full_graph`：在 `risk_gate` 节点之后、`journal_trade` / `journal_rejection` 之前插入 `evaluate` 节点。具体策略：在 risk_router 之后加 evaluate 共享给两条 journal 分支
- [x] T031 [US3] 修改 `src/cryptotrader/nodes/journal.py:write_case`：写入新 schema（含 trade_execution / causal_chain / ive_classification 默认空字段；保持向后兼容旧 case 读取）
- [x] T032 [US3] 修改 `src/cryptotrader/nodes/execution.py`：trade 完成后回写 case 的 `trade_execution` 字段（entry / sl / tp / fill_status / hit_sl / hit_tp / exit_reason）
- [x] T033 [US3] 修改 `src/cryptotrader/nodes/journal.py:write_case`：填充 `causal_chain` 字段（per-agent tool_calls 摘要 + verbal_reinforcement_input + debate_intermediate）— 从 state 中提取
- [x] T034 [US1/US3] 运行 `uv run python -m pytest tests/test_evolving_memory_provider.py tests/test_evolution_node.py -v --no-cov` 全 PASS
- [x] T035 [US1] 运行完整回归 `uv run python -m pytest tests/ --no-cov -x --ignore=tests/test_e2e_memory_evolution.py 2>&1 | tail -10` 确认无回归
- [x] T036 [US1] 运行 `grep -rn "class DefaultMemoryProvider" src/cryptotrader/` 断言返回空（DefaultMemoryProvider 退役）
- [x] T037 [US1] 验证 `nodes/agents.py:_get_or_build_pb` 返回的 PromptBuilder 含 `EvolvingMemoryProvider`（python -c 跑或单测断言）
- [x] T038 [US1] 在 staging / dev 环境跑 `python scripts/migrate_017_to_018.py --dry-run` 验证迁移路径无错（不需要实际修改）
- [ ] T039 **Commit C3**：atomic 提交 — `git add src/cryptotrader/learning/evolution/provider.py src/cryptotrader/agents/prompt_builder.py src/cryptotrader/nodes/agents.py src/cryptotrader/nodes/evolution.py src/cryptotrader/nodes/journal.py src/cryptotrader/nodes/execution.py src/cryptotrader/graph.py tests/test_evolving_memory_provider.py tests/test_evolution_node.py` + commit message `feat(spec-018/c3): EvolvingMemoryProvider integration + evaluate node + cases schema migration (atomic)`

**Checkpoint C3**：US-Z1 / US-Z2 / US-Z3 / US-Z4 全部满足；4 agent 真正走 EvolvingMemoryProvider 路径；DefaultMemoryProvider 退役。

---

## Phase 5: 前端 `/memory` + API + E2E — C4 commit

**目的**：US-Z5 完整可视化；E2E 验收 SC-Z15。

- [ ] T040 [US5] 创建 `src/api/routes/memory.py`：4 endpoints 见 contracts/memory-api-routes.md
  - `GET /api/memory/rules` — query 参数 agent / status；返回 patterns summary list
  - `GET /api/memory/cases` — query from / to / agent；返回 cases summary list 含 IVE
  - `GET /api/memory/transitions` — query since；返回 transitions list
  - `GET /api/memory/archived` — 无 query；返回 archived patterns list
- [ ] T041 [US5] 修改 `src/api/main.py`：`app.include_router(memory.router, prefix="/api/memory", tags=["memory"])`
- [ ] T042 [US5] 创建 `tests/test_api_memory.py` ≥ 6 用例：(a) `GET /api/memory/rules?agent=tech` 200；(b) `GET /api/memory/cases?agent=macro` 200；(c) `GET /api/memory/transitions?since=...` 200；(d) `GET /api/memory/archived` 200；(e) 错误参数 400（如 status=invalid）；(f) agent 不存在 404
- [ ] T043 [US5] 创建 `web/src/pages/memory/queries.ts`：4 个 React Query hooks（`useMemoryRules` / `useMemoryCases` / `useRecentTransitions` / `useArchivedRules`）含 stale_time 配置
- [ ] T044 [P] [US5] 创建 `web/src/pages/memory/components/RulesGrid.tsx`：4 agent × 5 状态 grid，每格显示 rule 数量 + 点击展开 list
- [ ] T045 [P] [US5] 创建 `web/src/pages/memory/components/CasesTimeline.tsx`：最近 24h IVE classification 时间线（按 timestamp 倒序，显示 case_id + failure_type + 简短 reasoning）
- [ ] T046 [P] [US5] 创建 `web/src/pages/memory/components/ArchivedRules.tsx`：archived rules 列表（rule_name / agent / archived_at / fundamental_streak / final_pnl_track 摘要）
- [ ] T047 [P] [US5] 创建 `web/src/pages/memory/components/RecentTransitions.tsx`：fsm_transition 事件流（rule_id / old_state → new_state / triggered_by / timestamp）
- [ ] T048 [US5] 创建 `web/src/pages/memory/MemoryPage.tsx`：组合 4 sections，复用 spec 014 既有页面 layout 模式（参考 `MetricsPage` 等）
- [ ] T049 [US5] 修改 `web/src/components/layout/sidebar.tsx`：在 `/risk` 项之后、`/metrics` 项之前加 `{ to: '/memory', labelKey: 'nav.memory', icon: Brain }`（用 lucide-react Brain icon）
- [ ] T050 [US5] 修改 `web/src/App.tsx`：加 lazy-load 路由 `const MemoryPage = lazy(() => import('@/pages/memory/MemoryPage'))` + Route 注册
- [ ] T051 [US5] 修改 `web/src/i18n/zh-CN.ts` + `web/src/i18n/en-US.ts`：加 `nav.memory` 文案 + 4 sections title 文案
- [ ] T052 [US5] 创建 `tests/web/test_memory_page.tsx`（vitest）≥ 4 用例：(a) Rules Grid 渲染 4 agent × 5 状态；(b) Cases Timeline 倒序渲染；(c) Archived Rules 显示 fundamental_streak；(d) Recent Transitions 显示 fsm_transition 事件
- [ ] T053 [US5] 创建 `tests/test_e2e_memory_evolution.py`：mocked cycle 全链路 PASS（4 agent → debate → verdict → risk_gate → evaluate → journal）；断言 evaluate 节点写 1 fsm_transition + 5 ive_classification + 6 telemetry 字段；断言 verdict 字段完整；断言 `/api/memory/rules` 返回更新后状态
- [ ] T054 [US5] 运行 `uv run python -m pytest tests/test_api_memory.py -v --no-cov` 全 PASS
- [ ] T055 [US5] 运行 `cd web && pnpm vitest run pages/memory --reporter=verbose` 全 PASS
- [ ] T056 [US5] 运行 `uv run python -m pytest tests/test_e2e_memory_evolution.py -v --no-cov` 全 PASS
- [ ] T057 [US5] 运行完整回归 `uv run python -m pytest tests/ --no-cov 2>&1 | tail -5` 确认 ≥ 2200 pass
- [ ] T058 [P] 运行 `ruff check src/cryptotrader/learning/evolution/ src/api/routes/memory.py tests/`；如有新错误加 per-file-ignores 到 pyproject.toml
- [ ] T059 [P] 运行 `ruff format src/cryptotrader/learning/evolution/ src/api/routes/memory.py tests/`
- [ ] T060 **Commit C4**：`git add src/api/routes/memory.py src/api/main.py web/src/pages/memory/ web/src/components/layout/sidebar.tsx web/src/App.tsx web/src/i18n/ tests/test_api_memory.py tests/web/test_memory_page.tsx tests/test_e2e_memory_evolution.py pyproject.toml` + commit message `feat(spec-018/c4): /memory frontend page + API routes + E2E test`

**Checkpoint C4**：US-Z5 验收完成；SC-Z12..Z16 全满足。

---

## Phase 6: Polish

- [ ] T061 [P] 验证 `wc -l src/cryptotrader/learning/evolution/*.py` 无单文件 > 400 行
- [ ] T062 跑 `pytest tests/ --no-cov 2>&1 | tail -3`，确认整体通过率 ≥ 2200 pass / 0 fail
- [ ] T063 检查 commit 序列：`git log --oneline 019-memory-evolution ^main` 含 4 commit（C1/C2/C3/C4），无意外 commit

---

## 依赖图

```
Phase 2 (T001-T011, C1) ──> Phase 3 (T012-T021, C2) ──> Phase 4 (T022-T039, C3 atomic) ──> Phase 5 (T040-T060, C4) ──> Phase 6 (T061-T063)
```

每个 commit 必须保证测试 PASS 后才进下一个。

## 并行执行示例

**Phase 3 内部**：T013/T015/T017（FSM/Pareto/IVE 算法实现）+ T014/T016/T018（对应单测）可 3 路并行：
```
worker1: T013 → T014 (FSM)
worker2: T015 → T016 (Pareto)
worker3: T017 → T018 (IVE)
顺序: T012 → T019 → T020 → T021
```

**Phase 5 内部**：T044/T045/T046/T047（4 个前端 component）可 4 路并行；T058/T059（lint）可并行：
```
worker1: T044 (RulesGrid)
worker2: T045 (CasesTimeline)
worker3: T046 (ArchivedRules)
worker4: T047 (RecentTransitions)
顺序: T040-T043 → workers → T048-T057 → T058/T059 (P) → T060
```

## MVP 范围

**MVP**：Phase 2 + Phase 3 + Phase 4（C1 + C2 + C3 commit）— 4 agent 真正接入 EvolvingMemoryProvider + 进化算法落地。
**完整交付**：Phase 2 + 3 + 4 + 5（C1-C4 commit）+ Phase 6 验证。

## 任务统计

| Phase | Task 数 | User stories | Commit |
|---|---|---|---|
| 2 Foundational (C1) | 11 | US4 | C1 |
| 3 算法层 (C2) | 10 | US2 / US3 | C2 |
| 4 Provider 集成 (C3) | 18 | US1 / US2 / US3 / US4 | C3 |
| 5 Frontend + E2E (C4) | 21 | US5 | C4 |
| 6 Polish | 3 | — | — |
| **总计** | **63** | — | 4 commit |

## Implementation Strategy

1. **C1 优先**：T001-T011 完成后 main 路径无 behavior 变化，安全 baseline
2. **C2 算法独立**：T012-T021 与 cycle 无关，可独立单测；C2 commit 后 main 仍 spec 017b 状态
3. **C3 atomic**：T022-T039 必须**全部完成**才提交，中间状态会让 cycle 无法跑通
4. **C4 前端 + E2E**：T040-T060 是 user-facing 收尾；vitest 在 web/ 目录跑
5. **回滚策略**：C4 失败 → revert C4，main 回 C3 状态（功能完整但无前端）；C3 失败 → revert C3，main 回 C2 状态（spec 017b 状态 + 算法模块未集成）

## BLACKLIST（implement subagent 不要触碰）

- CLAUDE.md
- spec 014 既有 `learning/memory.py` IO 函数（仅 read，不改实现）
- spec 014 既有 `learning/regime.py:tag_regime`
- spec 014 既有 `learning/curation.py` / `learning/skill_proposal.py`
- spec 014 既有 `agents/skills/{_constants,_frontmatter,_io,loader,tool,_compat}.py`
- 4 agent 类（spec 017b 已稳定）
- spec 010 既有 `tracing.py` / `otel.py`
- spec 015 既有 `security.py`
- 任何 `journal/` / `portfolio/` / `risk/checks/` / `execution/` 模块（除 evaluate_node 接入外）

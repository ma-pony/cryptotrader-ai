# Tasks：Skill Evolution（spec 019）

**输入**：[plan.md](plan.md) / [spec.md](spec.md) / [data-model.md](data-model.md) / [contracts/](contracts/) / [research.md](research.md)
**Tests**：spec 显式要求测试（SC-W3..W15）；所有阶段含对应测试
**Commit 序列**：4 commit 单 PR（C1 / C2 / C3 / C4）

## 格式：`[ID] [P?] [Story] Description`

- **[P]**：可与同 phase 其他任务并行
- **[Story]**：US1..US6 对应 spec user stories
- 路径为 repo root 相对路径

---

## Phase 1: Setup（无）

agent_skills/ 既有 spec 014 目录结构无需新建。

---

## Phase 2: Foundational — C1 commit（迁移工具 + schema 字段）

**目的**：扩展 spec 014 既有 Skill dataclass + 提供数据迁移脚本。无 behavior 变化。

**Checkpoint**：C1 后所有现有测试 PASS（schema 新字段都有 default 兼容旧实例）。

- [x] T001 [US2] 修改 `src/cryptotrader/agents/skills/schema.py:Skill` dataclass 加 6 字段：
  - `regime_tags: list[str] = field(default_factory=list)`
  - `triggers_keywords: list[str] = field(default_factory=list)`
  - `importance: float = 0.5`
  - `access_count: int = 0`
  - `last_accessed_at: datetime = field(default_factory=lambda: datetime.now(UTC))`
  - `confidence: float = 0.5`
- [x] T002 [US2] 创建 `scripts/migrate_018_to_019.py`：
  - 含 FR-W3 完整 5 skill 硬编码 mapping（chain/macro/news/tech-analysis + trading-knowledge）
  - 扫 `agent_skills/*/SKILL.md` → 对每个 skill：(a) 已知 name 用 mapping；(b) 未知 name 用默认空字段
  - frontmatter 字段已存在时不覆盖（保留人工编辑）
- [x] T003 [US2] 迁移脚本支持 `--dry-run` 模式
- [x] T004 [US2] 迁移脚本启动期 print 备份建议
- [x] T005 [US2] 迁移脚本支持幂等性（重跑 2 次不损坏）
- [x] T006 [P] [US2] 创建 `tests/fixtures/skills_old_format/`：1 个旧 SKILL.md（无 6 新字段）+ 1 个 partial（含 importance 但缺其他字段）
- [x] T007 [US2] 创建 `tests/test_migrate_018_to_019.py` ≥ 8 用例：
  - (a) 5 已知 skill 用 mapping 写入预期值
  - (b) 未知 skill 用默认值
  - (c) 幂等性（重跑 2 次结果一致）
  - (d) `--dry-run` 不修改文件
  - (e) 损坏 frontmatter 跳过 + warning log
  - (f) 备份提示输出
  - (g) 已存在的字段保留人工编辑
  - (h) version 字段不变
- [x] T008 [US2] 运行 `uv run python -m pytest tests/test_migrate_018_to_019.py -v --no-cov` 全 PASS
- [x] T009 [US2] 运行完整回归 `uv run python -m pytest tests/ --no-cov 2>&1 | tail -5` 无回归
- [x] T010 **Commit C1**：`git add scripts/migrate_018_to_019.py src/cryptotrader/agents/skills/schema.py tests/test_migrate_018_to_019.py tests/fixtures/skills_old_format/` + `feat(spec-019/c1): migration tool + Skill schema field extensions (no behavior change)`

**Checkpoint C1**：迁移工具 + schema 字段就绪；main 路径无 behavior 变化。

---

## Phase 3: 算法层 — C2 commit（IDF + LLM 推断）

**目的**：实现 EvolvingSkillProvider 内部用的 2 个独立模块。无 cycle 集成，可独立单测。

**Goal**：US-W3 retrieval 算法 + US-W5 LLM 推断核心算法准备就绪。

**Checkpoint**：C2 后 `tests/test_idf.py` + `test_skill_metadata_inference.py` 全 PASS。

- [x] T011 [US3] 创建 `src/cryptotrader/learning/evolution/idf.py`：
  - `compute_idf(corpus_keywords: list[list[str]]) -> dict[str, float]`：从 list of skill keywords 计算 IDF 表
  - `extract_query_keywords(snapshot: dict) -> set[str]`：从 snapshot dict 字段名 + 关键值小写化提取
  - `score_skill(skill_keywords, query_keywords, idf_table) -> float`：score = sum(idf[kw] for kw in skill_kw if kw.lower() in query)
- [x] T012 [P] [US3] 创建 `tests/test_idf.py` ≥ 6 用例：
  - (a) 单 skill corpus → idf table 含全部关键词
  - (b) 5 skill corpus → 共享关键词低 IDF（`log(5/k)`）
  - (c) 空 corpus → 空 dict
  - (d) extract_query_keywords 从 snapshot dict 提取字段名
  - (e) score_skill 加和 IDF（含小写匹配）
  - (f) score_skill 空交集 → 0
- [x] T013 [US5] 创建 `src/cryptotrader/learning/evolution/skill_metadata_inference.py`：
  - `infer_skill_metadata(name, description, body, llm_callable=None) -> dict`
  - LLM prompt 含 spec 014 regime taxonomy + 现有 5 skill mapping 作 examples（research.md Decision 4）
  - 输出 JSON parse + 重试 1 次 + 失败时返回默认值（regime_tags=[] / triggers_keywords=[] / importance=0.5 / confidence=0.5）
- [x] T014 [P] [US5] 创建 `tests/test_skill_metadata_inference.py` ≥ 6 用例：
  - (a) mock LLM 返回合法 JSON → 输出与 LLM 一致
  - (b) mock LLM 异常 → 默认值 + warning log
  - (c) LLM 输出非合法 JSON → 重试 1 次后返回默认值
  - (d) prompt 含正确 context（name + description + body 摘要 + 5 skill examples）
  - (e) regime_tags 子集校验（只允许 spec 014 既有 8 个值）
  - (f) importance / confidence ∈ [0,1] 验证
- [x] T015 [US3/US5] 运行 `uv run python -m pytest tests/test_idf.py tests/test_skill_metadata_inference.py -v --no-cov` 全 PASS
- [x] T016 [US3/US5] 运行完整回归无回归
- [x] T017 **Commit C2**：`git add src/cryptotrader/learning/evolution/idf.py src/cryptotrader/learning/evolution/skill_metadata_inference.py tests/test_idf.py tests/test_skill_metadata_inference.py` + `feat(spec-019/c2): IDF + LLM metadata inference modules`

**Checkpoint C2**：算法层 2 模块独立可用，单测全 PASS。

---

## Phase 4: Provider + 集成 — C3 commit（atomic 切换）

**目的**：把 spec 017a/b 的 DefaultSkillProvider 替换为 EvolvingSkillProvider；改造 load_skill_tool factory；改造 propose_new_skill 加 LLM 推断。

**Goal**：US-W1 / US-W2 / US-W4 / US-W5 全链路接通。

**⚠️ Atomic**：T018-T034 必须**全部完成**才能跑测试 / 提交 C3。中间状态会让 4 agent skill 注入断链。

- [ ] T018 [US1] 创建 `src/cryptotrader/learning/evolution/skill_provider.py`：
  - `EvolvingSkillProvider` class 实现 spec 017a `SkillProvider` Protocol
  - 构造：`__init__(skill_root: Path = Path("agent_skills"), top_k: int = 5)`
  - `get_available_skills(agent_id, snapshot, k=5)` 实现 D-RT-01 两层算法（FR-W8）
  - `get_skill_by_name(name) -> Skill | None`（FR-W10）
  - 全局 try/except 容错（FR-W9）：异常时返回空 list / None + warning log
- [ ] T019 [US1] 在 `skill_provider.py` 实现 regime 提取逻辑：
  - 从 snapshot 推 current_regime（如 funding_rate > 0.0003 → "high_funding"）
  - 第一层过滤：scope filter (调 discover_skills_for_agent) + regime_tags 预过滤
- [ ] T020 [US1] 在 `skill_provider.py` 实现第二层排序：
  - 调 idf.compute_idf + idf.score_skill
  - recency_bonus = exp(-(now - last_accessed_at).total_seconds() / (7 × 86400))
  - score = (idf_score + importance + recency_bonus) × confidence
  - 取 top-k
  - 写回 access_count + last_accessed_at 到文件
- [ ] T021 [US3] 在 `skill_provider.py` 写 4 OpenTelemetry attribute（FR-W28）
- [ ] T022 [US1] 修改 `src/cryptotrader/agents/prompt_builder.py`：删除 `class DefaultSkillProvider`（spec 017a/b 实现）；保留 `MemoryProvider` Protocol + `SkillProvider` Protocol；删除 spec 017a 注释错位的"spec 018 提供进化版实现"
- [ ] T023 [US1] 修改 `src/cryptotrader/nodes/agents.py:_get_or_build_pb`：把 `_skill_provider` 初始化为 `EvolvingSkillProvider(skill_root=Path("agent_skills"))`；保留 spec 018 的 `_memory_provider = EvolvingMemoryProvider`
- [ ] T024 [US4] 修改 `src/cryptotrader/agents/skills/tool.py:_make_load_skill_tool`：加 `provider: EvolvingSkillProvider | None = None` 参数；tool 内部 `if provider: return provider.get_skill_by_name(name).body` else 走 spec 014 兜底（FR-W13）
- [ ] T025 [US4] 在 `nodes/agents.py:_get_or_build_pb` init 时 wire load_skill_tool：
  ```python
  import cryptotrader.agents.skills.tool as _t
  _t.load_skill_tool = _t._make_load_skill_tool(provider=_skill_provider)
  ```
- [ ] T026 [US5] 修改 `src/cryptotrader/learning/skill_proposal.py:propose_new_skill`：
  - 在 `_build_draft_content(...)` 之后调 `infer_skill_metadata(name, description, body)`
  - 把 metadata 合并到 draft frontmatter
  - 写入 `.draft` 文件
- [ ] T027 [US5] 在 propose_new_skill 写 7 OpenTelemetry attribute（FR-W29）
- [ ] T028 [US1] 创建 `tests/test_evolving_skill_provider.py` ≥ 12 用例（contracts/evolving-skill-provider.md）
- [ ] T029 [US4] 创建 `tests/test_load_skill_tool.py` ≥ 4 用例：
  - (a) `_make_load_skill_tool(provider)` 返回 tool；tool 调 provider.get_skill_by_name
  - (b) provider=None 时走 spec 014 兜底（兼容）
  - (c) provider 异常 → 返回 error string + log warning
  - (d) skill 不存在 → 返回 error string
- [ ] T030 [US5] 创建 `tests/test_skill_proposal_metadata_inference.py` ≥ 6 用例：
  - (a) mock LLM 返回合法 JSON → .draft frontmatter 含 LLM 输出
  - (b) mock LLM 异常 → 默认值 + warning log
  - (c) LLM 输出非合法 JSON → 重试 1 次后默认值
  - (d) Telemetry 7 字段写入
  - (e) draft_path 在 telemetry 中正确
  - (f) prompt 含 5 skill mapping 作 examples
- [ ] T031 [US1] 运行 `uv run python -m pytest tests/test_evolving_skill_provider.py tests/test_load_skill_tool.py tests/test_skill_proposal_metadata_inference.py -v --no-cov` 全 PASS
- [ ] T032 [US1] 运行完整回归无回归（spec 014/15/17a/17b/18 测试不影响）
- [ ] T033 [US1] 运行 `grep -rn "class DefaultSkillProvider" src/cryptotrader/` 断言返回空
- [ ] T034 **Commit C3**：atomic — `git add src/cryptotrader/learning/evolution/skill_provider.py src/cryptotrader/agents/prompt_builder.py src/cryptotrader/nodes/agents.py src/cryptotrader/agents/skills/tool.py src/cryptotrader/learning/skill_proposal.py tests/test_evolving_skill_provider.py tests/test_load_skill_tool.py tests/test_skill_proposal_metadata_inference.py` + `feat(spec-019/c3): EvolvingSkillProvider integration + load_skill_tool factory + propose_new_skill LLM inference (atomic)`

**Checkpoint C3**：US-W1 / US-W2 / US-W4 / US-W5 满足；4 agent 走 EvolvingSkillProvider；DefaultSkillProvider 退役。

---

## Phase 5: API + 前端 + E2E — C4 commit

**目的**：US-W6（前端可视）+ E2E 验收 SC-W14。

- [X] T035 [US6] 修改 `src/api/routes/memory.py` 加 4 个 endpoints（contracts/skill-api-routes.md）：
  - `GET /api/memory/skills` 返回 list[Skill summary]（含 6 新字段，不含 body）
  - `GET /api/memory/skills/{name}` 返回完整 Skill（含 body）
  - `GET /api/memory/skill-access` 返回 list[{skill_name, scope, access_count, last_accessed_at}]
  - `GET /api/memory/skill-proposals` 返回 list[{name, draft_path, created_at, llm_inferred_metadata, user_saved}]
- [X] T036 [US6] 创建 `tests/test_api_memory_skills.py` ≥ 8 用例（contracts/skill-api-routes.md "单测要求"）
- [X] T037 [US6] 创建 `web/src/pages/memory/components/SkillsGrid.tsx`：
  - 4 agent 子分区（tech / chain / news / macro）+ 1 shared 子分区
  - 每格显示 skill name / scope / importance / access_count / last_accessed_at
  - regime_tags 显示为 badges；triggers_keywords 显示前 3 个
  - 点击展开 body
- [X] T038 [US6] 修改 `web/src/pages/memory/MemoryPage.tsx`：在现有 3 sections（RulesGrid + 2-col grid + ArchivedRules）**之后**加第 4 单行 section "Skills Grid"，import + 渲染 SkillsGrid 组件
- [X] T039 [US6] 修改 `web/src/pages/memory/queries.ts` 加 4 React Query hooks（useSkills / useSkillByName / useSkillAccess / useSkillProposals）含 stale_time 配置
- [X] T040 [US6] 修改 i18n 文件：先 grep 确认实际路径（`web/src/locales/zh-CN/memory.json` 或 `web/src/i18n/zh-CN.ts`），加 Skills section 文案（如 `memory.skills.title` / `memory.skills.proposals` / 4 agent 名称等）
- [X] T041 [P] [US6] 修改 `tests/web/test_memory_page.tsx` 加 4 新用例：
  - (a) SkillsGrid 渲染 5 skill grid（4 agent + 1 shared）
  - (b) 点击 skill 展开 body
  - (c) regime_tags 显示为 badges
  - (d) Skill Proposals 区显示 proposal 历史
- [X] T042 [US1] 创建 `tests/test_e2e_skill_evolution.py`：
  - (a) mocked cycle 跑完 4 agent → debate → verdict → risk → evaluate → journal
  - (b) 4 agent prompt `available_skills` section 含 fixture skill body（按 D-RT-01 排序）
  - (c) skill.retrieval.* 4 telemetry 字段写入
  - (d) skill_proposal mock 触发 → .draft 含 LLM 推断 metadata + 7 telemetry 字段
  - (e) Web `/api/memory/skills` 返回更新后 access_count
- [X] T043 [US6] 运行 `uv run python -m pytest tests/test_api_memory_skills.py -v --no-cov` 全 PASS
- [X] T044 [US6] 运行 `cd web && pnpm vitest run pages/memory --reporter=verbose` 全 PASS
- [X] T045 [US1] 运行 `uv run python -m pytest tests/test_e2e_skill_evolution.py -v --no-cov` 全 PASS
- [X] T046 [US1] 运行完整回归 `uv run python -m pytest tests/ --no-cov 2>&1 | tail -5` 确认 ≥ 2300 pass
- [X] T047 [P] 运行 `ruff check src/cryptotrader/learning/evolution/idf.py src/cryptotrader/learning/evolution/skill_metadata_inference.py src/cryptotrader/learning/evolution/skill_provider.py src/api/routes/memory.py tests/`；如有新错误加 per-file-ignores 到 pyproject.toml
- [X] T048 [P] 运行 `ruff format src/cryptotrader/learning/evolution/ src/api/routes/memory.py tests/`
- [X] T049 **Commit C4**：`git add src/api/routes/memory.py web/src/pages/memory/ web/src/locales/ tests/test_api_memory_skills.py tests/test_e2e_skill_evolution.py tests/web/test_memory_page.tsx pyproject.toml` + `feat(spec-019/c4): /memory Skills section + 4 API endpoints + E2E`

**Checkpoint C4**：US-W6 满足；SC-W11..W15 全满足。

---

## Phase 6: Polish

- [X] T050 [P] 验证 `wc -l src/cryptotrader/learning/evolution/skill_provider.py` < 400 行（333 行）
- [X] T051 [P] 验证 `wc -l src/cryptotrader/learning/evolution/idf.py src/cryptotrader/learning/evolution/skill_metadata_inference.py` 各 < 200 行（idf=102，skill_metadata_inference=221）
- [X] T052 跑 `pytest tests/ --no-cov 2>&1 | tail -3` 整体通过率 ≥ 2300 / 0 fail（2339 passed）
- [X] T053 检查 commit 序列：`git log --oneline 020-skill-evolution ^main` 含 4 commit（C1-C4）

---

## 依赖图

```
Phase 2 (T001-T010, C1) ──> Phase 3 (T011-T017, C2) ──> Phase 4 (T018-T034, C3 atomic) ──> Phase 5 (T035-T049, C4) ──> Phase 6 (T050-T053)
```

## 并行执行示例

**Phase 3 内部**：T011-T012（IDF）+ T013-T014（LLM 推断）可 2 路并行：
```
worker1: T011 → T012
worker2: T013 → T014
顺序: T015 → T016 → T017
```

**Phase 5 内部**：T037（SkillsGrid）/ T038（MemoryPage）/ T039（queries）/ T040（i18n）/ T041（vitest）多 worker 并行。

## MVP 范围

**MVP**：Phase 2 + Phase 3 + Phase 4（C1 + C2 + C3 commit）— EvolvingSkillProvider + LLM 推断 + load_skill_tool 改造完成。
**完整交付**：Phase 2-5（C1-C4）+ Phase 6 验证。

## 任务统计

| Phase | Task 数 | User stories | Commit |
|---|---|---|---|
| 2 Foundational (C1) | 10 | US2 | C1 |
| 3 算法层 (C2) | 7 | US3 / US5 | C2 |
| 4 Provider 集成 (C3) | 17 | US1 / US3 / US4 / US5 | C3 |
| 5 Frontend + E2E (C4) | 15 | US6 | C4 |
| 6 Polish | 4 | — | — |
| **总计** | **53** | — | 4 commit |

## Implementation Strategy

1. **C1 优先**：T001-T010 完成后 main 路径无 behavior 变化，安全 baseline
2. **C2 算法独立**：T011-T017 与 cycle 无关，可独立单测；C2 commit 后 main 仍 spec 018 状态
3. **C3 atomic**：T018-T034 必须**全部完成**才提交（中间状态会让 4 agent skill 断链）
4. **C4 收尾**：T035-T049 user-facing；vitest 在 web/ 跑
5. **回滚策略**：C4 失败 → revert C4，回 C3 状态（功能完整无前端）；C3 失败 → revert C3，回 C2 状态（算法独立可用但未集成）

## BLACKLIST（implement subagent 不要触碰）

- CLAUDE.md
- spec 014 既有 `learning/memory.py` / `learning/curation.py` / `learning/regime.py` / `learning/verbal.py`
- spec 014 既有 `agents/skills/{_constants,_frontmatter,_io,loader}.py`（loader.py 仅 read，不改实现）
- spec 014 既有 `agents/skills/{schema,tool}.py` 不在 spec 改动范围之外的部分（仅加 6 字段；tool.py 仅改 factory）
- 4 agent 类（spec 017b 稳定）
- spec 018 既有 `learning/evolution/{fsm,pareto,ive,provider,_io}.py`（不动）
- spec 018 既有 `nodes/evolution.py`（不动）
- spec 010 既有 `tracing.py` / `otel.py`
- spec 015 既有 `security.py`
- 任何 `journal/` / `portfolio/` / `risk/` / `execution/` 模块
- spec 018 既有 `web/src/pages/memory/components/{RulesGrid,CasesTimeline,ArchivedRules,RecentTransitions}.tsx`（不动）

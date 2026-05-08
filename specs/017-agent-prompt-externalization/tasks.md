# Tasks：Agent Prompt Builder Foundation（spec 017a，scope-narrowed）

**输入**：[plan.md](plan.md) / [spec.md](spec.md) / [data-model.md](data-model.md) / [contracts/](contracts/) / [research.md](research.md)
**Tests**：本 spec 显式要求测试（SC-X2 / SC-X3 / SC-X11），所有任务含对应测试。

**Scope Split（2026-05-08）**：原 spec 017 拆分为 017a（基建，本 spec）+ 017b（集成，待立项）。本 tasks.md Phase 1-2（T001-T014）属于 017a，Phase 3-6（T015-T043）已迁移至 spec 017b。

## 格式：`[ID] [P?] [Story] Description`

- **[P]**：可与其他同 phase 任务并行（不同文件，无相互依赖）
- **[Story]**：US-X1 / US-X2 / US-X3 / US-X4（对应 spec 中 user story 编号）
- 全部任务路径为绝对/相对 repo root 的具体路径

## Path Conventions

Single project — `src/cryptotrader/agents/` 与 `tests/` 在 repo root，`config/agents/` 为新增顶层目录。

---

## Phase 1: Setup

**目的**：基础目录结构

- [x] T001 创建 `config/agents/` 目录（mkdir -p config/agents）

---

## Phase 2: Foundational（前置依赖，所有 story 均需）

**目的**：PromptBuilder / Provider 协议 / TokenBudgetEnforcer / ConfigLoader 基建。完成本 phase 后，US-X1 / US-X2 / US-X3 / US-X4 均可推进。

**⚠️ 关键**：Phase 3-6 所有 user story 任务依赖本 phase 完成。

- [x] T002 创建 `src/cryptotrader/agents/prompt_builder.py`，实现 `ConfigValidationError` exception 类
- [x] T003 在 `src/cryptotrader/agents/prompt_builder.py` 实现 `AgentConfig` dataclass + `ConfigLoader` 类（含 frontmatter 切分、body section 切分、9 项校验规则），见 `contracts/agent-config-schema.md`
- [x] T004 [P] 在 `src/cryptotrader/agents/prompt_builder.py` 实现 `MemoryProvider` Protocol + `SkillProvider` Protocol + `Skill` dataclass，见 `contracts/memory-provider.md` / `contracts/skill-provider.md`
- [x] T005 [P] 在 `src/cryptotrader/agents/prompt_builder.py` 实现 `EnforceResult` dataclass + `TokenBudgetEnforcer` 类（复用 `cryptotrader.learning.context._estimate_tokens`），见 `contracts/prompt-builder.md`
- [x] T006 在 `src/cryptotrader/agents/prompt_builder.py` 实现 `DefaultMemoryProvider`（读 `agent_memory/<agent_id>/{patterns.md, cases.jsonl}`），见 `contracts/memory-provider.md`
- [x] T007 在 `src/cryptotrader/agents/prompt_builder.py` 实现 `DefaultSkillProvider`（扫 `agent_skills/<id>/SKILL.md`，按 agent_id 在 tags 中过滤），见 `contracts/skill-provider.md`
- [x] T008 在 `src/cryptotrader/agents/prompt_builder.py` 实现 `PromptBuilder` 类（构造 + `build()` + telemetry 8 字段挂 active span / log fallback），见 `contracts/prompt-builder.md`
- [x] T009 [P] 创建 `tests/test_config_loader.py`，覆盖 9 条校验规则（合法 frontmatter / 缺必填字段 / agent_id 不匹配 / budget≤0 / sections 缺核心 / body 缺段落 / priority 引用未声明 section / slot_overrides 引用错误 / slot_overrides 交集冲突）
- [x] T010 [P] 创建 `tests/test_token_budget.py`，覆盖 SC-X3 的 5 用例：(a) 不超 budget 不丢；(b) 超 budget 按优先级丢；(c) 远超 budget 触发降级；(d) system_prompt+output_schema 强制保留；(e) 估算误差 < 10% 与 tiktoken（如未安装则用预先存入的参考值）
- [x] T011 [P] 创建 `tests/test_prompt_builder.py`，覆盖 SC-X2 的 7 用例：(a) 加载合法 config；(b) 缺字段抛 ConfigValidationError；(c) 拼接产出 SystemMessage+HumanMessage；(d) memory_provider 空返回走占位；(e) skill_provider 空返回走占位；(f) slot_overrides 生效；(g) snapshot/portfolio 字段缺失走默认占位
- [x] T012 创建 `tests/fixtures/agent_configs/example.md` 作为单测 fixture（合法的最小 frontmatter + body）
- [x] T013 在 `src/cryptotrader/agents/__init__.py` 导出 `PromptBuilder` / `MemoryProvider` / `SkillProvider` / `Skill` / `DefaultMemoryProvider` / `DefaultSkillProvider` / `ConfigValidationError`
- [x] T014 运行 `pytest tests/test_config_loader.py tests/test_token_budget.py tests/test_prompt_builder.py -v`，确认全部 PASS

**Checkpoint**：基建就绪，4 个 agent 迁移可并行启动。

---

## Phase 3-6（DEFERRED → spec 017b）

下列 Phase 3-6 全部任务（T015-T043）从本 spec 移出，由 spec 017b 单独立项实施。理由：实施过程中发现 `base.py` BaseAgent/ToolAgent / `config.py` AgentRegistry / `agents/skills/middleware` 都需要整合，整合范围比原估算大 2-3 倍，应作为独立 spec 处理。

下列任务 **不属于** 本 spec 验收范围，仅作为 017b 立项参考保留。

---

## Phase 3: User Story 1 — TechAgent 配置驱动迁移（Priority: P1）🎯 MVP（DEFERRED → 017b）

**Goal**：把 TechAgent 的 ROLE prompt 外置到 `config/agents/tech.md`，构造器改为必填 `prompt_builder`。这一步独立完成即证明：(a) 配置驱动 prompt 修改可行（US-X1）；(b) Provider 协议设计满足 spec 018 所需（US-X2）。

**Independent Test**：单元测试 mock LLM 后断言 TechAgent 接收的 SystemMessage 来自 `config/agents/tech.md` 而非硬编码 ROLE。

- [ ] T015 [US1] 阅读 `src/cryptotrader/agents/tech.py` 当前 `ROLE` 字符串与 message 拼接逻辑，提取 system_prompt / output_schema 内容
- [ ] T016 [US1] 创建 `config/agents/tech.md`，frontmatter 含 `agent_id: tech` + 5 必填字段 + priority；body 含 5 个 section（system_prompt 来自步骤 T015 提取，output_schema 来自现有 JSON schema 描述，user_tail / available_skills / recent_memory 写占位说明文字）
- [ ] T017 [US1] 重构 `src/cryptotrader/agents/tech.py`：(a) 删除 `ROLE = """..."""` 常量；(b) 删除手工拼接 SystemMessage 字符串的代码；(c) 构造器签名改 `def __init__(self, *, prompt_builder: PromptBuilder, ...)`（必填、kw-only）；(d) `_build_messages`（或等价方法）改为 `return self._prompt_builder.build(snapshot, portfolio)`
- [ ] T018 [US1] 更新 `tests/test_tech_agent.py`：(a) fixture 注入真实 PromptBuilder（指向 `config/agents/`）；(b) mock LLM ainvoke；(c) 断言 SystemMessage.content 含 `config/agents/tech.md` 中 system_prompt 段落的标志性文字；(d) 断言 HumanMessage.content 含 snapshot 字段
- [ ] T019 [US1] 运行 `pytest tests/test_tech_agent.py -v`，确认 PASS
- [ ] T020 [US1] 运行 `wc -l src/cryptotrader/agents/tech.py` 确认 < 150 行（SC-X9 partial 验证）

**Checkpoint**：US-X1 + US-X2 已可独立验收（TechAgent 走通新链路）；其他 3 agent 仍走旧路径（但本 spec 范围内必须全迁移，进入 Phase 4）。

---

## Phase 4: User Story 4 — 剩余 3 个 Agent 迁移 + ROLE 退役（Priority: P2）

**Goal**：迁移 ChainAgent / NewsAgent / MacroAgent，使 4 个 agent 的 ROLE 常量与硬编码拼接全部退役（US-X4）。同时调用方一次性切换。

**Independent Test**：迁移完成后，`grep -rn "^ROLE\s*=" src/cryptotrader/agents/` 返回空（SC-X8）；4 agent 文件每个 < 150 行（SC-X9）。

- [ ] T021 [P] [US4] 创建 `config/agents/chain.md`（参考 T016 模板，system_prompt 来自当前 `src/cryptotrader/agents/chain.py` 的 ROLE）
- [ ] T022 [P] [US4] 创建 `config/agents/news.md`（system_prompt 来自当前 `src/cryptotrader/agents/news.py` 的 ROLE）
- [ ] T023 [P] [US4] 创建 `config/agents/macro.md`（system_prompt 来自当前 `src/cryptotrader/agents/macro.py` 的 ROLE）
- [ ] T024 [P] [US4] 重构 `src/cryptotrader/agents/chain.py`（同 T017 模板：删 ROLE / 删拼接 / 构造器加 prompt_builder / 改 _build_messages）
- [ ] T025 [P] [US4] 重构 `src/cryptotrader/agents/news.py`（同 T024）
- [ ] T026 [P] [US4] 重构 `src/cryptotrader/agents/macro.py`（同 T024）
- [ ] T027 [P] [US4] 更新 `tests/test_chain_agent.py`（同 T018 模板）
- [ ] T028 [P] [US4] 更新 `tests/test_news_agent.py`（同 T018）
- [ ] T029 [P] [US4] 更新 `tests/test_macro_agent.py`（同 T018）
- [ ] T030 [US4] 更新 `src/cryptotrader/nodes/agents.py`：(a) 启动期实例化 `DefaultMemoryProvider` / `DefaultSkillProvider` / 4 个 PromptBuilder；(b) 4 处 agent 实例化注入 `prompt_builder=...`；(c) 删除任何旧的 ROLE-based 实例化代码
- [ ] T031 [US4] 检查 `src/cryptotrader/graph.py` 是否有 agent 实例化代码；若有，同步更新（注入 prompt_builder）；若无则跳过
- [ ] T032 [US4] 运行 `grep -rn "^ROLE\s*=" src/cryptotrader/agents/`，断言返回空（SC-X8）
- [ ] T033 [US4] 运行 `wc -l src/cryptotrader/agents/{tech,chain,news,macro}.py`，断言每个 < 150 行（SC-X9）
- [ ] T034 [US4] 运行 `pytest tests/test_tech_agent.py tests/test_chain_agent.py tests/test_news_agent.py tests/test_macro_agent.py -v`，确认全部 PASS

**Checkpoint**：4 agent 全部走新链路，ROLE 退役完成（FR-X16 / FR-X17 / SC-X4 / SC-X8 / SC-X9 满足）。

---

## Phase 5: User Story 3 — E2E + Telemetry 验证（Priority: P2）

**Goal**：覆盖 SC-X5（E2E 全链路）+ SC-X6（telemetry 8 字段）+ SC-X7（token 节省 < 15% 差异）。

**Independent Test**：mocked cycle 跑完 4 agents → debate gate → verdict → risk gate；查询 trace 含 8 个 `prompt.builder.*` 字段。

- [ ] T035 [US3] 创建 `tests/test_e2e_prompt_externalization.py`：(a) mock 4 agent 的 LLM ainvoke 返回 fixture analysis；(b) mock debate / verdict / risk 节点最小输入；(c) 跑完整 LangGraph cycle；(d) 断言每个 agent telemetry attribute 含 `prompt.builder.agent_id` 等 8 字段；(e) 断言 verdict 字段完整（target_price / stop_loss / take_profit / R:R）
- [ ] T036 [US3] 运行 `pytest tests/test_e2e_prompt_externalization.py -v`，确认 PASS
- [ ] T037 [US3] 在 `tests/test_e2e_prompt_externalization.py` 增加 SC-X7 token 节省指标：用相同 snapshot 输入跑新路径，记录 `prompt_size_post`；与 git pre-T2 commit 的硬编码 ROLE 路径产出长度对比；差异 > 15% 时打印 warning（不 fail，仅观测）

**Checkpoint**：US-X3 验收完成；SC-X5 / SC-X6 / SC-X7 满足。

---

## Phase 6: Polish & Cross-Cutting

**目的**：最后清理与 lint。

- [ ] T038 [P] 运行 `ruff check src/cryptotrader/agents/ tests/` 修复任何新增 lint 警告
- [ ] T039 [P] 运行 `ruff format src/cryptotrader/agents/ tests/`
- [ ] T040 运行整体回归 `pytest tests/ -x --ignore=tests/test_e2e_prompt_externalization.py 2>&1 | tail -20`，确认无回归（spec 014 / 015 既有测试仍 PASS）
- [ ] T041 运行 `pytest tests/test_e2e_prompt_externalization.py -v`，确认 PASS
- [ ] T042 最终 grep gate：`grep -rn "^ROLE\s*=" src/cryptotrader/agents/`，断言返回空（SC-X8 final）
- [ ] T043 验证 `ls config/agents/*.md` 含 4 个文件（SC-X1 final）

---

## 依赖图

```
Phase 1 (T001) ──> Phase 2 (T002-T014)
                          │
                          ▼
                   Phase 3 (T015-T020) [US1, MVP]
                          │
                          ▼
                   Phase 4 (T021-T034) [US4]
                          │
                          ▼
                   Phase 5 (T035-T037) [US3]
                          │
                          ▼
                   Phase 6 (T038-T043)
```

## 并行执行示例

### Phase 2 内部

T004 / T005 / T009 / T010 / T011 可并行（不同文件 / 不同测试套）：

```bash
# 4 个并行 worker
pytest tests/test_config_loader.py &
pytest tests/test_token_budget.py &
pytest tests/test_prompt_builder.py &
# 等 T002/T003/T006/T007/T008 完成后
```

### Phase 4 内部

T021-T029（10 个任务）可并行 — 4 agent × 3 文件（config/source/test）互不冲突：

```text
worker1: T021 → T024 → T027   (chain)
worker2: T022 → T025 → T028   (news)
worker3: T023 → T026 → T029   (macro)
顺序执行: T030 → T031 → T032 → T033 → T034
```

## MVP 范围（推荐增量交付）

**MVP**：Phase 1 + Phase 2 + Phase 3（即 T001-T020）
- 验证 PromptBuilder 基建可行
- TechAgent 一个 agent 走通新链路
- 不影响其他 3 agent（仍走旧路径，本 MVP 后立即进 Phase 4 完成全迁移）

**完整交付**：Phase 1-6 全部完成（T001-T043）
- 4 agent 全迁移
- ROLE 退役
- E2E + telemetry 验收

## 任务统计（spec 017a 范围）

| Phase | Task 数 | User Story | 可并行 | 状态 |
|---|---|---|---|---|
| 1 Setup | 1 | — | — | ✅ DONE |
| 2 Foundational | 13 | US1 + US2 + US3 | T004/T005/T009/T010/T011 | ✅ DONE |
| **本 spec 总计** | **14** | — | — | ✅ DONE（44 单测 PASS） |
| 3 US1 (TechAgent) | 6 | — | — | DEFERRED → 017b |
| 4 US4 (3 agent) | 14 | — | T021-T029 | DEFERRED → 017b |
| 5 US3 (E2E) | 3 | — | — | DEFERRED → 017b |
| 6 Polish | 6 | — | T038/T039 | DEFERRED → 017b |
| **017b 待立项** | **29** | — | — | pending |

## Implementation Strategy

**spec 017a（本 spec，已完成）**：
- T001-T014 一次性提交（基建模块 + 测试）
- 不修改 4 agent 源码，确保隔离性 SC-X13

**spec 017b（待立项，参考）**：
- 先完成基建集成：base.py / config.py / nodes/agents.py 重构
- 再迁移 4 agent（chain → news → macro → tech 顺序，便于按 commit revert）
- 最后 E2E + telemetry 全链路验收

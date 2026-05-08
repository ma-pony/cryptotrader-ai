# Feature Specification: Agent Prompt Builder Foundation

**Feature Branch**: `017-agent-prompt-externalization`
**Created**: 2026-05-08
**Status**: Scope-narrowed（split decision 2026-05-08）
**Input**: User description: "agent-prompt-externalization — 把 4 个 analysis agents 的 ROLE 系统提示词外置到 config 文件，由 PromptBuilder 拼接 + TokenBudgetEnforcer 控量 + telemetry 可观测。"

## Scope Split History

- **2026-05-08 初版**：本 spec 同时覆盖 (a) PromptBuilder 基建 (b) 4 agent 迁移 + ROLE 退役 + base.py/config.py 集成
- **2026-05-08 调整**：实施过程中发现 base.py BaseAgent/ToolAgent 的 `role_description` 路径 + config.py AgentRegistry 的 `_resolve_role` / `_resolve_skills` + agents/skills/middleware 的 `SkillsInjectionMiddleware` 都需要整合，整合范围比原 6-task 估算大 2-3 倍。决策：拆分为两个 spec
- **本 spec（017a，narrowed）**：仅交付 PromptBuilder + Provider Protocol + 单元测试基建（即原 Phase 1+2，T001-T014）
- **后续 spec（017b，待立项）**：完成 4 agent 迁移 + base.py / config.py 重构 + ROLE 退役 + E2E 测试

## Purpose

为 spec 018（skill / memory 进化）提供 PromptBuilder 基建：把"agent 配置 + memory 注入 + skill 注入 + snapshot 渲染 + token 预算 + telemetry"的拼接管线统一到一个可测试模块，定义 `MemoryProvider` / `SkillProvider` 协议接口，使 spec 018 可以无缝注入进化版 Provider 实现。

本 spec **仅交付基建模块**，不修改 4 个 analysis agent 的现有 ROLE 字符串路径，不动 `base.py` / `config.py` / `nodes/agents.py` 的现有 wiring。4 agent 真正切换到 PromptBuilder 路径留待 017b。

后续 spec（017b 与 018）依赖本 spec 落地的 `PromptBuilder` 类与两个 Provider Protocol 接口稳定。

## User Scenarios & Testing *(mandatory)*

### User Story 1 - PromptBuilder 模块可独立测试 (Priority: P1) 🎯 MVP

作为架构师，我需要 `PromptBuilder` + `ConfigLoader` + `TokenBudgetEnforcer` + `Provider Protocol` 作为可独立单元测试的模块上线，使 spec 017b（agent 集成）与 spec 018（进化算法）都基于同一基建开发。

**Why this priority**：基建未上线之前，017b 和 018 的设计决策都悬空。基建本身的契约（Provider Protocol 签名、PromptBuilder.build() 返回值结构、TokenBudgetEnforcer 的丢/降语义）必须先稳定。

**Independent Test**：`pytest tests/test_prompt_builder.py tests/test_token_budget.py tests/test_config_loader.py -v` 全部 PASS（44+ 用例）；模块可被 `from cryptotrader.agents.prompt_builder import PromptBuilder` 导入并实例化（用 fixture config + mock Provider）。

**Acceptance Scenarios**：

1. **Given** `tests/fixtures/agent_configs/example.md` 合法 config，**When** `PromptBuilder("example", fixture_dir, mock_mem, mock_skl, model="test").build({}, {})` 调用，**Then** 返回 `(SystemMessage, HumanMessage)` 且 8 个 telemetry 字段写入
2. **Given** YAML frontmatter 缺 `agent_id`，**When** PromptBuilder 实例化，**Then** 抛 `ConfigValidationError`
3. **Given** mock `MemoryProvider` 返回 `"### Patterns\n- 测试"`，**When** PromptBuilder.build() 调用，**Then** UserMessage 含该字符串
4. **Given** total tokens > budget，**When** TokenBudgetEnforcer 触发，**Then** `dropped_sections` 含被丢段；`system_prompt` 与 `output_schema` 仍保留

---

### User Story 2 - Provider 协议契约稳定 (Priority: P1)

作为后续 spec 实施者（017b / 018），我需要 `MemoryProvider` / `SkillProvider` 协议签名稳定，使后续 spec 直接接入而无需修改本 spec 代码。

**Why this priority**：协议改一次，017b / 018 都要回滚。必须在合并前敲定。

**Independent Test**：编写满足协议的最小 mock 实现 + 注入 PromptBuilder + build() 成功；协议方法签名见 `contracts/memory-provider.md` / `contracts/skill-provider.md`。

**Acceptance Scenarios**：

1. **Given** 任何类实现 `def get_recent_memory(self, agent_id, snapshot, k=5) -> str` 即可作为 MemoryProvider 注入（不要求继承基类，鸭子类型）
2. **Given** 任何类实现 `def get_available_skills(self, agent_id, snapshot, k=5) -> list[Skill]` 即可作为 SkillProvider 注入
3. **Given** `DefaultMemoryProvider(memory_root=Path)` / `DefaultSkillProvider(skills_root=Path)` 默认实现，**When** 目录为空，**Then** 返回占位 / 空列表，不抛异常

---

### User Story 3 - TokenBudgetEnforcer 行为可观测 (Priority: P2)

作为后续 spec 实施者，我需要 `TokenBudgetEnforcer` 的丢/降决策可被 telemetry / EnforceResult 完整暴露，使 017b E2E 测试能断言"哪些 section 在 budget 紧张时被丢"。

**Why this priority**：黑盒丢段会让 spec 018 进化算法难以诊断"为什么 skill 没注入到 LLM"。

**Independent Test**：构造 sections 总长 > budget 的输入 → enforce() → 验证 `EnforceResult.dropped_sections` 与 `degraded_sections` 字段非空且符合 priority 顺序。

**Acceptance Scenarios**：

1. **Given** budget=100, sections 总 token 1000, priority `{snapshot: 1, memory: 5}`，**When** enforce()，**Then** `dropped_sections` 含 `memory`（priority 数字大先丢）
2. **Given** budget=100, sections 仍超后，**When** enforce()，**Then** `degraded_sections` 含 `recent_memory` / `available_skills`（截断而非丢段）
3. **Given** `protected={"system_prompt", "output_schema"}`，**When** enforce()，**Then** 这两段永不在 `dropped_sections` / `degraded_sections` 中

---

### Edge Cases

- `agent_memory/` 为空 → recent_memory section 占位（"暂无历史记忆"），不报错
- `agent_skills/` 0 条匹配 → available_skills section 占位（"暂无可用技能"），不报错
- snapshot 字段缺失 → 走默认占位字符串
- token 总数 >> budget → 按优先级丢 recent_memory / available_skills，强制保留 system_prompt + output_schema
- config YAML 损坏 / frontmatter 字段缺失 → PromptBuilder 实例化时 fail-fast 抛 `ConfigValidationError`
- `slot_overrides` 引用不存在的 section name → 实例化时校验失败
- 同一 section 同时出现在 system / user_tail 槽位（用户配置错误）→ 实例化时校验失败
- 进程内未在 OpenTelemetry tracing 上下文（单测 / CLI 调用）→ telemetry 字段降级到 structured log，不抛异常

## Requirements *(mandatory)*

### Functional Requirements

#### Configuration Layer

- **FR-X1**：每个 agent 的 prompt 配置 MUST 存放在 `config/agents/<agent_id>.md`，格式为 YAML frontmatter（必填字段）+ Markdown body，body 至少含 5 个 `## <section_name>` 标题段落（system_prompt / user_tail / available_skills / recent_memory / output_schema）。
- **FR-X2**：YAML frontmatter MUST 含以下字段：`agent_id`（与文件名一致）、`description`（一句话描述）、`sections`（list[str]，声明 body 内必须存在的 section）、`budget`（int，token 预算）、`priority`（dict[str, int]，section→优先级，数字越小越保留）。可选字段：`slot_overrides`（dict[str, list[str]]，覆盖默认 system/user_tail 槽位分配）。
- **FR-X3**：~~`config/agents/` 目录 MUST 至少含 4 个 agent 文件~~ → **移至 spec 017b**。本 spec 仅创建空目录 `config/agents/` 与 fixture `tests/fixtures/agent_configs/example.md`。
- **FR-X4**：`ConfigLoader.load(path: Path) -> AgentConfig` MUST 校验单个 config 文件：YAML 可解析、必填字段齐全、`sections` 声明的段落在 body 中均存在；任一不满足抛 `ConfigValidationError`。批量启动期校验由 spec 017b 的调用方负责。

#### PromptBuilder Runtime

- **FR-X5**：`src/cryptotrader/agents/prompt_builder.py` MUST 实现 `PromptBuilder` 类，构造签名为 `PromptBuilder(agent_id: str, config_dir: Path, memory_provider: MemoryProvider, skill_provider: SkillProvider, model: str)`。
- **FR-X6**：`PromptBuilder.build(snapshot, portfolio, agent_analyses=None) -> tuple[SystemMessage, UserMessage]` 是唯一对外方法，返回 LangChain message 对象供 agent 直接传给 LLM。
- **FR-X7**：默认槽位分配：`system_prompt` / `available_skills` / `output_schema` 入 SystemMessage；`recent_memory` / snapshot / portfolio / agent_analyses / `user_tail` 入 UserMessage。`slot_overrides` 可覆盖默认。
- **FR-X8**：`MemoryProvider` 协议（`Protocol`）：`get_recent_memory(agent_id, snapshot, k=N) -> str`，返回已格式化的 markdown 文本（含 patterns + cases，按 spec 016 D-MW-01）。
- **FR-X9**：`SkillProvider` 协议（`Protocol`）：`get_available_skills(agent_id, snapshot, k=N) -> list[Skill]`，返回 ranked skill 列表。
- **FR-X10**：`DefaultMemoryProvider` MUST 复用 spec 014 的 `agent_memory/` 目录结构（patterns.md + cases.jsonl）；`DefaultSkillProvider` MUST 复用 spec 014 的 `agent_skills/` 目录结构（SKILL.md 文件协议）。

#### Token Budget Enforcer

- **FR-X11**：`TokenBudgetEnforcer.enforce(sections: dict[str, str], budget: int, priority: dict[str, int]) -> EnforceResult` 必须按优先级丢/降 section：
  1. 估算总 token，若 ≤ budget 直接返回
  2. 按 priority 从低到高（数字越大越先丢）依次丢 section，记录到 `dropped_sections`
  3. 若仅丢仍超 budget，对 `recent_memory` / `available_skills` 截断（保留前 N 条），记录到 `degraded_sections`
  4. `system_prompt` 和 `output_schema` MUST 保留（不可丢、不可降）
- **FR-X12**：`EnforceResult` dataclass 字段：`final_sections: dict[str, str]`、`dropped_sections: list[str]`、`degraded_sections: list[str]`、`prompt_size_pre: int`、`prompt_size_post: int`、`budget: int`。
- **FR-X13**：Token 估算 MUST 使用 CJK-aware 启发式（复用 spec 014 `_estimate_tokens()`：ASCII÷4 + CJK÷1.5），估算误差 < 10%（与 tiktoken 实测对比）。

#### Memory & Skill Composition

- **FR-X14**：`recent_memory` section MUST 同时包含 patterns（精炼规则）+ cases（具体历史案例），按 spec 016 D-MW-01 决策；具体格式由 `DefaultMemoryProvider.get_recent_memory()` 实现。
- **FR-X15**：`available_skills` section MUST 列出 ranked top-k skills（k 由 config `priority` 段或 `budget` 间接控制），格式为 markdown bullet list（skill_id + 描述 + 关键步骤摘要）。

#### Migration（移至 spec 017b）

- ~~**FR-X16** / **FR-X17**~~：4 agent ROLE 退役 + 构造器签名变更 + 调用方更新 → **全部移至 spec 017b**。本 spec **不修改** `src/cryptotrader/agents/{tech,chain,news,macro}.py`、`base.py`、`config.py`、`nodes/agents.py`、`graph.py` 任何文件。

#### Telemetry

- **FR-X18**：每次 `PromptBuilder.build()` 调用 MUST 写入以下 telemetry 字段（trace span attribute 或 structured log）：
  - `prompt.builder.agent_id`
  - `prompt.builder.sections_included` (list[str])
  - `prompt.builder.dropped_sections` (list[str])
  - `prompt.builder.degraded_sections` (list[str])
  - `prompt.builder.prompt_size_pre` (int, tokens)
  - `prompt.builder.prompt_size_post` (int, tokens)
  - `prompt.builder.budget` (int)
  - `prompt.builder.duration_ms` (float)
- **FR-X19**：Telemetry 字段 MUST 在现有 OpenTelemetry tracing 体系内（spec 010 落地）注入，不引入新依赖；`opentelemetry` MUST 是软依赖（缺失时降级 structured log，不抛 ImportError）。

### Key Entities

- **AgentConfig**：单个 agent 的配置记录。属性：`agent_id`、`description`、`sections`（list[str]）、`budget`（int）、`priority`（dict[str, int]）、`slot_overrides`（dict[str, list[str]]，可选）、`body_sections`（dict[str, str]，section_name → markdown body）。来源：`config/agents/<agent_id>.md` frontmatter + body。
- **PromptBuilder**：运行时 prompt 组装器。依赖 `AgentConfig` + `MemoryProvider` + `SkillProvider`。产出 `(SystemMessage, UserMessage)`。
- **MemoryProvider**：记忆数据源协议。方法：`get_recent_memory(agent_id, snapshot, k) -> str`。本 spec 提供 `DefaultMemoryProvider`，spec 018 提供进化版实现。
- **SkillProvider**：技能数据源协议。方法：`get_available_skills(agent_id, snapshot, k) -> list[Skill]`。本 spec 提供 `DefaultSkillProvider`，spec 018 提供进化版实现。
- **Skill**：单个技能记录。来自 `agent_skills/<id>/SKILL.md`，含 `skill_id`、`description`、`steps`（关键步骤摘要）等字段（沿用 spec 014 schema）。
- **TokenBudgetEnforcer**：token 预算执行器。方法：`enforce(sections, budget, priority) -> EnforceResult`。
- **EnforceResult**：dataclass，字段：`final_sections`、`dropped_sections`、`degraded_sections`、`prompt_size_pre`、`prompt_size_post`、`budget`。
- **ConfigValidationError**：启动期 config 校验失败时抛出，含失败原因（缺字段 / YAML 损坏 / sections 声明缺失等）。

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-X1**：`tests/fixtures/agent_configs/example.md` fixture 文件存在，frontmatter 合法，body 至少含 5 个声明 section。~~`config/agents/{tech,chain,news,macro}.md` 4 个文件~~ → 移至 SC-017b-X1。
- **SC-X2**：`tests/test_prompt_builder.py` 至少含 7 个用例并 PASS：(a) 加载合法 config；(b) 缺字段抛 `ConfigValidationError`；(c) 拼接产出 SystemMessage + UserMessage；(d) memory_provider 空返回走占位；(e) skill_provider 空返回走占位；(f) `slot_overrides` 生效；(g) snapshot/portfolio 字段缺失走默认占位。
- **SC-X3**：`tests/test_token_budget.py` 至少 5 用例 PASS：(a) 不超 budget 不丢；(b) 超 budget 按优先级丢；(c) 远超 budget 触发降级；(d) system_prompt + output_schema 强制保留；(e) 估算误差 < 10%（与 tiktoken 对比 CJK + ASCII 混合样本）。
- **SC-X4** ~ **SC-X9**：~~4 agent 迁移 / E2E / telemetry 全链路 / 行数 / ROLE 退役~~ → **全部移至 spec 017b**。
- **SC-X10**：通过 `/spex:review-spec` 无 P0 / P1 issues。
- **SC-X11**（新增）：`pytest tests/test_config_loader.py tests/test_token_budget.py tests/test_prompt_builder.py -v` 全部 PASS（基建测试集）。
- **SC-X12**（新增）：`from cryptotrader.agents.prompt_builder import PromptBuilder, MemoryProvider, SkillProvider, Skill, DefaultMemoryProvider, DefaultSkillProvider, ConfigValidationError, EnforceResult, TokenBudgetEnforcer` 全部 import 成功（API 公开契约）。
- **SC-X13**（新增）：本 spec 落地 commit 不修改 `src/cryptotrader/agents/{tech,chain,news,macro}.py` / `base.py` / `config.py` / `nodes/agents.py` / `graph.py` 任一文件（隔离性 gate）。

## Assumptions

- 4 个 agent 当前 ROLE 字符串语义稳定，可一次性外置；外置过程不重写 prompt 内容（仅搬运 + 槽位化）。
- 迁移期间 `agent_memory/` / `agent_skills/` schema 无并发变更（spec 018 等本 spec 完成后再启动）。
- 用户接受"删旧即新"策略，不保留 `prompt_builder=None` fallback；回滚走 git revert 而非运行时开关。
- 现有 OpenTelemetry tracing 基础设施（spec 010 落地）可直接挂载新 telemetry 字段，不需扩容采样率或新增 collector。
- spec 014 的 `_estimate_tokens()` CJK-aware 估算已被验证误差 < 10%（无需本 spec 重新校准）。
- LangChain `SystemMessage` / `HumanMessage` 类型为既定 LLM 消息载体（已是项目约定）。

## Dependencies

**Upstream**：

- Spec 016 Phase 1 已完成（提供 D-PA-01..06、D-MW-01..03 决策依据）
- Spec 014 `agent_skills/` + `agent_memory/` 目录结构（作为 Provider 数据源）
- Spec 010 OpenTelemetry tracing 基础设施

**Downstream**：

- Spec 018（skill-evolution-v2）依赖本 spec 落地，复用 `MemoryProvider` / `SkillProvider` 协议接口与 PromptBuilder 拼接管线

**External tooling**：PyYAML（已有依赖）；token 估算用 spec 014 已有 `_estimate_tokens()`，不引入 tiktoken 依赖。

## Out of Scope（含从 spec 拆分迁出项）

**从 spec 017a 拆分迁出（移至 spec 017b）**：
- 4 agent 配置文件创建（`config/agents/{tech,chain,news,macro}.md`）
- 4 agent 源码 ROLE 常量删除
- `base.py` BaseAgent / ToolAgent 接入 PromptBuilder
- `config.py` AgentRegistry 的 `_resolve_role` / `_resolve_skills` 重构
- `agents/skills/middleware.py:SkillsInjectionMiddleware` 退役
- `nodes/agents.py` / `graph.py` 启动期实例化注入
- E2E 测试 `tests/test_e2e_prompt_externalization.py`
- ROLE grep CI gate

**原 OOS 保留**：
- Skill / Memory 进化算法 —— spec 018
- SKILL.md schema 升级 —— spec 018
- Skill retrieval 算法升级（IDF / Hermes match-score）—— spec 018
- Verdict / Debate / Risk gate 节点 prompt 外置 —— 单独 spec
- LLM 模型选型 / prompt 内容优化
- 新增 agent
- Frontend / API / Risk / Execution 层改动
- Anthropic prompt cache 配置 —— spec 018
- 配置热重载实现

## Reversibility

本 spec 落地后**不可逆**（按用户决策直接删旧）。回滚需 git revert T1-T6 全部 commit。降低风险措施：

- T1 基建独立 commit
- T2-T5 按 agent 逐个 commit（精准 revert 颗粒度）
- T6 E2E 测试单独 commit
- 整个 spec 落地预期 < 5 个 commit，便于全量回滚

## Implementation Outline

### 范围（已完成）

- T001 — 创建 `config/agents/`（空目录）
- T002-T013 — `src/cryptotrader/agents/prompt_builder.py`（666 行）：ConfigValidationError + AgentConfig + ConfigLoader + MemoryProvider/SkillProvider Protocols + Skill dataclass + EnforceResult + TokenBudgetEnforcer + DefaultMemoryProvider + DefaultSkillProvider + PromptBuilder + 模块导出
- T009-T011 — 3 个测试文件：`tests/test_config_loader.py`（9 项校验）/ `tests/test_token_budget.py`（5+ 用例）/ `tests/test_prompt_builder.py`（7+ 用例）
- T012 — `tests/fixtures/agent_configs/example.md`
- T014 — pytest 全部 44 用例 PASS

### 不包含（移至 spec 017b）

- T015-T034：4 agent ROLE 退役 + `base.py` / `config.py` 重构 + 调用方更新
- T035-T037：E2E + telemetry 全链路测试（依赖 4 agent 已迁移）
- T038-T043：lint + 回归 + final grep gate

### 后续 spec 衔接

- **spec 017b**（待立项）：完成 4 agent 迁移与 ROLE 退役
- **spec 018**（已规划）：注入 `EvolvingMemoryProvider` / `EvolvingSkillProvider` 替代 Default 实现，引入进化算法

# Feature Specification: Agent Prompt Builder Integration（spec 017b）

**Feature Branch**: `018-agent-prompt-builder-integration`
**Created**: 2026-05-08
**Status**: Draft
**Input**: User description: "spec-017b-agent-prompt-builder-integration — 4 agent 真正切换到 PromptBuilder + 删除 SkillsInjectionMiddleware / ANALYSIS_FRAMEWORK / role_description / prompt_template / _resolve_role / _resolve_skills，直接删旧不留 fallback。"

> **目录命名说明**：本 spec 是 spec 017a（specs/017-agent-prompt-externalization/）的续篇，逻辑名 "spec 017b"。但 spec-kit 按递增序号生成，分配到 018。本文档内引用一律称 "spec 017b" 以保持上下文连续性。

## Purpose

承接 spec 017a 的 PromptBuilder 基建（已合并 main，commit `cfd3acc` + `f1e37a9`）。017a 仅交付了基础模块 + 单测，4 个 analysis agent（tech / chain / news / macro）仍跑硬编码 ROLE 路径，没有真正接到 PromptBuilder。

本 spec 完成"集成切换"工作，包含 3 块整合：

1. **`base.py`** 的 `BaseAgent` / `ToolAgent` 重构 — 删 `role_description` 字段、删 `ANALYSIS_FRAMEWORK` 常量、`analyze()` 改用 `prompt_builder.build()`
2. **`config.py`** 的 `AgentsConfig.build()` 重构 — 删 `_resolve_role` / `_resolve_skills` / `prompt_template` 字段；接受 `prompt_builder` 注入
3. **`agents/skills/middleware.py`** 删除 — 由 `DefaultSkillProvider` + `PromptBuilder._render_skills()` 替代；`load_skill_tool` 保留为独立 LangChain tool

落地后 `grep -rn "^ROLE\s*=" src/cryptotrader/agents/` 返回空；4 agent 文件每个 < 150 行；`agents/skills/middleware.py` 不存在。

本 spec **直接删旧不留 fallback**（用户明确决策）。回滚走 git revert C1+C2+C3 三个 commit。

## User Scenarios & Testing *(mandatory)*

### User Story 1 - 4 agent 配置驱动 prompt (Priority: P1) 🎯 MVP

作为架构师 / Prompt Engineer，我需要 4 个 analysis agent 的 prompt 真的能从 `config/agents/<name>.md` 读取并被 LLM 使用，使生产环境修改 prompt 只需改 markdown 文件 + 重启服务（不改 Python 代码）。

**Why this priority**：spec 017a 只交付了基建，4 agent 实际还跑硬编码 ROLE 路径；本 user story 让 spec 017a 的基建真正生效。

**Independent Test**：修改 `config/agents/tech.md` 的 `system_prompt` 段一段文字 → 重启 API → 触发 1 次 cycle → mocked LLM 收到的 SystemMessage 含修改后的内容。

**Acceptance Scenarios**：

1. **Given** `config/agents/tech.md` 存在合法 frontmatter，**When** TechAgent 被 nodes/agents.py 实例化并调用 `analyze()`，**Then** PromptBuilder 从 config 读取拼装 SystemMessage 与 HumanMessage
2. **Given** 4 个 agent 全部迁移完成，**When** `grep -rn "^ROLE\s*=" src/cryptotrader/agents/`，**Then** 返回空
3. **Given** `wc -l src/cryptotrader/agents/{tech,chain,news,macro}.py`，**Then** 每个文件 < 150 行

---

### User Story 2 - Skill 加载语义零回归 (Priority: P1)

作为架构师，我需要 `SkillsInjectionMiddleware` 删除后，LLM 看到的 skill 内容（完整 SKILL.md body）与现状语义等价，不能因迁移而丢失现有 skill 信息。

**Why this priority**：skill 内容是 4 agent 决策的关键 context，删除不能附带语义降级（spec 018 才是按 ranking 截断 skill 的合适落点）。

**Independent Test**：构造一个 fixture skill `agent_skills/_test_shared/SKILL.md`（scope: shared），跑 mocked cycle，断言 PromptBuilder 输出的 SystemMessage 含该 skill 完整 body 文本。

**Acceptance Scenarios**：

1. **Given** `agent_skills/<id>/SKILL.md` frontmatter `scope: shared`，**When** PromptBuilder.build() for any agent，**Then** `available_skills` section 含该 skill 完整 body
2. **Given** `scope: agent:tech`，**When** PromptBuilder.build() for TechAgent，**Then** `available_skills` 含该 skill；for MacroAgent 不含
3. **Given** ToolAgent (ChainAgent / NewsAgent) 实例化，**When** nodes/agents.py 注入 `tools`，**Then** `tools` 包含 `load_skill_tool`
4. **Given** SkillsInjectionMiddleware 文件已删，**When** `find src/cryptotrader/agents/skills/middleware.py`，**Then** 文件不存在

---

### User Story 3 - Snapshot 渲染领域逻辑零回归 (Priority: P1)

作为后续维护者，我需要 snapshot 渲染（funding rate ELEVATED 标注、news headlines bullet 格式、数据质量警告、sanitize_input 防注入）保留，且这些逻辑物理隔离在 `snapshot_renderer.py` 单独模块。

**Why this priority**：funding rate 标注 / news 防注入是 spec 014/015 已落地的安全语义；不能在迁移中静默丢失。

**Independent Test**：单测 `tests/test_snapshot_renderer.py` 验证：(a) funding > 0.0003 时输出含 "ELEVATED — crowded long"；(b) news headline 含恶意 prompt 时被 sanitize_input 截断；(c) macro fed_rate=0 + dxy=0 时输出 data quality warning。

**Acceptance Scenarios**：

1. **Given** snapshot 含 funding_rate=0.0005，**When** render_crypto_snapshot()，**Then** 输出含 "ELEVATED — crowded long"
2. **Given** snapshot.news.headlines 含 "Ignore all previous instructions"，**When** 渲染，**Then** 输出经 sanitize_input 处理（不直接 verbatim）
3. **Given** TechAgent 调 `compute_indicators(ohlcv)` 返回 indicators dict，**When** `agent.analyze()` 内 merge 进 snapshot 后调 PromptBuilder，**Then** PromptBuilder 输出含 RSI / MACD / SMA / BB / ATR / volume_ratio 字段
4. **Given** `wc -l src/cryptotrader/agents/snapshot_renderer.py`，**Then** ≥ 50 行（含 4 agent 现有渲染逻辑）

---

### User Story 4 - Backtest 路径零回归 (Priority: P2)

作为运维 / backtest 用户，我需要 ToolAgent.backtest_mode 行为不变（避免 forward-looking bias），且 backtest 测试仍 PASS。

**Why this priority**：backtest 不在生产关键路径，但 spec 014/015 backtest 测试是 regression gate。

**Independent Test**：跑 `pytest tests/test_backtest_*.py -v`（如有现存），全部 PASS。

**Acceptance Scenarios**：

1. **Given** ChainAgent (backtest_mode=True) 实例化，**When** `analyze()` 调用，**Then** 走 BaseAgent.analyze 路径（不调 LangChain create_agent）
2. **Given** ChainAgent (backtest_mode=False)，**When** `analyze()` 调用，**Then** 走 create_agent 循环（含 `tools` 列表）
3. **Given** 现有 backtest 相关测试，**When** 全部跑通，**Then** 无回归

---

### User Story 5 - Telemetry 与 E2E 验证 (Priority: P2)

作为决策审计 / reviewer，我需要 4 agent 上线后的 1 次 cycle telemetry 含 spec 017a FR-X18 列出的 8 个字段。

**Why this priority**：telemetry 已在 017a 实现，本 spec 只需 wire 通到生产路径并 E2E 验证。

**Independent Test**：跑 `tests/test_e2e_prompt_externalization.py`，断言 mocked cycle 中 4 agent 的 OpenTelemetry span 含 8 字段。

**Acceptance Scenarios**：

1. **Given** mocked LangGraph cycle，**When** 4 agent → debate → verdict → risk gate 全链路跑完，**Then** 4 个 agent 各自 trace span 含 8 字段（agent_id / sections_included / dropped_sections / degraded_sections / prompt_size_pre / prompt_size_post / budget / duration_ms）
2. **Given** 现有 cycle 工具链（spec 010 OTel），**When** 生产 trigger 1 次 cycle，**Then** trace 后端可查到字段（手动 smoke test，不在自动测试内）

---

### Edge Cases

- `agent_memory/cases/` 为空（首次运行）→ recent_memory section 占位（"暂无历史记忆"），不报错
- `agent_skills/` 0 条匹配 scope filter → available_skills section 占位（"暂无可用技能"）
- `BaseAgent.analyze()` 调用方传非空 `experience: str` → PromptBuilder 直接用作 recent_memory，跳过 MemoryProvider
- `ToolAgent.backtest_mode=True` → 走 BaseAgent.analyze 路径（PromptBuilder 单次调用）
- `nodes/agents.py` 同 cycle 多次调 build() → 用 `_prompt_builders` dict cache 同 agent_id 实例
- 进程内未在 OpenTelemetry tracing 上下文 → telemetry 字段降级到 structured log（沿用 017a 实现）

## Requirements *(mandatory)*

### Functional Requirements

#### Configuration（4 agent config 文件）

- **FR-Y1**：`config/agents/{tech,chain,news,macro}.md` 4 个文件 MUST 存在，frontmatter 合法（agent_id / description / sections / budget / priority），body 至少含 5 个 `## section_name` 段落（system_prompt / user_tail / available_skills / recent_memory / output_schema）
- **FR-Y2**：每个 config 文件的 `system_prompt` 段 MUST 含：(a) 该 agent 当前 ROLE 字符串内容（搬运不重写）；(b) ANALYSIS_FRAMEWORK 的 discipline 部分（Rules + Pre-signal checklist + Confidence calibration + Data sufficiency self-assessment）
- **FR-Y3**：每个 config 文件的 `output_schema` 段 MUST 含 ANALYSIS_FRAMEWORK 的 JSON schema 部分（"CRITICAL: Output ONLY a JSON object..." + JSON 字段定义）

#### base.py 重构

- **FR-Y4**：`BaseAgent.__init__(self, *, agent_id: str, prompt_builder: PromptBuilder, model: str = "")` 签名，`role_description` 参数删除（必填 kw-only）
- **FR-Y5**：`BaseAgent.role_description` 字段删除；`self._prompt_builder` 替代
- **FR-Y6**：`BaseAgent.analyze(snapshot, experience: str = "")` 重写：experience 参数保留并直接传给 `prompt_builder.build(snapshot, portfolio={}, experience=experience)`
- **FR-Y6b**：`PromptBuilder.build(snapshot, portfolio, agent_analyses=None, experience: str = "")` 签名扩展加 experience 参数；experience 非空时直接作为 `recent_memory` section 内容；空时调 `memory_provider.get_recent_memory()` fallback（沿用 017a 默认行为）
- **FR-Y7**：`BaseAgent._build_prompt()` 方法删除（snapshot 渲染由 PromptBuilder 通过 snapshot_renderer 处理）
- **FR-Y8**：`ANALYSIS_FRAMEWORK` 模块级常量从 `base.py` 删除
- **FR-Y9**：`ToolAgent.__init__` 签名改 `(*, agent_id, prompt_builder, tools, model="", backtest_mode=False)`，`role_description` 参数删除
- **FR-Y10**：`ToolAgent.analyze()` 重写：`backtest_mode=True` → `super().analyze()`；`False` → `prompt_builder.build()` → `create_agent(llm, tools=self.tools, system_prompt=sys_msg.content)`；删除 SkillsInjectionMiddleware 调用

#### snapshot_renderer.py（新模块）

- **FR-Y11**：`src/cryptotrader/agents/snapshot_renderer.py` MUST 创建，含 `render_crypto_snapshot(snapshot: dict, experience: str = "") -> str` 函数
- **FR-Y12**：`render_crypto_snapshot()` MUST 包含从 `BaseAgent._build_prompt()` 搬运的全部逻辑：Pair / Timestamp / Ticker / Volatility / Funding rate（含 ELEVATED / NEGATIVE 标注）/ Futures volume 含 SPIKE / LOW 标注 / Open interest / News headlines（每条经 `sanitize_input`）/ Data quality warnings / experience 字段（若提供）经 `sanitize_input(max_chars=4000)`
- **FR-Y13**：`render_crypto_snapshot()` MUST 兼容 TechAgent 的额外 indicators 字段（snapshot dict 中含 `indicators` key 时附加 "Technical Indicators:" 段）
- **FR-Y14**：`PromptBuilder._render_snapshot()` MUST 改为调 `render_crypto_snapshot()`（默认行为）

#### 4 agent 类重构

- **FR-Y15**：`src/cryptotrader/agents/{tech,chain,news,macro}.py` 每个文件 MUST：(a) 删除 `ROLE = """..."""` 模块级常量；(b) 删除 `_build_prompt()` 方法（如有 override）；(c) 构造器改 `__init__(self, *, prompt_builder: PromptBuilder, model: str = "", [tools=..., backtest_mode=...])`
- **FR-Y16**：`TechAgent.compute_indicators(ohlcv)` 函数保留（仅计算逻辑，不含渲染）；`TechAgent.analyze()` 在调 `prompt_builder.build()` 前 MUST 把 `compute_indicators(snapshot.market.ohlcv)` 结果合并到 snapshot dict 的 `indicators` 字段
- **FR-Y17**：`grep -rn "^ROLE\s*=" src/cryptotrader/agents/` MUST 返回空
- **FR-Y18**：`wc -l src/cryptotrader/agents/{tech,chain,news,macro}.py` 每个 MUST < 150 行

#### config.py 重构

- **FR-Y19**：`AgentsConfig.build(agent_id, *, prompt_builder: PromptBuilder, backtest_mode=False, model_override="")` 签名加必填 `prompt_builder` 参数；删除 `regime_tags` 参数
- **FR-Y20**：`AgentsConfig._resolve_role()` 方法删除
- **FR-Y21**：`AgentsConfig._resolve_skills()` 方法删除
- **FR-Y22**：`AgentConfig.prompt_template` 字段删除
- **FR-Y23**：`AgentsConfig._build_builtin()` 重构：调用 4 个 builtin agent 的新构造签名
- **FR-Y24**：`agent.role_description += "STRATEGY SKILLS"` 拼接代码（共 2 处）删除

#### SkillsInjectionMiddleware 删除

- **FR-Y25**：`src/cryptotrader/agents/skills/middleware.py` 文件 MUST 删除
- **FR-Y26**：任何 `from cryptotrader.agents.skills.middleware import` 引用 MUST 删除（base.py:600 等）
- **FR-Y27**：`load_skill_tool` MUST 由 nodes/agents.py 显式 import 并注入 ToolAgent.tools

#### DefaultSkillProvider 修正

- **FR-Y28**：`DefaultSkillProvider.get_available_skills()` MUST 用 `discover_skills_for_agent(agent_id)` 替代 017a 当前 `agent_id in skill.tags` 过滤（与 spec 014 SKILL.md `scope: shared/agent:<id>` 语义一致）
- **FR-Y29**：`PromptBuilder._render_skills()` MUST 渲染完整 skill body（格式 `\n\n---\n## Skill: {name}\n\n{body}` 与 SkillsInjectionMiddleware 一致），不是 summary bullet
- **FR-Y30**：`Skill` dataclass 的 `name` 字段（如不存在）MUST 加入

#### nodes/agents.py wiring

- **FR-Y31**：`src/cryptotrader/nodes/agents.py` 顶层 MUST 增加 module-level singleton：`_memory_provider: DefaultMemoryProvider | None = None` / `_skill_provider: DefaultSkillProvider | None = None` / `_prompt_builders: dict[str, PromptBuilder] = {}`
- **FR-Y32**：`_get_or_build_pb(agent_id, model)` helper 函数 MUST 存在，lazy-init Provider 单例 + 缓存 PromptBuilder per agent_id
- **FR-Y33**：`nodes/agents.py:53` 调用 `cfg.agents.build(...)` 处 MUST 传入 `prompt_builder=_get_or_build_pb(agent_id, model_override)`；`regime_tags` 传参一并清理
- **FR-Y34**：`load_skill_tool` MUST 在 ToolAgent 实例化前 import 并加到 `tools` 列表（针对 chain / news 2 个 ToolAgent）

#### graph.py / 其他调用方

- **FR-Y35**：`src/cryptotrader/graph.py` 若有 agent 实例化代码 MUST 同步更新；spot-check 显示无匹配，FR-Y35 视作 NOOP
- **FR-Y36**：`src/cryptotrader/security.py` 中引用 `role_description` 的注释 MUST 更新为引用 `prompt_builder` / config

#### Telemetry

- **FR-Y37**：1 次 cycle 后 4 agent 的 OpenTelemetry span MUST 含 spec 017a FR-X18 列出的 8 字段（不需要本 spec 实现，仅 E2E 测试断言）

#### Migration（直接切换，无 fallback）

- **FR-Y38**：`prompt_builder` 参数 MUST 必填（无默认值）
- **FR-Y39**：本 spec 不引入任何运行时 feature flag / 环境变量切换新旧 prompt 路径

### Key Entities

- **AgentConfigFile**：`config/agents/<agent_id>.md`，YAML frontmatter + Markdown body；frontmatter 字段沿用 spec 017a `agent-config-schema.md`（agent_id / description / sections / budget / priority / slot_overrides 可选）；body 含 5 个核心 section（system_prompt / user_tail / available_skills / recent_memory / output_schema）
- **CryptoSnapshotRenderer**：`src/cryptotrader/agents/snapshot_renderer.py:render_crypto_snapshot(snapshot, experience) -> str`，crypto 领域专用 snapshot 渲染函数；含 funding annotation / news 防注入 / data quality warnings 等领域逻辑
- **PromptBuilderSingleton**：`src/cryptotrader/nodes/agents.py` module-level dict `_prompt_builders: dict[str, PromptBuilder]`，cycle 调用时按 agent_id 取缓存 PromptBuilder 实例

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-Y1**：`config/agents/{tech,chain,news,macro}.md` 4 个文件存在；frontmatter YAML 可解析；body 含 5 个声明 section
- **SC-Y2**：每个 config 文件 system_prompt 段含 ANALYSIS_FRAMEWORK discipline 部分（Rules / Pre-signal checklist / Confidence calibration / Data sufficiency）
- **SC-Y3**：每个 config 文件 output_schema 段含 ANALYSIS_FRAMEWORK JSON schema 部分（CRITICAL + 字段定义）
- **SC-Y4**：`grep -rn "^ROLE\s*=" src/cryptotrader/agents/` 返回空
- **SC-Y5**：`wc -l src/cryptotrader/agents/{tech,chain,news,macro}.py` 每个 < 150 行
- **SC-Y6**：`find src/cryptotrader/agents/skills/middleware.py` 不存在
- **SC-Y7**：`grep -rn "ANALYSIS_FRAMEWORK\|role_description\|prompt_template\|_resolve_role\|_resolve_skills\|SkillsInjectionMiddleware" src/cryptotrader/` 仅在 spec/test 文档，**不在** src/ .py 文件
- **SC-Y8**：`tests/test_snapshot_renderer.py` ≥ 6 用例 PASS：funding ELEVATED / NEGATIVE / news 防注入 / data warnings / experience cap / TechAgent indicators
- **SC-Y9**：`tests/test_{tech,chain,news,macro}_agent.py` 全部 PASS（每个含至少 1 用例验证 prompt 来自 config）
- **SC-Y10**：`tests/test_e2e_prompt_externalization.py` PASS：单 mocked cycle 跑 4 agent → debate → verdict → risk gate；4 agent 各 OTel span 含 8 字段；verdict 字段完整（含 target_price / stop_loss / take_profit / R:R）
- **SC-Y11**：017a 基建测试（`test_config_loader.py` / `test_token_budget.py` / `test_prompt_builder.py`）继续 PASS（44 用例不回归）
- **SC-Y12**：spec 014 / 015 既有测试不回归
- **SC-Y13**：`DefaultSkillProvider` 用 scope 过滤；fixture `agent_skills/_test_shared/SKILL.md` (scope: shared) 可被 4 agent 全部加载；`agent_skills/_test_tech/SKILL.md` (scope: agent:tech) 仅 TechAgent 加载
- **SC-Y14**：`PromptBuilder._render_skills()` 输出含完整 SKILL.md body
- **SC-Y15**：单次 cycle 触发后 OTel trace 后端可查到 4 agent 各 8 字段（手动 smoke test）
- **SC-Y16**：通过 `/spex:review-spec` 无 P0 / P1 issues
- **SC-Y17**：通过 `/spex:review-plan` 任务覆盖完整 + REVIEW-PLAN.md 生成

## Assumptions

- 4 agent 当前 ROLE 字符串语义稳定，可一次性外置；外置过程不重写 prompt 内容（仅搬运 + 槽位化）
- ANALYSIS_FRAMEWORK 内容 4 agent 共用（无 agent-specific 差异化需求）
- 现有 `agent_skills/<id>/SKILL.md` 文件 frontmatter `scope` 字段全部已正确填写（spec 014 落地保证）
- `discover_skills_for_agent` 函数可直接被 `DefaultSkillProvider` 复用，签名兼容
- `TechAgent.compute_indicators` 计算逻辑保留在 tech.py 不动
- `experience: str` 参数（`BaseAgent.analyze` 的 string 参数）由调用方传入，PromptBuilder 直接用作 `recent_memory` section（绕过 `DefaultMemoryProvider` 当前路径错误的实现 — 该路径修复推迟到 spec 018）
- spec 017a 的 PromptBuilder 公开 API（构造签名 / `build()` 返回值结构）稳定，不会因本 spec 修改

## Dependencies

**Upstream**：
- **spec 017a**（已合并 main，commit `cfd3acc` + merge `f1e37a9`）—— PromptBuilder / Provider Protocol / TokenBudgetEnforcer / DefaultMemoryProvider / DefaultSkillProvider / `Skill` dataclass / 模块导出
- **spec 014** —— `agent_skills/<id>/SKILL.md` 文件协议（含 frontmatter `scope` 字段）/ `discover_skills_for_agent()` / `load_skill_tool`
- **spec 010** —— OpenTelemetry tracing 基础设施
- **spec 015** —— `sanitize_input()` 防注入函数（`src/cryptotrader/security.py`）

**Downstream**：
- **spec 018**（待立项）—— skill / memory 进化算法可基于本 spec 完成的 4 agent 真实运行路径接入；spec 018 的 `EvolvingMemoryProvider` / `EvolvingSkillProvider` 替换 `DefaultMemoryProvider` / `DefaultSkillProvider` 即可

**External tooling**：无新依赖

## Out of Scope

**移至 spec 018（skill / memory 进化算法）**：
- Skill / Memory 进化算法（GEPA / Reflective Mutation / 5-stage / IDE/IVE/ESE / 5-signal FSM）
- SKILL.md schema 升级
- Skill retrieval 算法升级（IDF / Hermes match-score / regime-aware ranking）
- **DefaultMemoryProvider 路径修复**（patterns / cases 实际路径与 spec 014 对齐）—— 当前 017a Provider 是僵尸代码（路径错），spec 018 重写时一并修
- PromptBuilder 的 `snapshot_renderer` 注入参数化
- Anthropic prompt cache 配置
- `load_skill_tool` 删除决策

**本 spec 显式不动**：
- `AgentsConfig.regime_tags` 参数语义重设计 —— 仅删除该参数
- Verdict / Debate / Risk gate 节点 prompt 外置 —— 单独 spec
- 新增 agent
- Frontend / API / Risk / Execution 层改动
- 配置热重载
- prompt 内容优化 / LLM 模型选型（仅迁移搬运，不重写）

## Reversibility

本 spec 落地后**不可逆**（按用户决策直接删旧）。回滚需 git revert C1+C2+C3 三个 commit。降低风险措施：
- C1 commit 是纯新增（`config/agents/*.md` + `snapshot_renderer.py` + `test_snapshot_renderer.py`），无 behavior 变化；commit 后 main 仍正常运行
- C2 commit 是 atomic 切换；如 CI 失败可全量 revert C2，main 回到 C1 状态（仍正常）
- C3 commit 是 E2E + final gate；commit 后 final 验收

## Implementation Outline

### Commit 序列（Q6 B：3 commit 单 PR）

#### C1 — 纯新增（无 behavior 变化）

文件创建：
- `config/agents/tech.md`（NEW）
- `config/agents/chain.md`（NEW）
- `config/agents/news.md`（NEW）
- `config/agents/macro.md`（NEW）
- `src/cryptotrader/agents/snapshot_renderer.py`（NEW）

测试：
- `tests/test_snapshot_renderer.py`（NEW，≥ 6 用例）

CI 状态：所有现有测试 PASS（无代码路径变化）；新增测试 PASS。

预估 diff：~700 行

#### C2 — Atomic 切换

**删除**：
- `src/cryptotrader/agents/skills/middleware.py`（整个文件）
- `src/cryptotrader/agents/base.py:ANALYSIS_FRAMEWORK` 常量
- `src/cryptotrader/agents/base.py:BaseAgent.role_description` 字段
- `src/cryptotrader/agents/base.py:BaseAgent._build_prompt()` 方法
- `src/cryptotrader/agents/{tech,chain,news,macro}.py:ROLE` 常量（4 处）
- `src/cryptotrader/agents/{tech,chain,news,macro}.py:_build_prompt()` 方法（如有）
- `src/cryptotrader/config.py:AgentConfig.prompt_template` 字段
- `src/cryptotrader/config.py:AgentsConfig._resolve_role()` 方法
- `src/cryptotrader/config.py:AgentsConfig._resolve_skills()` 方法
- `src/cryptotrader/config.py:agent.role_description += "STRATEGY SKILLS"` 拼接代码（2 处）

**重构**：
- `src/cryptotrader/agents/base.py`：BaseAgent / ToolAgent 构造器 + analyze() 重写
- `src/cryptotrader/agents/{tech,chain,news,macro}.py`：4 处构造器 + analyze() 重构
- `src/cryptotrader/agents/prompt_builder.py`：`_render_skills()` 完整 body / `DefaultSkillProvider` scope filter / `_render_snapshot()` 调 snapshot_renderer / `Skill.name` 字段 / `build()` 加 experience 参数
- `src/cryptotrader/config.py`：AgentsConfig.build() 签名 + _build_builtin() 重写
- `src/cryptotrader/nodes/agents.py`：module-level singleton + `_get_or_build_pb()` helper + load_skill_tool 注入
- `src/cryptotrader/security.py`：注释更新

**测试更新**：
- `tests/test_tech_agent.py` / `test_chain_agent.py` / `test_news_agent.py` / `test_macro_agent.py`

CI 状态：所有测试 PASS

预估 diff：~1100 行

#### C3 — E2E + 最终 Gate

文件：
- `tests/test_e2e_prompt_externalization.py`（NEW）
- `pyproject.toml`：新文件 RUF001-003 per-file-ignores（如需要）

验证：
- `pytest tests/test_e2e_prompt_externalization.py -v` PASS
- `grep -rn "^ROLE\s*=" src/cryptotrader/agents/` 返回空
- `grep -rn "ANALYSIS_FRAMEWORK\|role_description\|prompt_template\|SkillsInjectionMiddleware" src/cryptotrader/` 仅在 spec/test 文档
- `wc -l src/cryptotrader/agents/{tech,chain,news,macro}.py` 每个 < 150 行
- `ruff check src/cryptotrader/agents/ tests/` PASS
- `pytest tests/ -x --ignore=tests/test_e2e_prompt_externalization.py` 无回归

预估 diff：~250 行

### 任务总数

约 35 task。具体细分由 `/speckit-tasks` 生成。

### 估时

| 阶段 | 工作量 |
|---|---|
| C1（config + snapshot_renderer + 单测） | 0.5 天 |
| C2（atomic 切换） | 1 天 |
| C3（E2E + gate） | 0.5 天 |
| Code review + 修复 | 0.5 天 |
| **合计** | **2.5 天** |

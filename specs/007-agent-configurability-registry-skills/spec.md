# 功能规格说明：Agent 可配置化 — Registry + Skills

**Feature Branch**: `007-agent-configurability-registry-skills`
**Created**: 2026-04-17
**Status**: Draft

---

## 背景

CryptoTrader-AI 当前系统运行 4 个硬编码 Agent（TechAgent、ChainAgent、NewsAgent、MacroAgent），
通过 `asyncio.gather()` 并行执行，Agent 的模型、提示词（role_description）、工具集和超时时间
全部在源码中写死。随着系统规模增长和使用场景多样化，用户无法在不修改代码的情况下：

- 调整单个 Agent 使用的模型或提示风格
- 禁用与当前行情分析无关的 Agent
- 添加新的专项 Agent（如衍生品专家、链上鲸鱼监控 Agent）
- 针对不同市场 Regime（趋势市、震荡市、极度恐慌）动态切换分析策略

此外，现有的 `skills.py` 中虽已有 `TRADING_SKILLS` 列表，但技能内容硬编码在 Python 代码中，
缺少用户可维护的声明式接口，也没有与 Regime 感知的经验记忆系统相结合。

本 Spec 旨在以"声明优于编码"的原则，将 Agent 配置与策略技能外置到 TOML 和 Markdown 文件，
实现 AgentRegistry（注册表）和 Skills（策略技能）两大能力。

---

## 用户场景与测试 *(必填)*

### 用户故事 1 — 通过 TOML 配置调整 Agent 行为 (优先级: P0)

作为运营者，我希望无需修改 Python 代码，只编辑 `config/local.toml` 就能为 NewsAgent 换用
更强的模型，并调整其分析超时时间。

**为何是 P0**：这是所有可配置化能力的最基础形态。若不能通过配置文件驱动 Agent 行为，
所有后续的 Registry/Skills 特性都失去落地基础。

**独立测试**：单元测试读取含 `[agents.news_agent]` 段落的 TOML，
验证构造出的 NewsAgent 持有正确的 model 和 timeout 值，无需依赖 LLM 调用。

**验收场景**：

1. **Given** `config/local.toml` 中写入 `[agents.news_agent] model = "gpt-5"` 和 `timeout_seconds = 45`，
   **When** 系统加载配置并构造 NewsAgent，
   **Then** NewsAgent 使用 `gpt-5` 模型，`asyncio.wait_for` 超时设为 45 秒。

2. **Given** `config/local.toml` 未配置任何 `[agents.*]` 段落，
   **When** 系统启动，
   **Then** 4 个内置 Agent 使用 `config/default.toml` 中的默认值，行为与现有系统完全一致。

3. **Given** `[agents.tech_agent] enabled = false` 写入配置，
   **When** 执行一次交易分析周期，
   **Then** 管道中不运行 TechAgent，verdict 步骤基于剩余 Agent 输出做出决策，无报错。

---

### 用户故事 2 — 自定义 Agent 覆盖内置 Agent (优先级: P1)

作为高级用户，我希望在 TOML 中声明一个与内置 TechAgent 同名的自定义配置，
用自己编写的提示词模板和更高精度模型覆盖默认行为。

**为何是 P1**：覆盖机制是扩展性的核心。P0 只处理参数调整，P1 处理用户完整自定义一个 Agent。

**独立测试**：构造含自定义 prompt_template 的 AgentConfig，
验证 BaseAgent 使用该 prompt_template 而非内置 ROLE 常量。

**验收场景**：

1. **Given** `config/local.toml` 中声明 `[agents.tech_agent]` 并指定
   `prompt_template = "prompts/my_tech_prompt.md"`，
   **When** 系统加载 AgentRegistry 并构建 TechAgent，
   **Then** TechAgent 的 `role_description` 来自该 Markdown 文件内容，而非代码内置的 `ROLE` 常量。

2. **Given** 自定义 prompt_template 文件路径不存在，
   **When** 系统启动，
   **Then** 系统抛出 `ConfigurationError`，携带清晰的错误信息指向具体字段路径，拒绝启动。

3. **Given** 自定义 Agent 使用与内置 Agent 不同的 `tools` 字段列表（如只保留 `get_derivatives_data`），
   **When** ChainAgent 执行分析，
   **Then** ToolAgent 仅能调用被声明工具，其余工具对该 Agent 不可见。

---

### 用户故事 3 — 添加全新的自定义 Agent (优先级: P1)

作为量化研究员，我希望在不修改 Python 源码的前提下，通过 TOML 声明一个专注于
链上鲸鱼行为分析的新 Agent，并将其纳入并行分析管道。

**为何是 P1**：这是系统可扩展性的核心证明——非内置 Agent 是否能被无缝注入管道。

**独立测试**：注册一个名为 `whale_agent` 的 AgentConfig，
验证 `nodes/agents.py` 中的 `_run_agent()` 能够从 Registry 中动态查找并构造该 Agent。

**验收场景**：

1. **Given** TOML 中新增 `[agents.whale_agent]` 并配置 `model`、`prompt_template` 和 `tools`，
   **When** 交易管道执行，
   **Then** `whale_agent` 与其他 Agent 并行运行，其分析结果被纳入 `state["data"]["analyses"]`，
   辩论和裁决步骤照常运行。

2. **Given** 新 Agent 名称在现有管道中既不是内置 Agent 也未被 `agents.yml` 声明，
   **When** 系统尝试构造该 Agent，
   **Then** AgentRegistry 抛出 `AgentNotFoundError`，日志中包含 Agent 名称和已注册 Agent 列表。

3. **Given** 新增 Agent 的总数量超过 4 个，
   **When** 执行分析，
   **Then** 辩论门控（debate gate）的共识计算基于所有活跃 Agent 的得分，而非硬编码的 4 个 Agent。

---

### 用户故事 4 — 声明式策略 Skill 文件 (优先级: P1)

作为策略研究员，我希望以 Markdown 文件描述一套交易策略（入场规则、止损规则、资金管理），
然后在 TOML 中引用该文件，让 Agent 在分析时自动获得策略约束提示。

**为何是 P1**：将策略规则与 Agent 实现解耦，策略可独立版本控制和迭代，无需修改代码。

**独立测试**：给定一个 Markdown skill 文件路径，验证 `SkillLoader.load()` 返回文件内容，
并能通过 `sanitize_input()` 注入 Agent 提示词。

**验收场景**：

1. **Given** `skills/momentum_strategy.md` 包含"入场：RSI < 35 且 MACD 金叉"等规则，
   且 `[agents.tech_agent] skills = ["momentum_strategy"]` 已配置，
   **When** TechAgent 构建提示词，
   **Then** 提示词中包含 `momentum_strategy.md` 的完整内容（经 sanitize_input 净化后），
   位于 `ANALYSIS_FRAMEWORK` 之前。

2. **Given** skill Markdown 文件包含 prompt injection 攻击内容（如 `Ignore all previous instructions`），
   **When** 内容经过 `sanitize_input()` 处理并注入提示词，
   **Then** 危险内容被净化，Agent 收到的提示词中不包含原始注入文本。

3. **Given** 引用的 skill 文件不存在，
   **When** AgentRegistry 启动加载，
   **Then** 系统记录 `WARNING` 日志，该 skill 被跳过，Agent 不加载该技能但仍正常运行。

---

### 用户故事 5 — Regime 自动匹配策略 Skill (优先级: P2)

作为量化研究员，我希望系统根据当前市场 Regime（如 `high_volatility`、`trending_up`）
自动为 Agent 注入对应的策略 Skill，而无需手动切换配置。

**为何是 P2**：这是 Skill 与 Regime 感知经验记忆的深度集成，建立在 P1 基础上，
需要现有 `regime_tags` 流水线稳定后再扩展。

**独立测试**：给定 `regime_tags = ["high_volatility"]` 和配置了 `regime_skills` 映射的 AgentConfig，
验证 `SkillSelector.select()` 返回匹配的 skill 内容列表。

**验收场景**：

1. **Given** `[agents.macro_agent.regime_skills]` 配置了 `"trending_up" = ["bull_momentum_skill"]`，
   且当前 `state["data"]["regime_tags"]` 包含 `"trending_up"`，
   **When** MacroAgent 构建提示词，
   **Then** 提示词包含 `bull_momentum_skill` 的内容，而不包含其他 Regime 的 skill 内容。

2. **Given** 当前 Regime 与所有 `regime_skills` 映射都不匹配，
   **When** Agent 构建提示词，
   **Then** 仅注入 `skills` 字段中声明的默认 skill（如有），Regime-specific skill 不注入。

3. **Given** 多个 Regime tag 同时存在（如 `["trending_up", "high_funding"]`），
   **When** SkillSelector 执行匹配，
   **Then** 所有匹配的 Regime skill 按优先级去重后合并注入，且合并后 token 总量不超过 `token_budget_pct` 上限。

---

### 用户故事 6 — Per-Agent 超时降级 (优先级: P0)

作为系统运维人员，我希望每个 Agent 拥有独立的超时配置，超时后以保守默认值降级
（返回空分析或 neutral），而不影响其他 Agent 的正常运行。

**为何是 P0**：当前 `timeout_seconds` 全局共享，任一 Agent 超时导致误配置时，
影响范围不可控。独立超时是生产环境的基本安全保障。

**独立测试**：Mock 一个 Agent 的 `analyze()` 使其永远挂起，
设置 `timeout_seconds = 0.01`，验证返回 `_MOCK_ANALYSIS_RESULT` 而非异常传播。

**验收场景**：

1. **Given** `[agents.chain_agent] timeout_seconds = 30` 而其他 Agent 保持默认 60 秒，
   **When** ChainAgent 的 LLM 调用耗时超过 30 秒，
   **Then** ChainAgent 超时降级，返回 `is_mock = true` 结果，其他 Agent 继续正常运行。

2. **Given** Agent 配置未显式设置 `timeout_seconds`，
   **When** Agent 执行分析，
   **Then** 使用 `config.models.timeout_seconds` 全局默认值，行为与当前系统一致。

3. **Given** Agent 超时降级，
   **When** 后续裁决步骤执行，
   **Then** `is_mock = true` 的分析结果在裁决权重计算中被折损（confidence 降为 0），
   裁决不因单个 Agent 降级而完全失效。

---

### 边界条件

- **空 agents 列表**：若所有内置 Agent 均被 `enabled = false`，系统拒绝启动并给出明确错误，
  不允许 0 个 Agent 进入管道。
- **同名 Agent 冲突**：内置 Agent 与用户自定义 Agent 同名时，用户自定义配置完全覆盖内置配置，
  内置代码中的 `ROLE` 常量不再被使用。
- **Skill 循环引用**：若 skill Markdown 文件中包含其他 skill 引用语法，该语法被视为普通文本，
  不进行递归加载。
- **TOML 格式错误**：`config/local.toml` 中 `[agents.*]` 段落格式非法时，
  系统抛出 `ConfigurationError` 并拒绝启动，日志中包含行号和字段路径。
- **Token 超限**：所有 skill 内容注入后超出 Agent 提示词 token 上限时，
  按优先级截断 skill，保留 `ANALYSIS_FRAMEWORK` 和 `role_description` 完整。
- **回测模式 Skill 注入**：回测模式下，Skill 内容照常注入（Skill 是声明式规则，不涉及实时数据），
  但 Regime 自动匹配基于回测快照的 `regime_tags`，非实时计算。

---

## 需求 *(必填)*

### 功能需求

**FR-001**：系统应支持在 `config/default.toml` 中新增 `[agents.<agent_id>]` 段落，
为每个 Agent 声明 `model`、`timeout_seconds`、`enabled`、`tools`、`prompt_template` 字段。
未声明的字段回退至系统全局默认值。

**FR-002**：系统应提供 `AgentRegistry`，在 `load_config()` 时解析所有 `[agents.*]` 段落，
构建 Agent 名称到 `AgentConfig` 的映射，并支持通过 `registry.get(agent_id)` 查询。

**FR-003**：`AgentRegistry.build(agent_id)` 应根据 `AgentConfig` 动态构造对应的 Agent 实例，
支持内置 Agent 类型（TechAgent、ChainAgent、NewsAgent、MacroAgent）和通过 `prompt_template`
声明的自定义 Agent 类型。

**FR-004**：当 `AgentConfig.enabled = false` 时，该 Agent 不参与当前分析周期，
`nodes/agents.py` 的并行执行列表应动态从 Registry 中读取活跃 Agent 列表，而非硬编码 4 个。

**FR-005**：用户自定义 Agent 若与内置 Agent 同名（如 `[agents.tech_agent]`），
则完全覆盖内置 Agent 配置，`prompt_template` 字段指定的外部 Markdown 文件内容
替换代码中的 `ROLE` 常量，覆盖关系在 `AgentRegistry` 构建时生效。

**FR-006**：系统应支持 `tools` 字段声明 Agent 可用的工具子集（工具名称字符串列表），
`ToolAgent` 在构建时只注入声明工具，未声明工具对该 Agent 不可访问，
实现工具集隔离（Tool Isolation）。

**FR-007**：系统应支持 `skills` 字段，接受 Markdown 文件名列表（不含路径和扩展名），
`SkillLoader` 从 `skills/` 目录加载对应 `.md` 文件内容，经 `sanitize_input()` 净化后
注入 Agent 提示词（追加在 `role_description` 之后、`ANALYSIS_FRAMEWORK` 之前）。

**FR-008**：系统应支持 `regime_skills` 字段，声明从 `regime_tag` 到 skill 文件名列表的映射，
在 Agent 构建提示词时，由 `SkillSelector` 查询当前 `regime_tags` 并自动注入匹配的 Skill 内容。
当无匹配时，回退至 `skills` 字段声明的默认 Skill。

**FR-009**：`AgentConfig` 的 `timeout_seconds` 字段若已设置，
则在 `_run_agent()` 的 `asyncio.wait_for()` 中使用该 Agent 级别的超时值，
替代 `config.models.timeout_seconds` 全局值，超时后行为与当前降级逻辑保持一致。

**FR-010**：`AppConfig` 中应增加 `AgentsConfig` 数据类，
包含 `dict[str, AgentConfig]` 字段，由 `_build_config()` 解析 `[agents.*]` TOML 段落构建。
现有 `ModelConfig` 中的 `tech_agent`、`chain_agent`、`news_agent`、`macro_agent` 字段
在此 Feature 结束后仍保留（向后兼容），但 `AgentConfig.model` 字段优先级更高。

**FR-011**：`validate_config()` 应对 `AgentsConfig` 执行合法性检查：
- 至少有 1 个 enabled Agent（否则抛出 `ConfigurationError`）
- 引用的 `prompt_template` 文件路径若不为空，则对应文件必须存在
- `timeout_seconds` 若设置，必须为正整数

**FR-012**：系统应提供 CLI 命令 `arena agent list`，
输出当前已注册的 Agent 列表（名称、类型、模型、状态、绑定 Skill），
帮助运营者快速确认 Registry 的实际状态。

**FR-013**：现有 4 个内置 Agent 作为默认注册项保留，
在 `default.toml` 中没有任何 `[agents.*]` 段落时，
AgentRegistry 自动使用内置 Agent 的默认配置，保持与当前系统完全一致的行为。

**FR-014**：`SkillLoader` 的 skill 文件搜索路径遵循以下优先级（高→低）：
1. 项目根目录的 `skills/` 目录
2. `~/.cryptotrader/skills/` 用户自定义目录
未找到时记录 `WARNING` 日志，该 Skill 被静默跳过，不影响 Agent 运行。

**FR-015**：当 AgentRegistry 接收到用户定义的全新 Agent（非内置 4 类）时，
默认将其视为 `BaseAgent` 类型（使用 `prompt_template` 作为 `role_description`，
无工具调用能力）；若 `tools` 字段不为空，则视为 `ToolAgent` 类型。

---

### 关键实体

**AgentConfig**：单个 Agent 的声明式配置，包含字段：
`agent_id`（名称）、`model`（LLM 模型，空=回退全局默认）、`timeout_seconds`（超时，0=回退全局默认）、
`enabled`（是否参与管道，默认 true）、`prompt_template`（外部 Markdown 路径，空=使用内置 ROLE）、
`tools`（可用工具名称列表，空=使用内置默认工具集）、`skills`（默认 Skill 文件名列表）、
`regime_skills`（Regime tag → Skill 列表映射）。

**AgentRegistry**：Agent 注册表，持有 `dict[str, AgentConfig]`，
提供 `get(agent_id)`、`list_active()`、`build(agent_id, backtest_mode)` 方法。
在 `load_config()` 时被构建，缓存于 `AppConfig.agents`。

**AgentsConfig**：`AppConfig` 中的顶层配置字段，包含所有 Agent 配置的集合，
同时持有 `AgentRegistry` 实例。

**SkillLoader**：负责从文件系统加载 Markdown skill 文件，
提供 `load(skill_name) -> str` 方法，内置文件搜索路径回退逻辑。

**SkillSelector**：负责根据 `regime_tags` 和 `AgentConfig.regime_skills` 映射
筛选并合并 Skill 内容，提供 `select(agent_cfg, regime_tags) -> list[str]` 方法，
遵循 token budget 上限约束。

**Skill 文件**：存放于 `skills/` 目录的 Markdown 文件（`.md`），
采用声明式格式描述分析规则和策略约束，不包含可执行代码。
文件名即为 Skill ID（不含扩展名），用于 TOML 中引用。

---

## 成功标准 *(必填)*

### 可度量的产出

**SC-001**：现有 742 条测试在此功能开发后全部通过，零回归。

**SC-002**：在 `config/local.toml` 中配置 `[agents.news_agent] model = "gpt-5" timeout_seconds = 30`，
系统端到端运行一次分析周期，日志中 NewsAgent 使用模型为 `gpt-5`，无额外代码改动。

**SC-003**：将任意一个内置 Agent 设为 `enabled = false` 后，系统单次分析成功完成，
`state["data"]["analyses"]` 中不包含该 Agent 的键名。

**SC-004**：在 `skills/` 目录放置一个 `.md` 文件并在 TOML 中引用，
Agent 提示词中包含该文件内容（可通过捕获 `ainvoke` 调用参数验证）。

**SC-005**：向 Registry 注册一个非内置的新 Agent，一次分析周期中该 Agent 的分析结果
出现在 `state["data"]["analyses"]` 中，管道流转至裁决步骤无报错。

**SC-006**：AgentConfig 级别超时比全局 `timeout_seconds` 短 50% 的情况下，
Mock LLM 挂起超过 Agent 级别超时时，该 Agent 降级返回 mock 结果，
其他 Agent 在全局超时内正常返回。

**SC-007**：`arena agent list` 命令输出格式化 Agent 列表，包含名称、模型、状态字段，
无运行时异常。

**SC-008**：引用不存在的 `prompt_template` 文件时，`load_config()` 抛出 `ConfigurationError`，
不进入 Agent 构建阶段。

**SC-009**：新增功能不引入任何 `noqa` 注释，`ruff check` 零 lint 错误。

**SC-010**：覆盖内置 TechAgent 的 `prompt_template` 后，捕获的 LLM 调用
SystemMessage 内容来自外部 Markdown 文件，而非代码内置的 `ROLE` 常量字符串。

---

## 假设

- Skill 文件是只读的声明式 Markdown，不包含可执行 Python 代码，不支持模板变量替换；
  若需要动态内容，Agent 的 `_build_prompt()` 方法是正确的扩展点，而非 Skill 文件。
- `AgentRegistry.build()` 构造的 Agent 实例不会被缓存跨周期复用；
  每次分析周期都重新从 Registry 构建 Agent 实例（保持与当前 `_run_agent()` 实例化模式一致）。
- 辩论（Debate）和裁决（Verdict）步骤不感知 Agent 数量的变化；
  它们消费 `state["data"]["analyses"]` 字典，键名数量动态变化对下游无副作用。
- 新增 Agent 不会引入新的数据类型（`DataSnapshot` 结构不变），
  自定义 Agent 通过 `_build_prompt()` 访问 snapshot 字段，与内置 Agent 一致。
- 现有 `config/default.toml` 中不添加任何 `[agents.*]` 段落，
  内置 Agent 的默认配置由 `AgentRegistry` 代码内构造，保持向后兼容。
- `prompt_template` 字段的文件路径相对于项目根目录解析；
  绝对路径也被支持（由 `Path` 对象处理）。
- 回测模式（`backtest_mode = True`）下，Registry 正常生效，
  ToolAgent 类型的 Agent 自动跳过工具调用（复用现有 `backtest_mode` 逻辑），无需额外处理。
- Skill Markdown 文件内容通过 `sanitize_input()` 净化后注入提示词，
  净化失败不阻塞 Agent 运行（返回空字符串并记录 WARNING）。

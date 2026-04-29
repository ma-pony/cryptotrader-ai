# 实施任务清单：Agent 可配置化 — Registry + Skills

> 格式说明：
> - `[P]` = 可与同 Phase 内其他 `[P]` 任务并行执行
> - `[S]` = 必须串行执行（依赖前序任务完成）
> - 任务 ID 格式：`T001`–`T080`（按 Phase 顺序编号）

---

## Phase 1：AgentConfig / AgentsConfig dataclass + TOML 解析

> 目标：`config.py` 中完整的配置数据模型，可通过 `load_config().agents` 访问。

- [x] T001 [S] 在 `src/cryptotrader/config.py` 中新增 `AgentConfig` dataclass，包含字段：`agent_id`、`model`（默认 `""`）、`timeout_seconds`（默认 `0`）、`enabled`（默认 `True`）、`prompt_template`（默认 `""`）、`tools: list[str]`（默认空列表）、`skills: list[str]`（默认空列表）、`regime_skills: dict[str, list[str]]`（默认空字典）
- [x] T002 [S] 在 `src/cryptotrader/config.py` 中新增 `AgentsConfig` dataclass，持有 `_agents: dict[str, AgentConfig]`，实现 `get(agent_id) -> AgentConfig | None`、`list_active() -> list[AgentConfig]`（含内置 Agent 默认配置的回退逻辑）
- [x] T004 [S] 在 `src/cryptotrader/config.py` 的 `AppConfig` dataclass 中新增 `agents: AgentsConfig = field(default_factory=AgentsConfig)` 字段
- [x] T003 [S] 扩展 `src/cryptotrader/config.py` 中的 `_build_config()` 函数，解析 TOML `data` 中的 `agents` 段落（`[agents.<agent_id>]` 形式），构建 `AgentsConfig` 实例并赋值给 `AppConfig.agents`
- [x] T005 [S] 扩展 `src/cryptotrader/config.py` 的 `validate_config()` 函数，增加 `AgentsConfig` 合法性校验：① `list_active()` 至少有 1 个 Agent（否则抛出 `ConfigurationError`，`field_path="agents"`）；② 非空 `prompt_template` 对应文件必须存在（`field_path="agents.<id>.prompt_template"`）；③ `timeout_seconds` 若非零则必须为正整数
- [x] T006 [P] 编写 `tests/test_agent_config.py`：测试 `AgentConfig` 默认值、`AgentsConfig.get()`、`AgentsConfig.list_active()` 在空配置时返回 4 个内置 Agent、TOML 段落解析正确填充字段、`validate_config()` 检查 enabled=0 时抛出 `ConfigurationError`、`prompt_template` 文件不存在时抛出 `ConfigurationError`

---

## Phase 2：AgentsConfig.build() — Agent 实例化注册表

> 目标：`AgentsConfig.build(agent_id, backtest_mode, regime_tags)` 能动态构造所有 Agent 类型。

- [x] T010 [S] 在 `src/cryptotrader/config.py` 的 `AgentsConfig` 中实现 `build(agent_id, backtest_mode, regime_tags)` 方法骨架：先查 `_agents.get(agent_id)`，不存在且为内置 4 类时构造默认 `AgentConfig`，否则抛出 `AgentNotFoundError`
- [x] T011 [S] 在 `src/cryptotrader/config.py` 中定义 `AgentNotFoundError(KeyError)` 异常类，错误信息包含 `agent_id` 和已注册 Agent 名称列表
- [x] T012 [S] 在 `AgentsConfig.build()` 中实现内置 Agent 类型分发：根据 `agent_id` 匹配 `tech_agent`/`chain_agent`/`news_agent`/`macro_agent`，实例化对应子类（`TechAgent`、`ChainAgent`、`NewsAgent`、`MacroAgent`），传入 `AgentConfig.model`（空时由子类/`create_llm()` 回退）和 `backtest_mode`
- [x] T013 [S] 在 `AgentsConfig.build()` 中实现自定义 Agent 类型分发：`tools` 为空时实例化 `BaseAgent`（`agent_id` 和从文件加载的 `role_description`），`tools` 非空时实例化 `ToolAgent`（附带工具过滤结果）；`prompt_template` 为空时沿用内置 `ROLE` 常量
- [x] T014 [S] 在 `AgentsConfig.build()` 中实现 `prompt_template` 文件加载：`Path(prompt_template)` 解析（相对路径基于项目根目录），读取文件内容作为 `role_description`；文件不存在时抛出 `ConfigurationError`（此处作为运行时二次保障，`validate_config` 为首道防线）
- [x] T015 [P] 编写 `tests/test_agent_registry.py`：测试 `build("tech_agent")` 返回 `TechAgent` 实例、`build("whale_agent")` 使用 `prompt_template` 返回 `BaseAgent`、`build("unknown_agent")` 抛出 `AgentNotFoundError`、`list_active()` 在 `enabled=false` 时排除对应 Agent、自定义 `model` 被正确传入

---

## Phase 3：nodes/agents.py 动态执行路径

> 目标：`_run_agent()` 从 `AgentsConfig` 动态查找并构造 Agent，超时使用 Agent 级别配置。

- [x] T020 [S] 重写 `src/cryptotrader/nodes/agents.py` 中 `_run_agent()` 的 Agent 实例化段落：移除硬编码的 `agents: dict` 字典，改为调用 `load_config().agents.build(agent_type, backtest_mode, regime_tags)`；保留现有 snapshot hash 复用逻辑不变
- [x] T021 [S] 在 `_run_agent()` 中修改超时取值逻辑：优先使用 `AgentConfig.timeout_seconds`（非零时）；为零时回退到 `load_config().models.timeout_seconds`（现有行为）
- [x] T022 [S] 修改 `src/cryptotrader/nodes/agents.py` 中 `_run_agent()` 的模型解析逻辑：`AgentConfig.model` 非空时优先使用，为空时回退到 `models_cfg.get(agent_type)`（现有路径），再回退到 `state["metadata"]["analysis_model"]`
- [x] T023 [S] 保留 `tech_analyze`、`chain_analyze`、`news_analyze`、`macro_analyze` 4 个 `@node_logger()` 函数，确保其内部仍调用 `_run_agent("tech_agent")`/`"chain_agent"` 等（字符串 ID 形式），与 LangGraph 图结构解耦
- [x] T024 [P] 编写 `tests/test_agent_timeout.py`：Mock `BaseAgent.analyze()` 永久挂起，配置 `[agents.chain_agent] timeout_seconds = 0.01`，验证 `_run_agent()` 返回 `is_mock=True` 结果且不传播异常；验证未设置 `timeout_seconds` 时使用全局 `models.timeout_seconds`

---

## Phase 4：SkillLoader — 文件系统 Skill 加载

> 目标：从文件系统加载 Skill Markdown 文件，支持双目录搜索路径优先级。

- [x] T030 [S] 新建 `src/cryptotrader/agents/skill_loader.py`，实现 `SkillLoader` 类：`__init__(search_paths: list[Path] | None = None)` 默认搜索路径为 `[PROJECT_ROOT / "skills", Path.home() / ".cryptotrader" / "skills"]`；`_find(skill_name) -> Path | None` 按序在各路径查找 `{skill_name}.md`
- [x] T031 [S] 在 `SkillLoader` 中实现 `load(skill_name) -> str`：找到文件后读取内容，调用 `sanitize_input()` 净化（`max_chars=8000`）；未找到时记录 `logger.warning("skill '%s' not found in search paths", skill_name)` 并返回 `""`；净化抛出异常时记录 WARNING 并返回 `""`
- [x] T032 [S] 在项目根目录创建 `skills/` 目录（作为 Skill 文件的默认存放位置）
- [x] T033 [P] 编写 `tests/test_skill_loader.py`：使用 `tmp_path` 作为自定义 `search_paths`，测试：`.md` 文件正确加载、内容经 `sanitize_input()` 净化、文件不存在时返回 `""` 并记录 WARNING、用户家目录路径作为第二搜索路径的优先级低于项目目录

---

## Phase 5：SkillSelector — Regime 匹配与 Token Budget

> 目标：`SkillSelector.select()` 根据 regime_tags 自动合并 Skill 内容，遵守 token budget。

- [x] T040 [S] 新建 `src/cryptotrader/agents/skill_selector.py`，实现 `SkillSelector` 类：`select(agent_cfg, regime_tags, loader, token_budget_chars) -> list[str]`
- [x] T041 [S] 在 `SkillSelector.select()` 中实现默认 Skill 加载：遍历 `agent_cfg.skills`，调用 `loader.load(name)`，过滤掉空字符串
- [x] T042 [S] 在 `SkillSelector.select()` 中实现 Regime 匹配：遍历 `agent_cfg.regime_skills` 的键（regime_tag），若该 tag 在 `regime_tags` 中，则将对应 Skill 名列表追加到待加载列表（去重：已在默认 Skill 中的跳过）
- [x] T043 [S] 在 `SkillSelector.select()` 中实现 token budget 截断：累计各 Skill 内容字符数，超过 `token_budget_chars` 时截断当前 Skill 并记录 `logger.warning("skill budget exceeded, truncating after %d chars", token_budget_chars)`，后续 Skill 跳过
- [x] T044 [S] 在 `AgentsConfig.build()` 中集成 `SkillSelector`：构造 Agent 实例后，创建 `SkillLoader()` 和 `SkillSelector()`，调用 `select(agent_cfg, regime_tags or [], loader, budget)`，将结果拼接为 `skill_content` 字符串传入 Agent 构造
- [x] T045 [S] 修改 `src/cryptotrader/agents/base.py` 的 `BaseAgent.__init__`，新增 `skill_content: str = ""` 参数，`_build_prompt()` 在 `role_description` 与 `ANALYSIS_FRAMEWORK` 之间插入 `skill_content`（若非空，以 `\n\n--- STRATEGY SKILLS ---\n` 为分隔头）
- [x] T046 [P] 编写 `tests/test_skill_selector.py`：测试默认 Skill 加载、Regime tag 匹配追加额外 Skill、无 tag 匹配时仅加载默认 Skill、多 tag 同时匹配时合并去重、token budget 截断行为、空配置时返回空列表

---

## Phase 6：工具隔离（Tool Isolation）

> 目标：`AgentConfig.tools` 非空时，ToolAgent 只能访问声明的工具子集。

- [x] T050 [S] 修改 `src/cryptotrader/agents/data_tools.py`，新增 `ALL_TOOLS: list` 导出，合并 `CHAIN_TOOLS`、`NEWS_TOOLS`（及其他内置工具列表），作为工具过滤的全量来源
- [x] T051 [S] 在 `AgentsConfig.build()` 的工具分发逻辑中实现过滤：`AgentConfig.tools` 为空时传入 Agent 类型默认工具集；非空时从 `ALL_TOOLS` 按 `tool.name` 属性过滤，保留声明列表中的工具；未识别工具名记录 `logger.warning("unknown tool '%s' in agents.%s.tools", tool_name, agent_id)`
- [x] T052 [P] 编写 `tests/test_tool_isolation.py`：配置 `tools = ["get_funding_rate"]` 的 `AgentConfig`，验证 `build()` 返回的 `ToolAgent.tools` 长度为 1；`tools = []` 时返回完整默认工具集；未识别工具名时记录 WARNING 但不抛出异常

---

## Phase 7：CLI 命令 `arena agent list`

> 目标：`arena agent list` 输出当前注册 Agent 的格式化列表。

- [x] T060 [S] 在 `src/cli/main.py` 中新增 `agent` 命令组（`@app.command("agent")` 或 Typer group），并在其下新增 `list` 子命令
- [x] T061 [S] 实现 `arena agent list` 输出逻辑：调用 `load_config().agents.list_active()`，以表格或列表形式输出每个 Agent 的：名称、类型（内置/自定义）、模型（空时显示 `"<default>"`）、状态（enabled/disabled）、绑定 Skill 数量；使用 `rich.table.Table` 或简单 print 对齐
- [x] T062 [P] 编写 `tests/test_cli_agent_list.py`：使用 `typer.testing.CliRunner` 调用 `arena agent list`，验证输出包含 4 个内置 Agent 名称、状态字段、无运行时异常；配置自定义 Agent 后验证其出现在列表中

---

## Phase 8：集成测试

> 目标：端到端验证所有成功标准（SC-001～SC-010）。

- [x] T070 [P] 编写 `tests/test_agent_integration.py` — SC-002/SC-003：Mock LLM 调用，通过 `AppConfig` 注入含 `[agents.news_agent] model = "gpt-5" timeout_seconds = 30` 的配置，验证 `_run_agent("news_agent", state)` 构造的 Agent `model == "gpt-5"`；设置 `enabled = false` 后验证 `state["data"]["analyses"]` 中不含该 Agent 键名
- [x] T071 [P] 编写集成测试 — SC-004/SC-010：在 `tmp_path/skills/` 中放置 `momentum_strategy.md`，配置 `AgentConfig(skills=["momentum_strategy"])`，捕获 `BaseAgent.analyze()` 的 `ainvoke` 调用参数，验证 SystemMessage 内容包含 Skill 文件内容；覆盖 `prompt_template` 后验证 SystemMessage 使用外部文件内容而非 `ROLE` 常量
- [x] T072 [P] 编写集成测试 — SC-005：注册 `whale_agent` 的 `AgentConfig`（含 `prompt_template` 和 `tools`），调用 `AgentsConfig.build("whale_agent")`，验证返回 `ToolAgent` 实例；在 Mock 状态下运行分析，验证结果出现在 `analyses` 字典中
- [x] T073 [P] 编写集成测试 — SC-006：配置 Agent 级别 `timeout_seconds` 为全局的 50%，Mock `analyze()` 挂起超过 Agent 级别超时但小于全局超时，验证该 Agent 降级而其他 Agent 正常返回
- [x] T074 [S] 运行完整测试套件，验证现有 742 条测试零回归（SC-001）；确认 `ruff check src/` 零 lint 错误（SC-009）

---

## 任务依赖概览

```
Phase 1 (T001-T006)
  → T001 → T002 → T004 → T003 → T005
  → T006 [可与 T001-T005 并行]

Phase 2 (T010-T015)
  → [依赖 Phase 1 完成]
  → T010 → T011 → T012 → T013 → T014
  → T015 [可与 T010-T014 并行]

Phase 3 (T020-T024)
  → [依赖 Phase 2 完成]
  → T020 → T021 → T022 → T023
  → T024 [可与 T020-T023 并行]

Phase 4 (T030-T033)
  → [可与 Phase 3 并行开始]
  → T030 → T031 → T032
  → T033 [可与 T030-T032 并行]

Phase 5 (T040-T046)
  → [依赖 Phase 4 完成]
  → T040 → T041 → T042 → T043 → T044 → T045
  → T046 [可与 T040-T045 并行]

Phase 6 (T050-T052)
  → [依赖 Phase 2 完成]
  → T050 → T051
  → T052 [可与 T050-T051 并行]

Phase 7 (T060-T062)
  → [依赖 Phase 1 完成]
  → T060 → T061
  → T062 [可与 T060-T061 并行]

Phase 8 (T070-T074)
  → [依赖 Phase 1-7 全部完成]
  → T070, T071, T072, T073 [可并行]
  → T074 [依赖 T070-T073 完成]
```

---

## 成功标准验收矩阵

| 成功标准 | 关键任务 | 验证方式 |
|----------|----------|----------|
| SC-001：742 条测试零回归 | T074 | `pytest` 全量运行 |
| SC-002：TOML 配置模型端到端生效 | T003、T020、T022 | `test_agent_integration.py` T070 |
| SC-003：`enabled=false` 不进入管道 | T002、T020 | `test_agent_integration.py` T070 |
| SC-004：Skill 文件内容注入提示词 | T031、T045、T044 | `test_agent_integration.py` T071 |
| SC-005：自定义 Agent 纳入分析管道 | T013、T020 | `test_agent_integration.py` T072 |
| SC-006：Agent 级别超时降级 | T021 | `test_agent_timeout.py` T024 / T073 |
| SC-007：`arena agent list` 正常输出 | T061 | `test_cli_agent_list.py` T062 |
| SC-008：`prompt_template` 不存在抛出 `ConfigurationError` | T005、T014 | `test_agent_config.py` T006 |
| SC-009：零 lint 错误 | 全部任务 | `ruff check src/` T074 |
| SC-010：自定义 `prompt_template` 替换 `ROLE` 常量 | T014、T045 | `test_agent_integration.py` T071 |

# 技术实施方案：Agent 可配置化 — Registry + Skills

## 技术上下文

### 当前 Agent 实例化分析

4 个内置 Agent 的实例化完全硬编码于 `src/cryptotrader/nodes/agents.py` 的 `_run_agent()` 函数：

```python
agents: dict[str, Any] = {
    "tech_agent":  lambda m: TechAgent(model=m),
    "chain_agent": lambda m: ChainAgent(model=m, backtest_mode=backtest_mode),
    "news_agent":  lambda m: NewsAgent(model=m, backtest_mode=backtest_mode),
    "macro_agent": lambda m: MacroAgent(model=m),
}
model = models_cfg.get(agent_type, state["metadata"].get("analysis_model", ""))
agent = agents[agent_type](model)
```

- Agent 类型枚举固定为 4 项，无法增删
- 超时统一取 `load_config().models.timeout_seconds`（全局共享）
- `_run_agent()` 被 4 个独立的 `@node_logger()` 函数（`tech_analyze`、`chain_analyze` 等）调用，函数名称与 Agent ID 一一对应且硬编码
- 并行执行由 `graph.py` 中的 `asyncio.gather()` 在图层面完成，节点名与 Agent ID 耦合

### BaseAgent / ToolAgent 类型体系

`src/cryptotrader/agents/base.py` 定义两个基类：

- **`BaseAgent`**：单次 LLM 调用，`analyze(snapshot, experience) -> AgentAnalysis`，`role_description` 来自代码常量 `ROLE`
- **`ToolAgent(BaseAgent)`**：工具调用循环（`create_agent`），`tools` 字段持有 LangChain 工具列表，`backtest_mode=True` 时退化为 `BaseAgent` 单次调用

各 Agent 子类（`TechAgent`、`ChainAgent` 等）的 `ROLE` 常量硬编码在各自模块顶层，构造时传入 `super().__init__(agent_id, ROLE, model)`。

### 现有 Skills 实现

`src/cryptotrader/agents/skills.py` 中 `TRADING_SKILLS` 是硬编码的 `list[Skill]`（TypedDict），内容以 Python 字符串形式写在源码中。`get_skill_descriptions()` 和 `load_skill_content()` 函数供 Agent 按需调用。**没有任何文件系统加载逻辑，也没有 Regime 感知的匹配机制。**

### 现有配置结构

`ModelConfig` 中有 `tech_agent`、`chain_agent`、`news_agent`、`macro_agent` 4 个字段，仅保存模型名称字符串。`AppConfig` 中没有任何结构化的 Agent 声明配置（`[agents.*]` 段落）。`config/default.toml` 中不包含 `[agents]` 段落。

### 现有 Regime 系统

`src/cryptotrader/learning/regime.py` 的 `tag_regime()` 根据 `snapshot_summary` 数据生成离散标签列表（如 `"high_funding"`、`"trending_up"`、`"extreme_fear"` 等），标签已稳定注入 `state["data"]["regime_tags"]`，可直接被 `SkillSelector` 消费。

### 现有 `sanitize_input()`

`src/cryptotrader/security.py` 导出 `sanitize_input(text, max_chars)` 函数，已在 `BaseAgent._build_prompt()` 中对外部新闻标题和经验记忆内容执行净化。Skill 文件内容属于同类外部输入，应走同一净化路径。

### 现有 `ConfigurationError`

`src/cryptotrader/config.py` 已定义 `ConfigurationError(ValueError)` 及 `validate_config(cfg)`，本特性直接复用并扩展此验证入口。

---

## 架构决策

### 决策 1：AgentConfig / AgentsConfig 作为纯 dataclass，嵌入 AppConfig

**选择**：在 `src/cryptotrader/config.py` 中新增 `AgentConfig` 和 `AgentsConfig` 两个 dataclass。`AppConfig` 新增 `agents: AgentsConfig` 字段（`field(default_factory=AgentsConfig)`）。`_build_config()` 在解析 TOML 时，若存在 `[agents.*]` 段落则填充 `AgentsConfig`；若不存在则保持空映射，AgentRegistry 自动回退到 4 个内置 Agent 的默认配置。

**理由**：与现有 `RiskConfig`、`BacktestConfig` 等的模式完全一致，无需引入额外机制。`AppConfig` 是单一配置入口，`load_config()` 已缓存，下游只需 `load_config().agents` 即可访问。

### 决策 2：AgentRegistry 作为 AgentsConfig 的方法，而非独立类

**选择**：`AgentsConfig` 直接持有 `dict[str, AgentConfig]` 映射，并实现 `get(agent_id)`、`list_active()`、`build(agent_id, backtest_mode)` 方法。不引入独立的 `AgentRegistry` 类文件。

**理由**：避免过度抽象。`AgentsConfig` 本身已是"注册表"的载体，方法内聚在同一 dataclass 上符合现有代码风格（参见 `ExchangesConfig.get()`）。`build()` 方法返回 `BaseAgent` 实例，构建逻辑集中于此处。

### 决策 3：`_run_agent()` 动态化，保留 4 个 node 函数但均委托给统一路径

**选择**：重写 `_run_agent(agent_type, state)` 使其从 `load_config().agents` 查找 `AgentConfig`；若找不到且 `agent_type` 是内置 4 类之一，则自动构造默认配置并执行（向后兼容）。4 个现有 `@node_logger()` 函数（`tech_analyze` 等）保留，内部改为调用统一的 `_run_agent()`。新增动态 Agent 可通过图扩展点（`build_trading_graph()` 的参数）注入，本特性不修改 LangGraph 图定义。

**理由**：图节点名称（`tech_analyze` 等）与 LangGraph 图结构绑定，修改成本高。动态 Agent 的并行执行放在 tasks.md T030 阶段讨论，本特性核心是配置+实例化层面的动态化，不改变图拓扑。

### 决策 4：`prompt_template` 文件路径相对于项目根目录解析，绝对路径直接支持

**选择**：`AgentsConfig.build()` 在加载 `prompt_template` 时，使用 `Path(prompt_template)`：若为相对路径则相对于 `_project_root()`（`config/default.toml` 所在目录的父目录）解析；若为绝对路径则直接使用。文件不存在时在 `validate_config()` 阶段即抛出 `ConfigurationError`，不延迟到运行时。

**理由**：与 Spec FR-011 和用户故事 2 的验收场景保持一致。早期验证（`validate_config` 而非 `build`）是 Fail-Fast 原则的体现。

### 决策 5：SkillLoader 独立模块，搜索路径优先级固定为项目 → 用户家目录

**选择**：新建 `src/cryptotrader/agents/skill_loader.py`，提供 `SkillLoader` 类，持有 `search_paths: list[Path]`（默认为 `[PROJECT_ROOT/skills/, ~/.cryptotrader/skills/]`），`load(skill_name) -> str` 方法按优先级搜索 `{skill_name}.md` 文件，找不到时记录 `WARNING` 并返回空字符串。

**理由**：将文件系统 I/O 与 Agent 实例化逻辑解耦，便于单元测试（可传入临时目录作为 `search_paths`）。`SkillLoader` 是无状态类，每次实例化开销可忽略。

### 决策 6：SkillSelector 合并 Regime 匹配与 token budget 截断逻辑

**选择**：新建 `src/cryptotrader/agents/skill_selector.py`，提供 `SkillSelector` 类，`select(agent_cfg, regime_tags, loader, token_budget_chars) -> list[str]` 方法按以下逻辑执行：① 加载 `agent_cfg.skills` 中的默认 Skill；② 根据 `regime_tags` 匹配 `agent_cfg.regime_skills` 中的额外 Skill；③ 去重合并；④ 若总字符数超过 `token_budget_chars`（由 `experience.token_budget_pct` 推导），按声明顺序截断并记录 `WARNING`。

**理由**：`SkillSelector` 承载策略选择逻辑，`SkillLoader` 承载文件 I/O，单一职责。`token_budget_chars` 从 `experience.token_budget_pct` 推导（`4096 * token_budget_pct`），复用现有配置字段，无需新增。

### 决策 7：工具隔离（Tool Isolation）在 `AgentsConfig.build()` 中实现，全量工具表来自 `data_tools.py`

**选择**：`AgentConfig.tools` 为空列表时，`build()` 传入 Agent 类型对应的默认工具列表（`CHAIN_TOOLS`、`NEWS_TOOLS` 等）；不为空时，从 `ALL_TOOLS`（`data_tools.py` 中新导出的合并工具集）按名称过滤，只注入声明的工具。未识别的工具名记录 `WARNING`，不抛出异常（避免配置错误阻塞启动）。

**理由**：工具过滤逻辑放在实例化层，不影响 `BaseAgent.analyze()` 路径。`ALL_TOOLS` 合并导出避免调用方枚举多个工具列表。

### 决策 8：向后兼容 ModelConfig 字段，AgentConfig.model 优先级更高

**选择**：`ModelConfig` 的 `tech_agent`、`chain_agent`、`news_agent`、`macro_agent` 字段继续保留，不做任何修改。在 `_run_agent()` 中，模型解析顺序为：`AgentConfig.model`（若非空）→ `models_cfg.get(agent_type)`（现有路径）→ `state["metadata"]["analysis_model"]`（回退）。这样不设置 `[agents.*]` 时行为完全不变。

**理由**：Spec FR-010 和 FR-013 明确要求向后兼容。`ModelConfig` 的按 Agent 模型字段是现有用户的配置习惯，不应强制迁移。

---

## 文件结构（新增 / 修改）

```
config/
  default.toml                       # 不修改（无 [agents.*] 段落 = 向后兼容）

skills/                              # 新增：Skill Markdown 文件根目录
  README.md                          # 说明文件（可选，非功能性）

src/cryptotrader/
  config.py                          # 修改：新增 AgentConfig、AgentsConfig dataclass
                                     #        新增 AgentNotFoundError 异常类
                                     #        AppConfig 新增 agents 字段
                                     #        validate_config() 扩展 AgentsConfig 检查
                                     #        _build_config() 解析 [agents.*] TOML 段落

  agents/
    base.py                          # 修改：BaseAgent.__init__ 接受可选 skill_content 参数
                                     #        _build_prompt() 在 role_description 后注入 Skill 内容
    skill_loader.py                  # 新增：SkillLoader 类（文件搜索、加载、sanitize）
    skill_selector.py                # 新增：SkillSelector 类（Regime 匹配、token budget 截断）
    data_tools.py                    # 修改：导出 ALL_TOOLS 合并工具集

  nodes/
    agents.py                        # 修改：_run_agent() 动态从 AgentsConfig.build() 实例化
                                     #        超时取 AgentConfig.timeout_seconds（若非零）
                                     #        list_active() 驱动并行执行逻辑

src/cli/
    main.py                          # 修改：新增 `arena agent list` 子命令

tests/
  test_agent_config.py               # 新增：AgentConfig / AgentsConfig dataclass 解析测试
  test_agent_registry.py             # 新增：AgentsConfig.build() / list_active() 测试
  test_skill_loader.py               # 新增：SkillLoader 搜索路径、sanitize、不存在跳过
  test_skill_selector.py             # 新增：Regime 匹配、token budget 截断、空匹配回退
  test_agent_timeout.py              # 新增：Agent 级别超时覆盖全局超时
  test_tool_isolation.py             # 新增：工具子集过滤
  test_cli_agent_list.py             # 新增：arena agent list 输出格式
  test_agent_integration.py         # 新增：端到端 AgentConfig → 分析管道集成测试
```

---

## 数据模型

### `AgentConfig` dataclass（`config.py`）

```python
@dataclass
class AgentConfig:
    agent_id: str
    model: str = ""                          # 空 = 回退 ModelConfig 或全局默认
    timeout_seconds: int = 0                 # 0 = 回退 models.timeout_seconds
    enabled: bool = True
    prompt_template: str = ""                # 空 = 使用内置 ROLE 常量
    tools: list[str] = field(default_factory=list)   # 空 = 使用内置默认工具集
    skills: list[str] = field(default_factory=list)  # 默认 Skill 文件名（不含扩展名）
    regime_skills: dict[str, list[str]] = field(default_factory=dict)  # regime_tag -> [skill_names]
```

### `AgentsConfig` dataclass（`config.py`）

```python
@dataclass
class AgentsConfig:
    _agents: dict[str, AgentConfig] = field(default_factory=dict)

    def get(self, agent_id: str) -> AgentConfig | None:
        """按 ID 查询 AgentConfig，不存在返回 None。"""

    def list_active(self) -> list[AgentConfig]:
        """返回所有 enabled=True 的 AgentConfig，含内置 Agent 默认配置。"""

    def build(
        self,
        agent_id: str,
        backtest_mode: bool = False,
        regime_tags: list[str] | None = None,
    ) -> BaseAgent:
        """根据 AgentConfig 构造 Agent 实例。
        - 内置类型（tech/chain/news/macro）：实例化对应子类
        - 自定义类型（tools 为空）：实例化 BaseAgent（prompt_template 作为 role_description）
        - 自定义类型（tools 非空）：实例化 ToolAgent
        - prompt_template 非空时，从文件读取内容替代 ROLE 常量
        - 注入 SkillSelector 选出的 Skill 内容
        """
```

### `SkillLoader` 类（`agents/skill_loader.py`）

```python
class SkillLoader:
    def __init__(self, search_paths: list[Path] | None = None) -> None:
        # 默认: [PROJECT_ROOT/skills/, Path.home() / ".cryptotrader/skills/"]

    def load(self, skill_name: str) -> str:
        """按优先级搜索 {skill_name}.md，sanitize_input() 净化后返回。
        未找到时记录 WARNING，返回空字符串。净化失败返回空字符串并记录 WARNING。
        """
```

### `SkillSelector` 类（`agents/skill_selector.py`）

```python
class SkillSelector:
    def select(
        self,
        agent_cfg: AgentConfig,
        regime_tags: list[str],
        loader: SkillLoader,
        token_budget_chars: int = 4000,
    ) -> list[str]:
        """
        返回净化后的 Skill 内容列表，按以下顺序构建：
        1. agent_cfg.skills 中的默认 Skill（顺序保留）
        2. 与 regime_tags 匹配的 regime_skills 中的 Skill（去重）
        3. 合并后若超过 token_budget_chars，按声明顺序截断，记录 WARNING
        """
```

### `BaseAgent._build_prompt()` 扩展

在现有 `role_description + ANALYSIS_FRAMEWORK` 结构中，Skill 内容注入位置：

```
SystemMessage:
  [role_description]           ← 保持不变（ROLE 常量 或 prompt_template 文件内容）
  [skill_content_block]        ← 新增：SkillSelector.select() 结果，以 \n\n 分隔各 Skill
  [ANALYSIS_FRAMEWORK]         ← 保持不变
```

Skill 内容通过 `BaseAgent.__init__` 的 `skill_content: str = ""` 参数注入，`AgentsConfig.build()` 在构造 Agent 后调用 `SkillSelector` 并将结果传入。

---

## TOML 配置示例（`config/local.toml`）

```toml
# 覆盖内置 NewsAgent 的模型和超时
[agents.news_agent]
model = "gpt-5"
timeout_seconds = 45
enabled = true

# 使用自定义 prompt_template 覆盖 TechAgent
[agents.tech_agent]
prompt_template = "prompts/my_tech_prompt.md"
skills = ["momentum_strategy"]

# 配置 MacroAgent 的 Regime 感知 Skill
[agents.macro_agent.regime_skills]
trending_up = ["bull_momentum_skill"]
high_vol = ["volatility_playbook"]
extreme_fear = ["fear_accumulation_skill"]

# 添加全新的自定义 Agent（非内置）
[agents.whale_agent]
prompt_template = "prompts/whale_prompt.md"
tools = ["get_whale_transfers", "get_exchange_flows"]
timeout_seconds = 60
```

---

## 向后兼容性保证

| 场景 | 保证 |
|------|------|
| `default.toml` 中无 `[agents.*]` 段落 | `AgentsConfig._agents` 为空字典，`list_active()` 返回 4 个内置 Agent 的默认 `AgentConfig`，行为与现有系统完全一致 |
| `ModelConfig.tech_agent` 等字段保留 | `_run_agent()` 在 `AgentConfig.model == ""` 时仍回退到 `models_cfg.get(agent_type)` |
| 现有 4 个 `@node_logger()` node 函数保留 | 函数签名和名称不变，内部委托给统一的 `_run_agent()` 路径 |
| `TRADING_SKILLS` 硬编码列表保留 | `agents/skills.py` 不删除，内置 Skill 作为补充；文件系统 Skill 是新增能力层 |
| `validate_config()` 现有检查项不变 | 新增 `AgentsConfig` 检查项追加在函数末尾，不影响现有验证逻辑 |
| 所有现有 742 条测试无需修改 | `AgentsConfig` 字段有 `default_factory`，`AppConfig` 实例化时无感知变化 |

---

## 依赖变更

无需新增外部依赖。所有实现仅使用：
- `pathlib.Path`（标准库）：文件搜索
- `tomllib`（标准库，Python 3.11+）：已在 `config.py` 中使用
- `src/cryptotrader/security.sanitize_input()`：已有
- `src/cryptotrader/agents.data_tools`：已有，仅新增 `ALL_TOOLS` 导出

---

## 风险与缓解

| 风险 | 严重度 | 缓解措施 |
|------|--------|---------|
| 所有内置 Agent 被 `enabled=false` 禁用，管道进入空状态 | 高 | `validate_config()` 在启动时校验 `list_active()` 至少有 1 个 Agent，不满足则抛出 `ConfigurationError` |
| `prompt_template` 文件路径错误导致运行时崩溃 | 高 | `validate_config()` 阶段提前检查文件是否存在，失败时给出 `field_path="agents.<id>.prompt_template"` 的明确错误信息 |
| Skill Markdown 文件包含 Prompt Injection 内容 | 中 | `SkillLoader.load()` 对文件内容执行 `sanitize_input()`，净化失败时返回空字符串并记录 WARNING |
| Skill 内容超出 Agent 提示词 token 上限导致截断策略不一致 | 中 | `SkillSelector` 按字符数截断并记录 WARNING，`ANALYSIS_FRAMEWORK` 和 `role_description` 始终完整保留（它们不经过 `SkillSelector`） |
| 新增自定义 Agent 导致辩论门控共识计算基数变化 | 低 | `compute_consensus_strength()` 在 `debate/convergence.py` 中基于传入的分析结果动态计算，不依赖 Agent 数量硬编码，无需修改 |
| `[agents.*]` TOML 段落格式错误导致启动崩溃 | 中 | `_build_config()` 对 `[agents.*]` 段落的解析包裹在 `try/except`，格式错误抛出 `ConfigurationError` 含行号提示 |
| 工具名称拼写错误导致 Agent 工具集为空 | 低 | 未识别工具名记录 `WARNING` 但不抛出异常，Agent 仍以空工具列表运行（退化为 `BaseAgent` 单次调用模式） |
| `AgentsConfig.build()` 每次分析周期重新实例化 Agent（无缓存） | 低 | 与现有 `_run_agent()` 中直接实例化的模式一致，Agent 构造开销可忽略（不含 LLM 调用） |

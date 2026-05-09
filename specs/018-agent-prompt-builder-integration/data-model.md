# Phase 1：数据模型

**关联 spec**：[spec.md](spec.md) — Key Entities 段
**Date**: 2026-05-08

## 实体关系图

```
┌────────────────────────┐
│  AgentConfigFile       │ — 4 个文件（tech/chain/news/macro.md）
│  config/agents/<id>.md │   YAML frontmatter + Markdown body
│  (沿用 017a schema)    │   含 ANALYSIS_FRAMEWORK 拆段
└──────────┬─────────────┘
           │ 读取
           ▼
┌────────────────────────────────────────────┐
│  PromptBuilder（spec 017a 沿用 + 扩展）    │
│  + experience: str = "" 参数（FR-Y6b）      │
│  + _render_skills 完整 body（FR-Y29）       │
│  + _render_snapshot 调 snapshot_renderer    │
└──────────┬─────────────────────────────────┘
           │ 调用
           ▼
┌────────────────────────────────────────────┐
│  snapshot_renderer.render_crypto_snapshot()│ — NEW 模块
│  含 funding annotation / news 防注入 /     │
│  data warnings / sanitize_input            │
└────────────────────────────────────────────┘

┌─────────────────────────┐
│  DefaultSkillProvider   │ — 修正 scope filter
│  (017a 沿用 + bug fix)  │   discover_skills_for_agent(agent_id)
└──────────┬──────────────┘
           │ 读取
           ▼
agent_skills/<id>/SKILL.md（spec 014 既有协议）
   frontmatter scope: shared | agent:<id>

┌─────────────────────────┐
│  PromptBuilderSingleton │ — NEW，nodes/agents.py module-level
│  _memory_provider       │   lazy-init Provider 单例
│  _skill_provider        │
│  _prompt_builders[id]   │   per-agent_id 缓存
└─────────────────────────┘

BaseAgent / ToolAgent — 重构构造器
  __init__(*, agent_id, prompt_builder, model, [tools, backtest_mode])
  analyze() → 调 prompt_builder.build()
```

## 实体定义

### AgentConfigFile（沿用 spec 017a schema）

**职责**：单个 agent 的 prompt 配置载体

**位置**：`config/agents/<agent_id>.md`（4 个文件：tech / chain / news / macro）

**Schema**（沿用 017a `contracts/agent-config-schema.md`）：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `agent_id` | str | ✓ | 与文件名一致 |
| `description` | str | ✓ | 一句话描述 |
| `sections` | list[str] | ✓ | body 必须含的 section 名（5 个：system_prompt / user_tail / available_skills / recent_memory / output_schema）|
| `budget` | int | ✓ | token 预算 |
| `priority` | dict[str, int] | ✓ | section→优先级 |
| `slot_overrides` | dict | ✗ | 覆盖默认 slot 分配 |

**Body**：5 个 `## section_name` 段，本 spec 中：
- `system_prompt`：原 ROLE 字符串 + ANALYSIS_FRAMEWORK discipline 部分（30 行）
- `user_tail`：可短，如"请基于上述输入输出 JSON 决策"
- `available_skills`：占位（运行时由 SkillProvider 注入）
- `recent_memory`：占位（运行时由 MemoryProvider 或 experience 参数注入）
- `output_schema`：ANALYSIS_FRAMEWORK JSON schema 部分（5 行）

---

### PromptBuilder（spec 017a 沿用 + 本 spec 扩展）

**变更**：

1. **`build()` 签名扩展（FR-Y6b）**：
   ```python
   build(
       snapshot: dict,
       portfolio: dict,
       agent_analyses: dict | None = None,
       experience: str = "",  # NEW — spec 017b 加
   ) -> tuple[SystemMessage, HumanMessage]
   ```
   - `experience` 非空 → 直接作为 `recent_memory` section 内容
   - `experience` 空 → 调 `memory_provider.get_recent_memory(...)` fallback（沿用 017a 路径）

2. **`_render_skills()` 改为完整 body 渲染（FR-Y29）**：
   ```python
   def _render_skills(self, skills: list[Skill]) -> str:
       if not skills:
           return "暂无可用技能"
       parts = []
       for skill in skills:
           parts.append(f"\n\n---\n## Skill: {skill.name}\n\n{skill.body}")
       return "".join(parts)
   ```
   - 输出与 SkillsInjectionMiddleware 删除前等价

3. **`_render_snapshot()` 改为调 snapshot_renderer（FR-Y14）**：
   ```python
   def _render_snapshot(self, snapshot: dict) -> str:
       from cryptotrader.agents.snapshot_renderer import render_crypto_snapshot
       return render_crypto_snapshot(snapshot)
   ```

---

### DefaultSkillProvider（spec 017a 沿用 + 本 spec bug fix）

**变更**：

`get_available_skills()` 内部过滤逻辑（FR-Y28）：

```python
# OLD（017a 错误）：
relevant = [s for s in skills if agent_id in s.tags]

# NEW（spec 017b 修正）：
from cryptotrader.agents.skills.loader import discover_skills_for_agent
return discover_skills_for_agent(agent_id, skill_dir=self._root)[:k]
```

---

### Skill dataclass（spec 017a 沿用 + 本 spec 加字段）

**字段**：

| 字段 | 类型 | 说明 |
|---|---|---|
| `skill_id` | str | 唯一标识（沿用） |
| `description` | str | 一句话描述（沿用） |
| `tags` | list[str] | 关联（沿用，但本 spec 不再用作过滤） |
| `steps` | list[str] | 关键步骤（沿用） |
| `body` | str | 完整 SKILL.md body（沿用） |
| `name` | str | NEW（FR-Y30）— skill 显示名（如缺则 fallback 到 skill_id） |

---

### snapshot_renderer.py（NEW 模块）

**职责**：crypto 领域专用 snapshot 渲染

**核心函数**：

```python
def render_crypto_snapshot(
    snapshot: dict,
    experience: str = "",
) -> str:
    """渲染 snapshot dict 为 markdown 字符串。

    含 spec 014 / 015 落地的所有领域逻辑：
    - funding rate ELEVATED / NEGATIVE 标注
    - futures volume SPIKE / LOW 标注
    - news headlines 经 sanitize_input 防注入
    - data quality warnings（onchain / news / macro 缺失）
    - experience 字段经 sanitize_input(max_chars=4000)
    - TechAgent indicators 字段（snapshot 含 'indicators' key 时附加）
    """
    ...
```

**搬运来源**：`BaseAgent._build_prompt()` 整段（base.py:420-467）

**参数约束**：

| 参数 | 类型 | 说明 |
|---|---|---|
| `snapshot` | dict | 必填；含 pair / timestamp / market.* / news.* / onchain.* / macro.* / indicators (可选) |
| `experience` | str | 可选；非空时附加为 markdown 段（经 sanitize cap） |

**返回**：`str`（markdown 文本）

**安全保证**：
- 所有 external 内容（news headlines / experience）经 `sanitize_input()` 处理
- 内部 prompt（agent ROLE / ANALYSIS_FRAMEWORK）NOT 经过 sanitize（trusted）
- 沿用 spec 015 的防注入 invariant

---

### PromptBuilderSingleton（NEW，nodes/agents.py module-level）

**职责**：cycle 调用时为 4 agent 提供缓存的 PromptBuilder 实例

**实现**：

```python
# src/cryptotrader/nodes/agents.py 顶层
from pathlib import Path
from cryptotrader.agents.prompt_builder import (
    DefaultMemoryProvider, DefaultSkillProvider, PromptBuilder
)

_memory_provider: DefaultMemoryProvider | None = None
_skill_provider: DefaultSkillProvider | None = None
_prompt_builders: dict[str, PromptBuilder] = {}

def _get_or_build_pb(agent_id: str, model: str) -> PromptBuilder:
    global _memory_provider, _skill_provider
    if _memory_provider is None:
        _memory_provider = DefaultMemoryProvider(memory_root=Path("agent_memory"))
        _skill_provider = DefaultSkillProvider(skills_root=Path("agent_skills"))
    if agent_id not in _prompt_builders:
        _prompt_builders[agent_id] = PromptBuilder(
            agent_id=agent_id,
            config_dir=Path("config/agents"),
            memory_provider=_memory_provider,
            skill_provider=_skill_provider,
            model=model,
        )
    return _prompt_builders[agent_id]
```

**生命周期**：进程内 lazy-init，永不释放（PromptBuilder 是无状态对象）

---

### BaseAgent / ToolAgent（spec 014 沿用 + 本 spec 重构）

**重构后签名**：

```python
class BaseAgent:
    def __init__(self, *, agent_id: str, prompt_builder: PromptBuilder, model: str = "") -> None:
        self.agent_id = agent_id
        self._prompt_builder = prompt_builder
        self.model = model

    async def analyze(self, snapshot: DataSnapshot, experience: str = "") -> AgentAnalysis:
        try:
            sys_msg, usr_msg = self._prompt_builder.build(
                snapshot=self._snapshot_to_dict(snapshot),
                portfolio={},
                experience=experience,
            )
            llm = create_llm(model=self._resolve_model())
            response = await llm.ainvoke([sys_msg, usr_msg])
            log_llm_usage(response, caller=self.agent_id)
            return await self._parse_response(extract_content(response), snapshot.pair, llm=llm)
        except Exception:
            # ... 现有 mock fallback
            ...


class ToolAgent(BaseAgent):
    def __init__(self, *, agent_id, prompt_builder, tools, model="", backtest_mode=False) -> None:
        super().__init__(agent_id=agent_id, prompt_builder=prompt_builder, model=model)
        self.tools = list(tools)
        self.backtest_mode = backtest_mode

    async def analyze(self, snapshot, experience=""):
        if self.backtest_mode:
            return await super().analyze(snapshot, experience)
        sys_msg, usr_msg = self._prompt_builder.build(
            snapshot=self._snapshot_to_dict(snapshot),
            portfolio={},
            experience=experience,
        )
        agent = create_agent(
            _create_chat_model(self.model),
            tools=self.tools,
            system_prompt=sys_msg.content,
        )
        result = await agent.ainvoke({"messages": [{"role": "user", "content": usr_msg.content}]})
        # ... 现有 parse 逻辑
        ...
```

**注**：`_snapshot_to_dict(snapshot: DataSnapshot)` 是 helper，把 spec 014 的 `DataSnapshot` 对象转 dict（沿用现有 `_build_prompt()` 的字段映射逻辑）。可以放在 base.py 或 snapshot_renderer.py 内部。

---

### 4 agent 类（spec 014 沿用 + 本 spec 重构）

**重构后**（以 TechAgent 为例）：

```python
# src/cryptotrader/agents/tech.py
from cryptotrader.agents.base import BaseAgent
from cryptotrader.agents.prompt_builder import PromptBuilder
from cryptotrader.agents import _indicators as ta

# ROLE 常量删除
# _build_prompt 方法删除

class TechAgent(BaseAgent):
    def __init__(self, *, prompt_builder: PromptBuilder, model: str = "") -> None:
        super().__init__(agent_id="tech", prompt_builder=prompt_builder, model=model)

    async def analyze(self, snapshot, experience=""):
        # TechAgent-specific：先 compute_indicators，merge 进 snapshot dict
        snapshot_dict = self._snapshot_to_dict(snapshot)
        snapshot_dict["indicators"] = compute_indicators(snapshot.market.ohlcv)
        # 改用临时 snapshot dict 调 base.analyze 的等价逻辑
        sys_msg, usr_msg = self._prompt_builder.build(
            snapshot=snapshot_dict,
            portfolio={},
            experience=experience,
        )
        # ... LLM 调用 + parse（与 BaseAgent.analyze 相同）
```

**保留**：`compute_indicators(ohlcv)` 函数（计算逻辑）+ helper（`_safe_last` / `_recent_values` / `_compute_volume_indicators` / `_compute_trend_fields`）

**注**：TechAgent 不再 extend BaseAgent.analyze；自己实现 analyze 内含 indicators merge 这一步。或者改为重用 BaseAgent.analyze 的 LLM 调用部分（提取 helper）。具体 task 阶段决定 refactor 颗粒度。

---

### AgentsConfig（spec 014 沿用 + 本 spec 重构）

**变更**：

```python
# src/cryptotrader/config.py
@dataclass
class AgentConfig:
    # prompt_template 字段删除（FR-Y22）
    ...

@dataclass
class AgentsConfig:
    def build(
        self,
        agent_id: str,
        *,
        prompt_builder: PromptBuilder,  # NEW 必填（FR-Y19）
        backtest_mode: bool = False,
        model_override: str = "",
        # regime_tags 参数删除（FR-Y19）
    ):
        # _resolve_role / _resolve_skills 调用删除（FR-Y20 / Y21）
        # role_description += "STRATEGY SKILLS" 删除（FR-Y24）
        ...
```

`_build_builtin()` 重构：

```python
@staticmethod
def _build_builtin(agent_id: str, *, prompt_builder: PromptBuilder, model: str, backtest_mode: bool):
    builders = {
        "tech_agent": lambda: TechAgent(prompt_builder=prompt_builder, model=model),
        "chain_agent": lambda: ChainAgent(prompt_builder=prompt_builder, model=model, backtest_mode=backtest_mode),
        "news_agent": lambda: NewsAgent(prompt_builder=prompt_builder, model=model, backtest_mode=backtest_mode),
        "macro_agent": lambda: MacroAgent(prompt_builder=prompt_builder, model=model),
    }
    return builders[agent_id]()
```

## 数据流（一次 cycle）

```
nodes/agents.py:agent_node(state)
    │
    ▼
prompt_builder = _get_or_build_pb(agent_id, model_override)
    │ (lazy-init Provider 单例 + 缓存 PromptBuilder)
    │
    ▼
agent = cfg.agents.build(
    agent_id,
    prompt_builder=prompt_builder,
    backtest_mode=...,
    model_override=...,
)
    │ (AgentsConfig.build 实例化 BaseAgent / ToolAgent / 4 builtin)
    │ (ToolAgent 还要拿 tools 列表 + load_skill_tool)
    │
    ▼
analysis = await agent.analyze(snapshot, experience=experience_str)
    │
    ├─ TechAgent: compute_indicators → snapshot_dict["indicators"]
    │
    ▼
prompt_builder.build(snapshot, portfolio={}, experience=exp)
    │
    ├─ render_crypto_snapshot(snapshot) → snapshot_text
    ├─ skill_provider.get_available_skills(agent_id, snapshot)
    │   └─ discover_skills_for_agent(agent_id) (scope filter)
    ├─ render skills 完整 body → available_skills section
    ├─ if experience: skip memory_provider; recent_memory = experience
    │  else: memory_provider.get_recent_memory(...)（占位）
    │
    ▼
TokenBudgetEnforcer.enforce(sections, budget, priority)
    │
    ▼
组装 (SystemMessage, HumanMessage) + 写 8 字段 OTel telemetry
    │
    ▼
ToolAgent (backtest_mode=False): create_agent(..., system_prompt=sys.content) → ainvoke
BaseAgent / ToolAgent (backtest_mode=True): llm.ainvoke([sys, usr])
    │
    ▼
parse_response → AgentAnalysis
```

## 与 spec 014 / 017a 的契约

- **spec 014 不动**：`agent_skills/<id>/SKILL.md` 协议、`agent_memory/cases/<id>.md` 写入路径、`discover_skills_for_agent` 函数、`load_skill_tool` 函数、`AgentAnalysis` dataclass
- **spec 017a 兼容**：PromptBuilder 公开 API 沿用，仅向后兼容地新增 `experience: str = ""` 参数；DefaultSkillProvider 内部逻辑修 bug；DefaultMemoryProvider 不动（其内部路径 bug 推迟 spec 018 修）
- **spec 015 不动**：`sanitize_input()` 函数沿用，snapshot_renderer 调用方式与 BaseAgent._build_prompt 完全一致

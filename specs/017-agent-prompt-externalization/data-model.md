# Phase 1：数据模型

**关联 spec**：[spec.md](spec.md) — Key Entities 段
**Date**: 2026-05-08

## 实体关系图

```
┌──────────────────┐         ┌──────────────────┐
│  AgentConfig     │ 1     1 │  PromptBuilder   │
│  (frontmatter +  │◄────────┤  (runtime)       │
│   body sections) │         └────────┬─────────┘
└──────────────────┘                  │
                                       │ uses
                          ┌────────────┼────────────┐
                          ▼            ▼            ▼
                ┌────────────────┐ ┌────────────┐ ┌──────────────┐
                │ MemoryProvider │ │SkillProvider│ │TokenBudget   │
                │ (Protocol)     │ │(Protocol)   │ │Enforcer      │
                └───────┬────────┘ └──────┬─────┘ └──────┬───────┘
                        │                 │              │
                  default impl       default impl       returns
                        │                 │              │
                        ▼                 ▼              ▼
                ┌──────────────┐ ┌────────────────┐ ┌──────────────┐
                │DefaultMemory │ │DefaultSkill    │ │EnforceResult │
                │Provider      │ │Provider        │ │(dataclass)   │
                └──────┬───────┘ └────────┬───────┘ └──────────────┘
                       │                  │
                  reads from         reads from
                       │                  │
                       ▼                  ▼
                ┌──────────────┐ ┌────────────────┐
                │agent_memory/ │ │agent_skills/   │
                │(spec 014)    │ │(spec 014)      │
                └──────────────┘ └────────────────┘
```

## 实体定义

### AgentConfig

**职责**：单个 agent 配置的内存表示，由 ConfigLoader 从 `config/agents/<agent_id>.md` 解析得出。

**字段**：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `agent_id` | `str` | ✓ | agent 标识，必须与文件名一致（如 `tech` / `chain` / `news` / `macro`） |
| `description` | `str` | ✓ | 一句话描述（用于 telemetry / 调试） |
| `sections` | `list[str]` | ✓ | 声明 body 中必须存在的 section 名（如 `["system_prompt", "user_tail", "available_skills", "recent_memory", "output_schema"]`） |
| `budget` | `int` | ✓ | token 预算上限（如 `8000`） |
| `priority` | `dict[str, int]` | ✓ | section→优先级；数字小越保留；缺省 section 视作 `999`（最先丢） |
| `slot_overrides` | `dict[str, list[str]]` | ✗ | 覆盖默认 slot 分配（如 `{"system": ["system_prompt", "output_schema"], "user_tail": ["available_skills", "recent_memory", "snapshot", "portfolio", "agent_analyses", "user_tail"]}`） |
| `body_sections` | `dict[str, str]` | ✓（解析后） | section_name → markdown body 字符串；由 body 解析得出 |

**Validation Rules**：
- `agent_id` MUST 匹配文件名（不含扩展名）
- `sections` 中声明的每个名字 MUST 在 `body_sections` 中存在
- `priority` 中的 key MUST 全部出现在 `sections` 中（反之未必）
- `budget` MUST > 0
- `slot_overrides` 若提供，其 value 中引用的 section 名 MUST 在 `sections` 中

**State Transitions**：无状态机；config 是不可变值对象（启动期加载一次）。

---

### PromptBuilder

**职责**：运行时 prompt 组装器；每个 agent 持有独立实例。

**构造签名**：
```python
PromptBuilder(
    agent_id: str,
    config_dir: Path,
    memory_provider: MemoryProvider,
    skill_provider: SkillProvider,
    model: str,
)
```

**主要方法**：

| 方法 | 签名 | 说明 |
|---|---|---|
| `build` | `(snapshot: dict, portfolio: dict, agent_analyses: dict \| None = None) -> tuple[SystemMessage, HumanMessage]` | 组装 LLM messages 唯一对外入口 |

**内部协作**：
1. 构造时读取 `config_dir/<agent_id>.md`，校验后存为 `self.config: AgentConfig`
2. `build()` 调用：
   1. `memory_provider.get_recent_memory(agent_id, snapshot)` → str
   2. `skill_provider.get_available_skills(agent_id, snapshot)` → list[Skill]
   3. 渲染 snapshot / portfolio / agent_analyses → str（按 body 中 `## snapshot` / `## portfolio` 段的模板）
   4. `TokenBudgetEnforcer.enforce(sections, budget, priority)` → `EnforceResult`
   5. 按 slot_overrides 或默认分配，组装 SystemMessage + HumanMessage
   6. 写 telemetry（8 字段）

**生命周期**：每个 agent 实例化时创建一个 PromptBuilder；进程生命周期内复用。

---

### MemoryProvider（Protocol）

**职责**：记忆数据源的协议接口；本 spec 提供 `DefaultMemoryProvider`，spec 018 提供进化版实现。

**Protocol 定义**：
```python
class MemoryProvider(Protocol):
    def get_recent_memory(
        self,
        agent_id: str,
        snapshot: dict,
        k: int = 5,
    ) -> str: ...
```

**返回值约定**：已格式化的 markdown 字符串；空记忆返回固定占位（"暂无历史记忆"）；实现方负责内部排名 / 截断。

---

### DefaultMemoryProvider

**职责**：本 spec 的默认 MemoryProvider 实现，复用 spec 014 `agent_memory/<agent_id>/{patterns.md, cases.jsonl}` 目录结构。

**字段**：

| 字段 | 类型 | 说明 |
|---|---|---|
| `memory_root` | `Path` | 默认 `agent_memory/`，可注入 |
| `top_k_patterns` | `int` | 默认 5 |
| `top_k_cases` | `int` | 默认 3 |

**输出格式**：
```markdown
### Patterns
- [pattern_id] description (confidence=0.85)
- ...

### Cases
- [case_id] context summary → outcome (PnL=+2.3%)
- ...
```

**Validation Rules**：
- 若 `agent_memory/<agent_id>/` 目录不存在 → 返回占位字符串"暂无历史记忆"
- `cases.jsonl` 解析失败的行 → 跳过 + warning log（不抛异常）

---

### SkillProvider（Protocol）

**职责**：技能数据源的协议接口；本 spec 提供 `DefaultSkillProvider`，spec 018 提供进化版（含 IDF / Hermes match-score）。

**Protocol 定义**：
```python
class SkillProvider(Protocol):
    def get_available_skills(
        self,
        agent_id: str,
        snapshot: dict,
        k: int = 5,
    ) -> list[Skill]: ...
```

---

### DefaultSkillProvider

**职责**：本 spec 的默认 SkillProvider 实现，复用 spec 014 `agent_skills/<id>/SKILL.md` 文件协议。

**字段**：

| 字段 | 类型 | 说明 |
|---|---|---|
| `skills_root` | `Path` | 默认 `agent_skills/`，可注入 |

**输出格式**：返回 `list[Skill]`，每个 Skill 含 `skill_id` / `description` / `steps`（关键步骤摘要）等字段（沿用 spec 014 schema）。

**ranking 策略（本 spec 范围）**：简单 keyword match — agent_id 出现在 SKILL.md frontmatter `tags` 中即入选；不做 IDF / 进化（留 spec 018）。

**Validation Rules**：
- 若 `agent_skills/` 目录不存在 → 返回 `[]`
- 单个 SKILL.md 解析失败 → 跳过 + warning log

---

### Skill

**职责**：单个技能的数据载体（沿用 spec 014 schema，本 spec 不变）。

**字段**：

| 字段 | 类型 | 说明 |
|---|---|---|
| `skill_id` | `str` | 唯一标识（如 `funding-rate-extreme-fade`） |
| `description` | `str` | 一句话描述 |
| `tags` | `list[str]` | 关联 agent / 主题（如 `["macro", "funding-rate"]`） |
| `steps` | `list[str]` | 关键步骤摘要（用于 markdown bullet 渲染） |
| `body` | `str` | 完整 SKILL.md body（本 spec 不直接使用，留 spec 018） |

---

### TokenBudgetEnforcer

**职责**：按优先级丢/降 section 至 token 预算内。

**主要方法**：

| 方法 | 签名 | 说明 |
|---|---|---|
| `enforce` | `(sections: dict[str, str], budget: int, priority: dict[str, int]) -> EnforceResult` | 核心方法 |

**算法**（伪码）：
```
1. total = sum(_estimate_tokens(v) for v in sections.values())
2. if total <= budget: return EnforceResult(...)
3. dropped = []
4. sorted_sections = sorted(sections, key=lambda k: priority.get(k, 999), reverse=True)
   # priority 数字大越先丢
5. for name in sorted_sections:
       if name in {"system_prompt", "output_schema"}: continue  # 强制保留
       sections.pop(name)
       dropped.append(name)
       if sum(_estimate_tokens(v) for v in sections.values()) <= budget: break
6. degraded = []
7. if total still > budget:
       for name in ["recent_memory", "available_skills"]:
           if name in sections:
               sections[name] = _truncate(sections[name], target_tokens=...)
               degraded.append(name)
8. return EnforceResult(sections, dropped, degraded, total_pre, total_post, budget)
```

---

### EnforceResult

**职责**：TokenBudgetEnforcer 输出的 dataclass。

**字段**：

| 字段 | 类型 | 说明 |
|---|---|---|
| `final_sections` | `dict[str, str]` | 经过丢/降后的最终 section 字典 |
| `dropped_sections` | `list[str]` | 被丢弃的 section 名 |
| `degraded_sections` | `list[str]` | 被截断（降级）的 section 名 |
| `prompt_size_pre` | `int` | 处理前估算 token 数 |
| `prompt_size_post` | `int` | 处理后估算 token 数 |
| `budget` | `int` | 输入的 budget（便于 telemetry 反查） |

---

### ConfigValidationError

**职责**：启动期 config 校验失败时抛出。

**字段**：
| 字段 | 类型 | 说明 |
|---|---|---|
| `file_path` | `Path` | 失败的 config 文件路径 |
| `reason` | `str` | 失败原因（如 "缺少必填字段 'agent_id'"、"YAML 解析失败: ..."、"sections 声明的 'output_schema' 在 body 中不存在"） |

**继承**：`Exception`

## 数据流

### 启动期（一次性）

```
nodes/agents.py 实例化 4 个 agent
        │
        ▼
对每个 agent：
  PromptBuilder(agent_id, config_dir, memory_provider, skill_provider, model)
        │
        ▼
  ConfigLoader.load(config_dir/<agent_id>.md)
        │  ← 失败抛 ConfigValidationError，进程崩
        ▼
  AgentConfig (frontmatter + body_sections)
        │
        ▼
  PromptBuilder 持有 self.config 备用
```

### 运行期（每 cycle 4×）

```
agent.run(snapshot, portfolio, agent_analyses)
        │
        ▼
prompt_builder.build(snapshot, portfolio, agent_analyses)
        │
        ├─→ memory_provider.get_recent_memory(...) → str
        ├─→ skill_provider.get_available_skills(...) → list[Skill]
        ├─→ 渲染 snapshot/portfolio/agent_analyses → str
        │
        ▼
sections: dict[str, str] = {
    "system_prompt": ...,
    "user_tail": ...,
    "available_skills": markdown bullet list,
    "recent_memory": ...,
    "output_schema": ...,
    "snapshot": ...,
    "portfolio": ...,
    "agent_analyses": ... | "",  # 可选
}
        │
        ▼
TokenBudgetEnforcer.enforce(sections, budget, priority) → EnforceResult
        │
        ▼
按 slot_overrides / 默认分配组装：
  SystemMessage(content=join(system_slot_sections))
  HumanMessage(content=join(user_slot_sections))
        │
        ▼
写 telemetry（8 字段挂当前 active span 或 structured log）
        │
        ▼
return (SystemMessage, HumanMessage)
        │
        ▼
agent 把它们传给 LLM
```

## 与 spec 014 数据模型的关系

- **不修改** `agent_skills/<id>/SKILL.md` 文件协议（spec 014 定义）
- **不修改** `agent_memory/<agent_id>/{patterns.md, cases.jsonl}` 目录结构（spec 014 定义）
- **不修改** spec 014 的 5 层防过拟合 / PnL maturity FSM / 单写者反射模型（这些是 spec 018 才会动的）

本 spec 仅在 4 个 analysis agent 的 prompt 拼接路径上引入新结构，不破坏 spec 014 任何 invariant。

# Phase 1：数据模型

**关联 spec**：[spec.md](spec.md)
**Date**: 2026-05-09

## 实体关系图

```
┌─────────────────────────────────┐
│  Skill (spec 014 既有 + 本 spec)  │ — agent_skills/<name>/SKILL.md
│  + 6 新字段（全 default）：        │   YAML frontmatter + Markdown body
│    regime_tags / triggers_keywords│
│    importance / access_count /    │
│    last_accessed_at / confidence  │
└──────────┬───────────────────────┘
           │ stored as YAML frontmatter
           ▼
┌─────────────────────────────────┐
│  EvolvingSkillProvider (本 spec) │ — implements spec 017a SkillProvider Protocol
│  - get_available_skills()        │
│  - get_skill_by_name()           │
└──────────┬───────────────────────┘
           │ uses
           ▼
   IDF / RecencyBonus / Pareto-style sort
   ┌───────────┬────────────┐
   ▼           ▼            ▼
┌──────┐  ┌──────────┐  ┌──────────────┐
│ idf  │  │recency   │  │importance ×  │
│ .py  │  │bonus     │  │confidence    │
└──────┘  └──────────┘  └──────────────┘

┌─────────────────────────────────┐
│  load_skill_tool (factory)       │ — _make_load_skill_tool(provider, skill_dir)
│  返回 LangChain @tool             │
└──────────┬───────────────────────┘
           │ 调用 (本 spec 改造后)
           ▼
   provider.get_skill_by_name(name) (FR-W13)

┌─────────────────────────────────┐
│  propose_new_skill (改造)        │ — 生成 .draft + LLM 推断 metadata
└──────────┬───────────────────────┘
           │ 调用
           ▼
   skill_metadata_inference.infer_skill_metadata()
                │
                ▼
   LLM JSON output: {regime_tags, triggers_keywords, importance, confidence}
```

## 实体定义

### Skill（spec 014 既有 + 本 spec 扩展）

**位置**：`src/cryptotrader/agents/skills/schema.py:46`

**spec 014 既有字段**（不变）：

| 字段 | 类型 | 说明 |
|---|---|---|
| `name` | str | skill 名 |
| `description` | str | 一句话描述 |
| `scope` | str | "shared" 或 "agent:<agent_id>" |
| `body` | str | SKILL.md body |
| `file_path` | Path | 文件路径 |
| `manually_edited` | bool | 是否人工编辑 |
| `version` | str | 版本号（如 "1.0"） |
| `mtime` | float | 磁盘 mtime |

**本 spec 新增字段**（FR-W2，全部 default 兼容旧实例）：

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `regime_tags` | list[str] | `[]` | 空 list 视为 match all regime（向后兼容 spec 014/15/17b 全注入语义） |
| `triggers_keywords` | list[str] | `[]` | IDF 输入；空 list 时 IDF 评分=0 |
| `importance` | float | `0.5` | 0.0-1.0 |
| `access_count` | int | `0` | 每次注入到 prompt +1（含 retrieval 路径与 load_skill_tool 路径） |
| `last_accessed_at` | datetime | factory `datetime.now(UTC)` | 时间衰减输入 |
| `confidence` | float | `0.5` | 0.0-1.0；reflection / LLM 推断设定 |

**Validation rules**：
- `importance ∈ [0.0, 1.0]`
- `confidence ∈ [0.0, 1.0]`
- `access_count >= 0`
- `last_accessed_at <= now`

**Property methods**（spec 014 既有）：
- `agent_id` 从 `scope` 提取 agent_id（"agent:tech" → "tech"，"shared" → None）
- `is_shared` = (scope == "shared")

---

### EvolvingSkillProvider（本 spec 新增）

**位置**：`src/cryptotrader/learning/evolution/skill_provider.py`（NEW）

**实现 Protocol**：spec 017a `SkillProvider`

**构造**：
```python
class EvolvingSkillProvider:
    def __init__(
        self,
        skill_root: Path = Path("agent_skills"),
        top_k: int = 5,
    ): ...
```

**主要方法**：

| 方法 | 签名 | 说明 |
|---|---|---|
| `get_available_skills` | `(agent_id, snapshot, k=5) -> list[Skill]` | 实现 SkillProvider Protocol |
| `get_skill_by_name` | `(name) -> Skill \| None` | FR-W12 / load_skill_tool 用 |

**容错**（FR-W9）：所有 public 方法 try/except 包裹，异常时返回空 list / None + warning log。

---

### Transition（沿用 spec 018 概念，本 spec 不重新定义）

skill 状态没有 FSM transition（不加 maturity 字段），所以本 spec **不输出** transition events。

---

### load_skill_tool（spec 014 既有 + 本 spec 改造）

**位置**：`src/cryptotrader/agents/skills/tool.py`

**Factory 签名**（本 spec 改造）：
```python
def _make_load_skill_tool(
    provider: EvolvingSkillProvider | None = None,
    skill_dir: Path | None = None,
) -> Callable: ...
```

**Module-level instance**：
- `load_skill_tool = _make_load_skill_tool()`（spec 014 既有，provider=None 兜底）
- `nodes/agents.py` 在 init 时替换：
  ```python
  import cryptotrader.agents.skills.tool as _t
  _t.load_skill_tool = _t._make_load_skill_tool(provider=_skill_provider)
  ```

**Tool 内部行为**（FR-W13）：
- provider 非空 → `provider.get_skill_by_name(name)` → 返回 body 或 error string
- provider 空 → 走 spec 014 兜底（直接 `load_skill(name, skill_dir)` 读文件）
- access_count 仅在 provider 路径累计（spec 014 兜底路径不累计）

---

### propose_new_skill（spec 014 既有 + 本 spec 改造）

**位置**：`src/cryptotrader/learning/skill_proposal.py:propose_new_skill`

**spec 014 既有逻辑**：
1. 分析 active patterns 找共同 regime/theme 子集
2. 生成 proposed_name（如 `<agent>-<theme>`）
3. 构建 draft_content（含 frontmatter + body）
4. 写入 `agent_skills/<proposed_name>/SKILL.md.draft`

**本 spec 新增步骤**（FR-W16）：
- 步骤 3 之后、步骤 4 之前：调 `infer_skill_metadata(name, description, body)` 获取 LLM 推断 metadata
- LLM 推断结果合并到 frontmatter
- LLM 失败时使用默认值

**新增 telemetry**（FR-W29）：写 7 OpenTelemetry attribute。

---

### skill_metadata_inference（本 spec 新增模块）

**位置**：`src/cryptotrader/learning/evolution/skill_metadata_inference.py`（NEW）

**核心函数**：
```python
def infer_skill_metadata(
    name: str,
    description: str,
    body: str,
    llm_callable: Callable | None = None,
) -> dict:
    """LLM 推断 skill metadata。

    Returns:
        {"regime_tags": [...], "triggers_keywords": [...],
         "importance": 0.0-1.0, "confidence": 0.0-1.0}

    LLM 失败 / parse 失败 / 重试失败 → 默认值
    """
```

**LLM Prompt 结构**（FR-W17）：见 research.md Decision 4。

---

### IDF 算法（本 spec 新增模块）

**位置**：`src/cryptotrader/learning/evolution/idf.py`（NEW）

**核心函数**：
```python
def compute_idf(corpus_keywords: list[list[str]]) -> dict[str, float]:
    """计算 IDF 表。corpus_keywords: 每个 skill 的 triggers_keywords list。

    返回 {keyword: idf_score} 字典。
    """

def score_skill(
    skill_keywords: list[str],
    query_keywords: set[str],
    idf_table: dict[str, float],
) -> float:
    """计算 skill 在 query 上的 IDF score。"""

def extract_query_keywords(snapshot: dict) -> set[str]:
    """从 snapshot dict 提取 query keywords（字段名 + 关键值小写化）。"""
```

---

### PromptBuilderSingleton（spec 017b/18 既有 + 本 spec 修改）

**位置**：`src/cryptotrader/nodes/agents.py` module-level dict

**spec 018 状态**：
```python
_memory_provider = EvolvingMemoryProvider(...)  # spec 018
_skill_provider = DefaultSkillProvider(...)      # spec 017a/b（待替换）
_prompt_builders: dict[str, PromptBuilder] = {}
```

**本 spec 修改**：
```python
_memory_provider = EvolvingMemoryProvider(...)   # spec 018 不动
_skill_provider = EvolvingSkillProvider(...)     # 本 spec 替换
# 同时 wire load_skill_tool
import cryptotrader.agents.skills.tool as _t
_t.load_skill_tool = _t._make_load_skill_tool(provider=_skill_provider)
```

## 数据流（一次 cycle）

```
nodes/agents.py:agent_node(state)
    │
    ▼
prompt_builder = _get_or_build_pb(agent_id, model)
    │ (lazy-init Provider 单例)
    │ _memory_provider = EvolvingMemoryProvider (spec 018)
    │ _skill_provider = EvolvingSkillProvider (本 spec)
    │
    ▼
agent.analyze(snapshot, experience)
    │
    ▼
prompt_builder.build(snapshot, portfolio, experience=experience)
    │
    ├─→ EvolvingMemoryProvider.get_recent_memory()  # spec 018，不变
    │
    ├─→ EvolvingSkillProvider.get_available_skills()  # 本 spec
    │   ├─ 第一层：scope filter（discover_skills_for_agent）+ regime_tags 预过滤
    │   ├─ 第二层：score = (idf + importance + recency_bonus) × confidence
    │   ├─ 取 top-k
    │   ├─ 写回 access_count + last_accessed_at
    │   └─ 写 4 telemetry attribute
    │   (任一步异常 → 空 list + warning log)
    │
    ▼
LLM 调用 → 4 agent 完成
    │
    ▼
... debate / verdict / risk_gate / evaluate (spec 018) / journal ...

# Skill 进化触发器（C 决策）：本 spec 不在 cycle 中跑高级进化；
# access_count / last_accessed_at 已在 retrieval 时累计。
# 高级进化（importance 重计算 / stale 标记 / curation）推迟 spec 020 daemon。
```

```
ToolAgent.analyze (spec 017b 流程)
    │
    ▼
LangChain create_agent loop
    │
    ▼
LLM 调用 load_skill_tool(name="...")
    │
    ▼
load_skill_tool 内部：
    if provider is not None:
        skill = provider.get_skill_by_name(name)  # 本 spec 路径
        return skill.body                           # access_count +1
    else:
        return load_skill(name)                     # spec 014 兜底
```

```
spec 014 既有 propose_new_skill 触发（手动 / spec 020 daemon）
    │
    ▼
1. 分析 active patterns
2. 生成 proposed_name + draft_content
3. (本 spec) 调 infer_skill_metadata(name, description, body) → LLM JSON
4. 把 metadata 合并到 frontmatter
5. 写入 SKILL.md.draft
6. 写 7 telemetry attribute（FR-W29）

用户后续 review .draft 文件 → manual save → 变 SKILL.md（metadata 已就位）
```

## 与 spec 014 / 17a / 17b / 18 的契约

- **spec 014 不动**：Skill dataclass 既有 8 字段；discover_skills_for_agent；load_skill 函数；propose_new_skill 函数名 + .draft 写入路径
- **spec 017a 兼容**：SkillProvider Protocol 不变
- **spec 017b 兼容**：4 agent / BaseAgent / ToolAgent / nodes/agents.py 框架不动；DefaultSkillProvider class 删除（FR-W11）
- **spec 018 兼容**：EvolvingMemoryProvider 不动；同 module-level singleton 中并存
- **spec 015 不动**：sanitize_input 沿用（LLM 推断 prompt 中清洗 description / body）

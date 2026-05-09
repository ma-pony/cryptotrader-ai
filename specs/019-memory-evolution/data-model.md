# Phase 1：数据模型

**关联 spec**：[spec.md](spec.md)
**Date**: 2026-05-09

## 实体关系图

```
┌─────────────────────────────────┐
│  PatternRecord (spec 014 既有)   │ — agent_memory/<agent>/patterns/<name>.md
│  + spec 018 新增字段：           │   YAML frontmatter + Markdown body
│    importance / access_count /  │
│    last_accessed_at /            │
│    last_modified_at /            │
│    fundamental_failure_streak    │
└──────────┬───────────────────────┘
           │ stored as
           ▼
┌─────────────────────────────────┐
│  Maturity (sealed Literal)       │
│  spec 014: observed/probationary │
│           /active/deprecated     │
│  spec 018 加: archived           │
└──────────────────────────────────┘

┌─────────────────────────────────┐
│  CaseRecord (spec 014 既有)      │ — agent_memory/cases/<cycle_id>.md
│  + spec 018 新增字段：           │
│    trade_execution: dict         │
│    causal_chain: dict            │
│    ive_classification: dict      │
└──────────┬───────────────────────┘
           │ classified by
           ▼
┌─────────────────────────────────┐
│  FailureClassification (本 spec) │
│  case_id / failure_type /        │
│  reasoning / confidence /        │
│  diagnostic_answers              │
└──────────────────────────────────┘

┌─────────────────────────────────┐
│  EvolvingMemoryProvider          │ — implements MemoryProvider Protocol
│  - get_recent_memory()           │
│  - evaluate_all_rules()          │
│  - classify_pending_cases()      │
└──────────┬───────────────────────┘
           │ uses
           ▼
   FSM / Pareto / IVE 三个独立模块
```

## 实体定义

### PatternRecord（spec 014 既有 + 本 spec 扩展）

**位置**：`src/cryptotrader/agents/skills/schema.py:74`

**spec 014 既有字段**（不变）：

| 字段 | 类型 | 说明 |
|---|---|---|
| `name` | str | rule 名 |
| `agent` | str | tech / chain / news / macro |
| `description` | str | 一句话描述 |
| `body` | str | rule 完整 markdown body |
| `regime_tags` | list[str] | regime 标签（如 `["trending"]`） |
| `pnl_track` | PnLTrack | 封装 successes / losses / total_pnl |
| `maturity` | Maturity | 沿用既有 Literal |
| `source_cycles` | list[str] | 来源 cycle id |
| `created` | datetime | 创建时间 |
| `file_path` | Path | 文件路径 |
| `manually_edited` | bool | 是否人工编辑 |
| `version` | int | 版本号 |

**本 spec 新增字段**（FR-Z6）：

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `importance` | float (0.0-1.0) | 0.5 | reflect 设定，PnL 稳定的 rule 高分 |
| `access_count` | int | 0 | 每次注入到 prompt +1 |
| `last_accessed_at` | datetime | factory `datetime.now(UTC)` | 时间衰减输入 |
| `last_modified_at` | datetime | factory `datetime.now(UTC)` | FSM "5 cycle 无修改"输入 |
| `fundamental_failure_streak` | int | 0 | 累计 fundamental IVE 分类次数 |

**Validation rules**：
- `importance` ∈ [0.0, 1.0]
- `access_count >= 0`
- `fundamental_failure_streak >= 0`
- `last_accessed_at <= now`
- `last_modified_at <= now`

---

### Maturity（spec 014 既有 + 本 spec 扩展）

**位置**：`src/cryptotrader/agents/skills/schema.py:15`

**spec 014 既有 Literal**：`Literal["observed", "probationary", "active", "deprecated"]`

**spec 018 扩展**：加入 `archived` 终态

**完整定义**：
```python
Maturity = Literal["observed", "probationary", "active", "deprecated", "archived"]
```

**State Transitions**（FR-Z11）：

| 起始 | 信号 | 目标 |
|---|---|---|
| `observed` | `pnl_track.successes >= 3` | `probationary` |
| `probationary` | `(now - last_modified_at) >= 5 cycle 或 3 day` 且 frontmatter 全填 + body ≤ 300 行 | `active` |
| `active` | `fundamental_failure_streak >= 3` | `archived` |
| `active` | rule 在 active 被 reflect 修改 | `probationary`（撤销） |
| `deprecated` | — | 终态（不评估） |
| `archived` | — | 终态（不评估） |

---

### CaseRecord（spec 014 既有 + 本 spec 扩展）

**位置**：`src/cryptotrader/agents/skills/schema.py:92`

**spec 014 既有字段**（不变）：

| 字段 | 类型 | 说明 |
|---|---|---|
| `cycle_id` | str | 唯一 ID |
| `timestamp` | datetime | cycle 时间 |
| `pair` | str | 交易对 |
| `snapshot_summary` | dict | snapshot 摘要 |
| `agent_analyses` | dict[str, str] | 4 agent 分析 |
| `verdict_action` | str | long/short/hold/close |
| `verdict_reasoning` | str | verdict 理由 |
| `applied_patterns` | list[str] | 应用的 pattern |
| `risk_gate_passed` | bool | risk 通过 |
| `execution_status` | dict \| None | 执行状态 |
| `final_pnl` | float \| None | 最终 PnL |
| `file_path` | Path | 文件路径 |

**本 spec 新增字段**（FR-Z6b）：

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `trade_execution` | dict \| None | None | entry_price / stop_loss / take_profit / actual_exit_price / fill_status / hit_sl / hit_tp / exit_reason |
| `causal_chain` | dict \| None | None | per-agent tool_calls list 摘要 + verbal_reinforcement_input + debate_intermediate |
| `ive_classification` | dict \| None | None | failure_type / reasoning / confidence / diagnostic_answers |

**Body Schema 扩展**：

```markdown
---
[frontmatter — spec 014 字段不变]
---

# Cycle Record: <cycle_id>

## Agent Analyses           # spec 014 既有
## Verdict Reasoning        # spec 014 既有

## Trade Execution          # spec 018 新增
- entry_price: ...
- stop_loss: ...
- take_profit: ...
- actual_exit_price: ...
- fill_status: ...
- hit_sl: bool
- hit_tp: bool
- exit_reason: ...

## Causal Chain             # spec 018 新增
### Tool Calls (per agent)
[summary list, each ≤500 chars]

### Verbal Reinforcement Input
[experience: str from spec 014]

### Debate Intermediate
[debate rounds output if any]

## IVE Classification       # spec 018 新增
- failure_type: implementation | fundamental | noise
- confidence: 0.0-1.0
- reasoning: ...
- diagnostic_answers:
  1. (yes|no|uncertain): ...
  2. (yes|no|uncertain): ...
  3. (yes|no|uncertain): ...
  4. (yes|no|uncertain): ...
  5. (yes|no|uncertain): ...
```

---

### FailureClassification（本 spec 新增 dataclass）

**位置**：`src/cryptotrader/learning/evolution/ive.py`（NEW）

```python
@dataclass
class FailureClassification:
    case_id: str
    failure_type: Literal["implementation", "fundamental", "noise"]
    reasoning: str
    confidence: float  # 0.0-1.0
    diagnostic_answers: list[str]  # 5 items, each "yes" | "no" | "uncertain"
```

**写回路径**：序列化为 dict 写入 `CaseRecord.ive_classification` + Markdown body 的 `## IVE Classification` 段。

---

### EvolvingMemoryProvider（本 spec 新增 class）

**位置**：`src/cryptotrader/learning/evolution/provider.py`（NEW）

**实现 Protocol**：spec 017a `MemoryProvider`

**构造**：

```python
class EvolvingMemoryProvider:
    def __init__(
        self,
        memory_root: Path = Path("agent_memory"),
        top_k_rules: int = 5,
        top_n_cases: int = 5,
    ): ...
```

**主要方法**：

| 方法 | 签名 | 说明 |
|---|---|---|
| `get_recent_memory` | `(agent_id, snapshot, k=5) -> str` | 实现 Protocol；返回 markdown |
| `evaluate_all_rules` | `() -> list[Transition]` | spec 020 的 trigger 接口 |
| `classify_pending_cases` | `() -> list[FailureClassification]` | 对未分类 case 跑 IVE |

**容错机制**（FR-Z9）：所有 public 方法用 try/except 包裹，异常时返回空字符串 / 空列表 + warning log。

---

### Transition（本 spec 新增 dataclass）

**位置**：`src/cryptotrader/learning/evolution/fsm.py`（NEW）

```python
@dataclass
class Transition:
    rule_id: str
    agent_id: str
    old_state: Maturity
    new_state: Maturity
    triggered_by: str  # 哪个信号触发的（"pnl_threshold" / "time_elapsed" / "fundamental_streak" / "reflect_modified"）
    timestamp: datetime
```

---

### evaluate_node 节点（本 spec 新增 LangGraph node）

**位置**：`src/cryptotrader/nodes/evolution.py`（NEW）

**签名**：

```python
async def evaluate_node(state: ArenaState) -> dict:
    """Cycle 末段：触发 Provider.evaluate_all_rules + classify_pending_cases，写 telemetry。"""
    ...
```

**输入**：state（含 cycle_id / current case data）
**输出**：partial state update（含 evaluation summary 或空）

**插入位置**（FR-Z23）：`graph.py:_build_full_graph` 的 `risk_gate` 节点之后、`journal_trade`/`journal_rejection` 之前。具体策略：在 risk_router 后加 evaluate 节点共享给两条 journal 分支。

---

### Memory API（本 spec 新增）

**位置**：`src/api/routes/memory.py`（NEW）

**4 个 endpoints**：

| Method | Path | Query 参数 | 返回 |
|---|---|---|---|
| GET | `/api/memory/rules` | `agent: str?` / `status: Maturity?` | list[PatternRecord summary] |
| GET | `/api/memory/cases` | `from: ISO?` / `to: ISO?` / `agent: str?` | list[CaseRecord summary 含 IVE] |
| GET | `/api/memory/transitions` | `since: ISO?` | list[Transition] |
| GET | `/api/memory/archived` | — | list[archived rule summary] |

**注册**（FR-Z25）：`src/api/main.py` 加 `app.include_router(memory.router, prefix="/api/memory", tags=["memory"])`

---

### Web `/memory` 页面（本 spec 新增）

**位置**：`web/src/pages/memory/MemoryPage.tsx`（NEW）

**4 sections**：

```tsx
function MemoryPage() {
  return (
    <div>
      <RulesGrid />          {/* 4 agent × 5 maturity grid */}
      <CasesTimeline />      {/* IVE classification 时间线 */}
      <ArchivedRules />      {/* archived rules 历史 */}
      <RecentTransitions />  {/* fsm_transition 事件流 */}
    </div>
  )
}
```

**Sidebar 集成**（FR-Z27）：`web/src/components/layout/sidebar.tsx` 在 `/risk` 后、`/metrics` 前加：

```tsx
{ to: '/memory', labelKey: 'nav.memory', icon: Brain }
```

## 数据流（一次 cycle）

```
nodes/agents.py:agent_node(state)
    │
    ▼
prompt_builder = _get_or_build_pb(agent_id, model)
    │ (lazy-init Provider 单例 — 现在是 EvolvingMemoryProvider)
    │
    ▼
agent.analyze(snapshot, experience)
    │
    ▼
prompt_builder.build(snapshot, portfolio, experience=experience)
    │
    ├─→ EvolvingMemoryProvider.get_recent_memory(agent_id, snapshot)
    │   ├─ 读 agent_memory/<agent>/patterns/*.md (排除 archived/deprecated)
    │   ├─ Pareto frontier 排序
    │   ├─ importance × log(1+access_count) × time_decay 二次排
    │   ├─ 取 top-k
    │   ├─ 写回 access_count + last_accessed_at
    │   ├─ 读 agent_memory/cases/*.md 最近 N case
    │   └─ 渲染 markdown
    │   (任一步异常 → 空字符串 + warning log)
    │
    ▼
LLM 调用 → 4 agent 完成
    │
    ▼
... debate / verdict / risk_gate ...
    │
    ▼
risk_router (verdict pass / reject)
    │
    ▼
evaluate_node (NEW — 本 spec 插入)
    │
    ├─→ provider.evaluate_all_rules()  # FSM transitions
    │   └─ 写文件回 patterns/*.md
    │
    ├─→ provider.classify_pending_cases()  # IVE LLM × 5 case
    │   └─ 写回 cases/*.md
    │
    └─→ 写 6 telemetry attributes
    │
    ▼
journal_trade / journal_rejection
    │
    ▼
END
```

## 与 spec 014 / 017a / 017b 的契约

- **spec 014 不动**：`PatternRecord` / `CaseRecord` / `PnLTrack` / `Maturity` 4 状态保留；`learning/memory.py` 既有 IO 函数沿用
- **spec 017a 兼容**：PromptBuilder 公开 API 不变；DefaultMemoryProvider class 删除（其路径错代码 — 本 spec C3 commit）
- **spec 017b 兼容**：experience 参数路径不变；4 agent / BaseAgent / ToolAgent 不动；nodes/agents.py 仅改 _get_or_build_pb 内部
- **spec 015 不动**：`sanitize_input` 函数沿用（Causal Chain 段渲染时清洗）

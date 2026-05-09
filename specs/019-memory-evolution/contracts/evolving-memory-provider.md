# Contract：EvolvingMemoryProvider API

**模块路径**：`src/cryptotrader/learning/evolution/provider.py`（本 spec 新增）

## 实现 Protocol

`EvolvingMemoryProvider` 实现 spec 017a 的 `MemoryProvider` Protocol：

```python
class EvolvingMemoryProvider:
    def get_recent_memory(self, agent_id: str, snapshot: dict, k: int = 5) -> str: ...
```

返回 markdown 字符串，结构与 spec 017a `DefaultMemoryProvider` 输出格式一致（`### Patterns` + `### Cases`），但内容由进化算法决定（FSM 状态过滤 + Pareto 排序 + IVE 归档）。

## 公共方法

### `__init__`

```python
def __init__(
    self,
    memory_root: Path = Path("agent_memory"),
    top_k_rules: int = 5,
    top_n_cases: int = 5,
) -> None: ...
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `memory_root` | `Path("agent_memory")` | 项目 memory 根目录（spec 014 既有路径） |
| `top_k_rules` | 5 | get_recent_memory 返回的 top-k rule 数量 |
| `top_n_cases` | 5 | get_recent_memory 返回的 top-n case 数量 |

### `get_recent_memory`（实现 Protocol）

```python
def get_recent_memory(
    self,
    agent_id: str,
    snapshot: dict,
    k: int = 5,
) -> str: ...
```

**行为约定**（FR-Z8）：
1. 从 `<memory_root>/<agent_id>/patterns/*.md` 读所有非 archived/deprecated 状态规则
2. 调 `pareto.rank_rules(rules)` 排序
3. 二次排序：`importance × log(1 + access_count) × time_decay(last_accessed_at)`
4. 取 top-k；写入 `access_count += 1` / `last_accessed_at = now()` 回文件
5. 从 `<memory_root>/cases/*.md` 读最近 N case（按 timestamp 倒序）
6. 渲染 markdown：`### Patterns`（top-k rules）+ `### Cases`（top-N cases 摘要）

**容错**（FR-Z9）：内部任一步骤异常 → catch + log warning + return `""`（空字符串）。**不抛异常**。

**Telemetry 副作用**：写 OpenTelemetry span attribute `memory.provider.type = "EvolvingMemoryProvider"`（区分 spec 017a 默认实现）。

### `evaluate_all_rules`（spec 020 trigger 接口）

```python
def evaluate_all_rules(self) -> list[Transition]: ...
```

**行为约定**：
1. 遍历 `<memory_root>/<agent>/patterns/*.md` 所有 rule
2. 对每个 rule 调 `fsm.evaluate_transitions(rule)` 计算新状态
3. 状态变化的 rule 写回文件（更新 maturity / last_modified_at）
4. archived 状态 rule 移文件到 `<agent>/patterns/.archived/<rule_name>.md`
5. 返回 list[Transition]（含 rule_id / agent_id / old_state / new_state / triggered_by / timestamp）

**容错**：单 rule 异常 → log warning + 跳过，继续处理其他 rule。

### `classify_pending_cases`（spec 020 trigger 接口）

```python
def classify_pending_cases(self) -> list[FailureClassification]: ...
```

**行为约定**：
1. 遍历 `<memory_root>/cases/*.md` 找 `ive_classification: None` 的 case
2. 对每个调 `ive.classify_case(case)` 跑 LLM
3. 写回 case 文件 `## IVE Classification` 段
4. 若 `failure_type == "fundamental"` → 找该 case 应用的 rule (`applied_patterns`) → rule.fundamental_failure_streak += 1
5. 若 `failure_type != "fundamental"` → 重置 streak = 0
6. 返回 list[FailureClassification]

**容错**：单 case 异常 → 写 `failure_type=noise` + warning log + 继续。

## 集成点

### 调用方 1：PromptBuilder

```python
# nodes/agents.py:_get_or_build_pb
from cryptotrader.learning.evolution.provider import EvolvingMemoryProvider

if _memory_provider is None:
    _memory_provider = EvolvingMemoryProvider(memory_root=Path("agent_memory"))
```

替换 spec 017a 的 `DefaultMemoryProvider`；spec 017a class 整体删除（FR-Z10）。

### 调用方 2：evaluate_node（cycle 末段）

```python
# nodes/evolution.py
async def evaluate_node(state: ArenaState) -> dict:
    provider = _memory_provider  # 从 nodes/agents.py module-level 取
    transitions = provider.evaluate_all_rules()
    classifications = provider.classify_pending_cases()
    # 写 telemetry...
    return {}  # 不修改 state
```

## Schema 升级路径

spec 020 若需替换为 `OptimizedMemoryProvider`（含 daemon + cache）：直接替换 `_memory_provider` 实例即可。本 spec 接口稳定。

## 单测要求

参考 spec.md SC-Z4：`tests/test_evolving_memory_provider.py` ≥ 10 用例 PASS：
- (a) 加载 active rule 走 Pareto 排序
- (b) 加载混合状态规则按状态过滤掉 archived/deprecated
- (c) 加载 case 按 timestamp 倒序 top-N
- (d) `access_count` / `last_accessed_at` 在调用后被回写文件
- (e) IVE LLM 异常时返回空字符串 + warning log
- (f) FSM 异常时返回空字符串
- (g) Pareto 异常时返回空字符串
- (h) IO 异常时返回空字符串
- (i) 空目录返回 "暂无历史记忆"
- (j) Provider 实现 spec 017a `MemoryProvider` Protocol（鸭子类型）

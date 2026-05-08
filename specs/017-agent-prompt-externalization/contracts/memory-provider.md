# Contract：MemoryProvider Protocol

**模块路径**：`src/cryptotrader/agents/prompt_builder.py`（与 PromptBuilder 同模块）

## Protocol 定义

```python
from typing import Protocol


class MemoryProvider(Protocol):
    def get_recent_memory(
        self,
        agent_id: str,
        snapshot: dict,
        k: int = 5,
    ) -> str:
        """返回已格式化的 markdown 记忆文本。

        Args:
            agent_id: 当前调用的 agent 标识（用于读取该 agent 的 memory 子目录）
            snapshot: 当前市场快照（实现方可用其做 regime 匹配）
            k: top-k 截断（实现方决定如何分配 patterns/cases 配额）

        Returns:
            markdown 字符串。空记忆返回固定占位 "暂无历史记忆"。
            实现方负责内部 ranking / 截断 / 格式化。
        """
        ...
```

## 默认实现：DefaultMemoryProvider

```python
class DefaultMemoryProvider:
    def __init__(
        self,
        memory_root: Path = Path("agent_memory"),
        top_k_patterns: int = 5,
        top_k_cases: int = 3,
    ) -> None:
        self._root = memory_root
        self._k_patterns = top_k_patterns
        self._k_cases = top_k_cases

    def get_recent_memory(
        self,
        agent_id: str,
        snapshot: dict,
        k: int = 5,
    ) -> str:
        agent_dir = self._root / agent_id
        if not agent_dir.exists():
            return "暂无历史记忆"

        patterns = self._read_patterns(agent_dir / "patterns.md")
        cases = self._read_cases(agent_dir / "cases.jsonl")

        if not patterns and not cases:
            return "暂无历史记忆"

        parts = []
        if patterns:
            parts.append("### Patterns\n" + "\n".join(f"- {p}" for p in patterns[: self._k_patterns]))
        if cases:
            parts.append("### Cases\n" + "\n".join(f"- {c}" for c in cases[: self._k_cases]))
        return "\n\n".join(parts)
```

### 输入

| 参数 | 类型 | 说明 |
|---|---|---|
| `agent_id` | str | 用于定位 `agent_memory/<agent_id>/` 子目录 |
| `snapshot` | dict | 本默认实现忽略；spec 018 进化版会用其做 regime 匹配 |
| `k` | int | top-k；本默认实现按 `top_k_patterns + top_k_cases` 内部分配 |

### 输出格式（示例）

```markdown
### Patterns
- [pattern_id_1] funding_rate>0.05 + RSI>70 → 24h 内 65% 概率出现 -2% 回调（confidence=0.78）
- [pattern_id_2] OI 单日 +30% + 价格不创新高 → 顶背离信号（confidence=0.71）
- ...

### Cases
- [case_2026-04-12] BTC funding=0.08, RSI=78 → 24h 后 -3.2% (PnL: -$120)
- [case_2026-04-15] ETH OI +35%, 价格滞涨 → 48h 后 -5.1% (PnL: +$280 short)
- ...
```

### 失败模式

| 触发 | 行为 |
|---|---|
| `agent_memory/<agent_id>/` 目录不存在 | 返回 "暂无历史记忆" |
| `patterns.md` 不存在 | 跳过 patterns 段，仅输出 cases |
| `cases.jsonl` 不存在 | 跳过 cases 段，仅输出 patterns |
| `cases.jsonl` 单行 JSON 解析失败 | 跳过该行 + warning log，不抛异常 |
| 读文件 IOError | 抛异常上抛（PromptBuilder 会捕获并降级） |

### 内部辅助方法

```python
def _read_patterns(self, path: Path) -> list[str]:
    if not path.exists():
        return []
    # 简单格式：每行 - 开头视作一条 pattern
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln[2:].strip() for ln in lines if ln.startswith("- ")]


def _read_cases(self, path: Path) -> list[str]:
    if not path.exists():
        return []
    cases: list[str] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            obj = json.loads(ln)
        except json.JSONDecodeError as e:
            logger.warning("cases.jsonl 第 %d 行解析失败: %s", len(cases) + 1, e)
            continue
        # 格式：[case_id] context → outcome (PnL: X)
        cases.append(self._format_case(obj))
    return cases


def _format_case(self, obj: dict) -> str:
    cid = obj.get("case_id", "?")
    ctx = obj.get("context", "")
    outcome = obj.get("outcome", "")
    pnl = obj.get("pnl", "")
    return f"[{cid}] {ctx} → {outcome} (PnL: {pnl})"
```

## 与 spec 018 的协议契约

spec 018 将提供进化版 `EvolvingMemoryProvider`，满足同一 Protocol 即可注入 PromptBuilder：

```python
# spec 018 示意（不在本 spec 实现范围）
class EvolvingMemoryProvider:
    def get_recent_memory(self, agent_id, snapshot, k=5) -> str:
        # 1. 用 snapshot 做 regime 匹配
        # 2. 应用 GEPA / Reflective Mutation 排名
        # 3. 应用 5-signal maturity FSM 过滤
        # 4. 输出 markdown
        ...
```

本 spec 落地的 Protocol 不约束 spec 018 的 ranking 算法；只约束输入 / 输出类型。

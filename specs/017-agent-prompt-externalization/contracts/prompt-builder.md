# Contract：PromptBuilder API

**模块路径**：`src/cryptotrader/agents/prompt_builder.py`

## 公共接口

### `class PromptBuilder`

#### 构造签名

```python
class PromptBuilder:
    def __init__(
        self,
        agent_id: str,
        config_dir: Path,
        memory_provider: MemoryProvider,
        skill_provider: SkillProvider,
        model: str,
    ) -> None: ...
```

#### 参数约束

| 参数 | 类型 | 约束 |
|---|---|---|
| `agent_id` | `str` | 非空，必须等于 `config_dir/<agent_id>.md` 文件名 |
| `config_dir` | `pathlib.Path` | 必须存在且包含 `<agent_id>.md` |
| `memory_provider` | `MemoryProvider` | 满足 Protocol 即可 |
| `skill_provider` | `SkillProvider` | 满足 Protocol 即可 |
| `model` | `str` | LLM 模型名（影响 telemetry，不影响拼接逻辑） |

#### 行为

- 构造时调用 `ConfigLoader.load(config_dir / f"{agent_id}.md")`
- 校验失败抛 `ConfigValidationError`
- 校验通过后存 `self.config: AgentConfig`、`self._memory_provider`、`self._skill_provider`、`self._model`、`self._enforcer = TokenBudgetEnforcer()`

---

#### `build()` 方法

```python
def build(
    self,
    snapshot: dict,
    portfolio: dict,
    agent_analyses: dict | None = None,
) -> tuple[SystemMessage, HumanMessage]: ...
```

#### 参数约束

| 参数 | 类型 | 约束 |
|---|---|---|
| `snapshot` | `dict` | 当前市场数据快照，包含 OHLCV / 资金费率等字段 |
| `portfolio` | `dict` | 当前组合状态，包含持仓、可用资金等 |
| `agent_analyses` | `dict \| None` | 可选；其他 agent 的分析结果（仅 verdict-style 用）；analysis agents 传 None |

#### 返回值约定

- 返回 `(SystemMessage, HumanMessage)` tuple
- `SystemMessage.content` = system slot 中各 section 拼接（按 `slot_overrides.system` 或默认顺序，用 `\n\n` 连接）
- `HumanMessage.content` = user_tail slot 中各 section 拼接

#### 失败模式

| 触发 | 行为 |
|---|---|
| `snapshot` / `portfolio` 字段缺失 | 走默认占位字符串 `"<missing>"`，不抛异常 |
| `memory_provider.get_recent_memory()` 抛异常 | 捕获，warning log，section 退化为占位"暂无历史记忆" |
| `skill_provider.get_available_skills()` 抛异常 | 捕获，warning log，section 退化为占位"暂无可用技能" |
| token 估算后远超 budget 即使全部丢/降仍超 | 仍返回（不抛异常）；telemetry 标记 over_budget=True，由 LLM 端处理超限报错 |

#### Telemetry 副作用

每次调用必须写入以下 8 个 attribute 到当前 active OpenTelemetry span（如无则降级 structured log）：

```
prompt.builder.agent_id          : str
prompt.builder.sections_included : list[str]
prompt.builder.dropped_sections  : list[str]
prompt.builder.degraded_sections : list[str]
prompt.builder.prompt_size_pre   : int
prompt.builder.prompt_size_post  : int
prompt.builder.budget            : int
prompt.builder.duration_ms       : float
```

---

### `class TokenBudgetEnforcer`

#### 主要方法

```python
@dataclass
class EnforceResult:
    final_sections: dict[str, str]
    dropped_sections: list[str]
    degraded_sections: list[str]
    prompt_size_pre: int
    prompt_size_post: int
    budget: int


class TokenBudgetEnforcer:
    def enforce(
        self,
        sections: dict[str, str],
        budget: int,
        priority: dict[str, int],
        protected: set[str] = frozenset({"system_prompt", "output_schema"}),
    ) -> EnforceResult: ...
```

#### 行为约定

1. 计算 `prompt_size_pre = sum(_estimate_tokens(v) for v in sections.values())`
2. 若 `prompt_size_pre <= budget`：直接返回 `EnforceResult(sections, [], [], pre, pre, budget)`
3. 否则按 `priority` 从大到小（数字大先丢）排序，跳过 `protected`，依次 `pop` 直到 ≤ budget 或全部可丢 section 都丢光
4. 若仍 > budget：对 `recent_memory` / `available_skills`（若仍存在）截断到 `budget * 0.3` token，记入 `degraded_sections`
5. `prompt_size_post = sum(_estimate_tokens(v) for v in final_sections.values())`
6. 返回 `EnforceResult(final_sections, dropped, degraded, pre, post, budget)`

#### Token 估算实现

```python
from cryptotrader.learning.context import _estimate_tokens
```

复用 spec 014 已有 CJK-aware 启发式（ASCII÷4 + CJK÷1.5）。误差 < 10% vs tiktoken（spec 014 已验证）。

---

### `class ConfigLoader`

#### 主要方法

```python
class ConfigLoader:
    @staticmethod
    def load(path: Path) -> AgentConfig: ...
```

#### 行为约定

按 [agent-config-schema.md](agent-config-schema.md) 中的"校验规则汇总"9 条逐项检查，任一失败抛 `ConfigValidationError(file_path=path, reason=...)`。

#### Frontmatter / Body 切分

```python
import re
content = path.read_text(encoding="utf-8")
m = re.match(r"^---\n(.*?)\n---\n(.*)$", content, re.DOTALL)
if not m:
    raise ConfigValidationError(path, "无法找到 YAML frontmatter（缺少 --- 分隔符）")
fm_text, body_text = m.group(1), m.group(2)
fm = yaml.safe_load(fm_text)
```

#### Body section 切分

```python
sections: dict[str, str] = {}
current_name: str | None = None
current_lines: list[str] = []
for line in body_text.splitlines():
    if line.startswith("## "):
        if current_name is not None:
            sections[current_name] = "\n".join(current_lines).strip()
        current_name = line[3:].strip()
        current_lines = []
    else:
        current_lines.append(line)
if current_name is not None:
    sections[current_name] = "\n".join(current_lines).strip()
```

---

### `class ConfigValidationError(Exception)`

```python
class ConfigValidationError(Exception):
    def __init__(self, file_path: Path, reason: str) -> None:
        self.file_path = file_path
        self.reason = reason
        super().__init__(f"Config 校验失败 [{file_path}]: {reason}")
```

## 调用契约总结

```python
# 启动期（nodes/agents.py）
from pathlib import Path
from cryptotrader.agents.prompt_builder import (
    PromptBuilder, DefaultMemoryProvider, DefaultSkillProvider
)

config_dir = Path("config/agents")
memory_provider = DefaultMemoryProvider(memory_root=Path("agent_memory"))
skill_provider = DefaultSkillProvider(skills_root=Path("agent_skills"))

tech_pb = PromptBuilder("tech", config_dir, memory_provider, skill_provider, model="claude-3-5-sonnet")
chain_pb = PromptBuilder("chain", config_dir, memory_provider, skill_provider, model="claude-3-5-sonnet")
news_pb = PromptBuilder("news", config_dir, memory_provider, skill_provider, model="claude-3-5-sonnet")
macro_pb = PromptBuilder("macro", config_dir, memory_provider, skill_provider, model="claude-3-5-sonnet")

# Agent 实例化（必填 prompt_builder）
tech_agent = TechAgent(prompt_builder=tech_pb, ...)
# ...

# 运行期（每 cycle 4×）
sys_msg, usr_msg = tech_pb.build(snapshot, portfolio)
response = await llm.ainvoke([sys_msg, usr_msg])
```

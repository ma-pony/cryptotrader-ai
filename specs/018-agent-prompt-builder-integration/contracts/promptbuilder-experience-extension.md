# Contract：PromptBuilder.build() experience 参数扩展

**模块路径**：`src/cryptotrader/agents/prompt_builder.py`（spec 017a 沿用，本 spec 扩展签名）

## 变更概述

spec 017a 的 `PromptBuilder.build()` 当前签名：

```python
def build(
    self,
    snapshot: dict,
    portfolio: dict,
    agent_analyses: dict | None = None,
) -> tuple[SystemMessage, HumanMessage]:
    ...
```

spec 017b 扩展为：

```python
def build(
    self,
    snapshot: dict,
    portfolio: dict,
    agent_analyses: dict | None = None,
    experience: str = "",  # NEW
) -> tuple[SystemMessage, HumanMessage]:
    ...
```

**变更性质**：向后兼容（新参数 default 空字符串）

## 行为约定

### experience 非空时

```python
build(snapshot, portfolio, experience="### Patterns\n- ...\n### Cases\n- ...")
```

行为：
1. **跳过** `self._memory_provider.get_recent_memory(agent_id, snapshot)` 调用
2. `recent_memory` section 内容直接 = `experience` 字符串
3. 其他流程不变（rendering / token budget / telemetry）

### experience 为空时（默认）

```python
build(snapshot, portfolio)
# 等价于 build(snapshot, portfolio, experience="")
```

行为：
1. 调 `self._memory_provider.get_recent_memory(agent_id, snapshot)` 取得文本
2. 沿用 spec 017a `recent_memory` section 注入逻辑

## Telemetry 副作用

`prompt.builder.experience_source` attribute 写入 OpenTelemetry span，取值：
- `"caller"` — experience 非空，由调用方提供
- `"provider"` — experience 空，走 MemoryProvider fallback
- `"empty"` — 双方都空，section 显示占位"暂无历史记忆"

（本 attribute 是 spec 017a FR-X18 列出的 8 字段之外的辅助字段，不计入 8 字段 SC）

## 调用契约

### 调用方 1：BaseAgent.analyze()

```python
async def analyze(self, snapshot: DataSnapshot, experience: str = "") -> AgentAnalysis:
    sys_msg, usr_msg = self._prompt_builder.build(
        snapshot=self._snapshot_to_dict(snapshot),
        portfolio={},
        experience=experience,  # 透传给 PromptBuilder
    )
    ...
```

### 调用方 2：ToolAgent.analyze()

```python
async def analyze(self, snapshot, experience=""):
    if self.backtest_mode:
        return await super().analyze(snapshot, experience)
    sys_msg, usr_msg = self._prompt_builder.build(
        snapshot=self._snapshot_to_dict(snapshot),
        portfolio={},
        experience=experience,
    )
    ...
```

### 调用方 3：4 agent 类（继承 BaseAgent / ToolAgent）

不直接调 `prompt_builder.build()`，通过 super().analyze() 间接传递 experience。

## 失败模式

| 触发 | 行为 |
|---|---|
| experience 不是 str（如 None / list） | 抛 TypeError；调用方有责任传字符串 |
| experience 长度 > token budget | TokenBudgetEnforcer 截断（沿用 017a 路径，experience 进 recent_memory，按 priority 处理） |
| memory_provider.get_recent_memory() 抛异常（experience 空时）| 沿用 017a 异常处理：捕获 + warning log + 占位回退 |

## Schema 升级路径

spec 018 重写 EvolvingMemoryProvider 后，experience 参数仍保留：

```python
# spec 018 调用方示意
sys, usr = pb.build(snapshot, portfolio, experience="")  # 走 EvolvingMemoryProvider
```

EvolvingMemoryProvider 实现 spec 014 路径修复（cases.jsonl → cases/*.md）+ 进化 ranking。spec 017b 的 experience 参数路径作为"调用方覆盖"机制保留。

## 单测要求（含在 spec 017b 现有 SC-Y11）

spec 017a 已有的 7 个 PromptBuilder 测试 + spec 017b 新增 1 个：

- 已有：基础 build / 错误处理 / slot_overrides / snapshot 字段 缺失等 7 用例
- 新增 1 用例：`test_build_experience_overrides_memory_provider` — 传非空 experience，断言 memory_provider 未被调用，recent_memory section 内容 = experience 字符串

新增用例放在 `tests/test_prompt_builder.py`（不创新文件），由 SC-Y11 "44 用例不回归"覆盖（实际变 45 用例）。

## 实施任务（task 阶段拆细）

- 修改 `prompt_builder.py:PromptBuilder.build()` 签名加 `experience: str = ""`
- 修改 build() 内部逻辑：experience 非空 → skip memory_provider，sections["recent_memory"] = experience
- 添加 telemetry attribute `prompt.builder.experience_source`
- 更新 `tests/test_prompt_builder.py` 加 1 个新用例
- 更新 spec 017a `contracts/prompt-builder.md` 反映新签名（属于跨 spec 文档维护，可在 review-code 阶段补）

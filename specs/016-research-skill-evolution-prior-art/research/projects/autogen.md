---
name: microsoft/autogen
url: https://github.com/microsoft/autogen
license: MIT（代码）/ CC-BY-4.0（文档）
tier: 2
last_accessed: 2026-05-08
phase_1_complete: true
phase_2_complete: false
replaces: autoresearch  # per D-PROC-01
---

# microsoft/autogen — Microsoft

## 替换说明（D-PROC-01 执行记录）

autoresearch（uditgoenka）在 Phase 1 审查后被判定与 spec 016 核心主题"技能进化"仅有表面关联：其技能文件为静态 Markdown、无检索层、无动态组装、无版本演化机制。相关分析已归档至 `_deferred-autoresearch.md`。

AutoGen 被选为替换项目的理由：
1. **提示组装有明确的工程模型**：`SystemMessage` + `ChatCompletionContext` + `Memory.update_context()` 构成三层可插拔架构，与 D-PA-01（命名区段模式）高度同构。
2. **记忆-工具-历史三元分离**：AutoGen 显式区分 `_system_messages`（角色描述）、`_tools`（工具列表）、`model_context`（历史）和 `memory`（检索注入），这四个槽位与 cryptotrader-ai spec 017 的四类提示区段存在直接映射。
3. **行业参考地位**：AutoGen 是目前被引用最广泛的多智能体框架之一（截至 2026-05 已进入维护期，合并为 Microsoft Agent Framework），其设计决策代表了工业级提示组装的既有共识。

---

## Architecture Overview

AutoGen 采用分层包结构：`autogen-core`（消息传递与事件驱动基础）、`autogen-agentchat`（高层 API，面向快速原型）、`autogen-ext`（LLM 客户端扩展）、`autogenstudio`（无代码 GUI）。框架面向 Python 3.10+，.NET 侧有独立实现。

核心抽象是 `BaseChatAgent`：定义 `on_messages()`、`on_reset()`、`save_state()`、`load_state()` 四个生命周期方法。每个 agent 实例是有状态的——框架约定每次调用只传入"自上次调用以来的新消息"，agent 自行维护历史上下文。`AssistantAgent` 是最主要的具体实现，集成了 LLM 调用、工具执行、记忆注入的完整流水线。

团队层（`Team`）负责多 agent 编排（RoundRobin、Swarm、Selector 等），但团队协作不属于本次 Phase 1 范围，暂不展开。

AutoGen 目前已进入维护期（仅修复 bug 和安全补丁），Microsoft 推荐新项目迁移至 Microsoft Agent Framework（AutoGen + Semantic Kernel 合并产物）。本研究以 AutoGen 稳定版为准，其设计模式在 MAF 中延续。

---

## Prompt Assembly（Phase 1）

### 系统提示构造

`AssistantAgent.__init__` 接收 `system_message: str | None` 参数，将其包装为 `List[SystemMessage]` 存入 `self._system_messages`。默认值为：

> "You are a helpful AI assistant. Solve tasks using your tools. Reply with TERMINATE when the task has been completed."

默认提示是**单条字符串，非模板**。AutoGen 本身不强制命名区段，提示结构完全由调用方自定义。

源码路径：`python/packages/autogen-agentchat/src/autogen_agentchat/agents/_assistant_agent.py`，`__init__` 及 `_call_llm` 方法（约第 80-280 行）。

### 推理时的消息拼接顺序

每次 `on_messages()` 调用触发 `_call_llm()`，拼接顺序如下：

```
[SystemMessage(s)]              ← self._system_messages（agent 角色定义）
  +
[memory 注入的 LLMMessage(s)]   ← 由 Memory.update_context() 写入 model_context
  +
[历史对话 + 本轮新消息]          ← ChatCompletionContext.get_messages()
```

最终传给 model client：`model_client.create(llm_messages, tools=tools, ...)`

关键细节：**系统提示在 model_client 层拼接，而非写入 ChatCompletionContext**——这意味着 `get_messages()` 返回的列表不包含 SystemMessage，内存记忆内容则写入 context（出现在历史消息中，位置由实现决定）。

### 工具注入

工具以 `Callable | Awaitable[Callable] | BaseTool` 形式注入 `__init__`，内部统一包装为 `FunctionTool`，存入 `self._tools`。Handoff 工具单独存储在 `self._handoff_tools`。

推理时所有工具在 `_call_llm` 中收集并作为 `tools=` 参数传给 model client（即 OpenAI-style function calling 协议）。工具的**名称、描述、参数 JSON Schema** 由函数签名和 docstring 自动提取，不需要手动编写提示文本。

支持 `Workbench` 协议：`for wb in workbench: tools += await wb.list_tools()`——这是一种工具集合的动态注册机制，与"技能目录"概念最接近，但仍是静态列表，无运行时评分或选择。

工具执行循环由 `max_tool_iterations` 参数控制（默认值通常为 10），达到上限后强制停止。

---

## Memory ↔ Skill（Phase 1 lite）

### Memory 抽象接口

`autogen_core.memory.Memory` 是一个协议接口，定义四个异步方法：

| 方法 | 作用 |
|------|------|
| `update_context(model_context)` | 将检索结果写入 ChatCompletionContext（核心注入点） |
| `query(query, **kwargs) → MemoryQueryResult` | 检索 |
| `add(content: MemoryContent)` | 存储 |
| `clear()` | 清空 |

`MemoryContent` 的结构：
- `content: str | bytes | Dict | Image`
- `mime_type: MemoryMimeType`（TEXT / JSON / MARKDOWN / IMAGE / BINARY）
- `metadata: Dict | None`

内置实现 `ListMemory`：按时间序维护 `List[MemoryContent]`，`update_context()` 将最近条目追加到 model_context 的消息列表尾部（作为 `UserMessage` 或类似类型）。

### 记忆注入时机

`AssistantAgent` 在每次 `on_messages()` 调用的**LLM 推理之前**执行：

```python
for mem in self._memory:
    result = await mem.update_context(model_context)
    # 触发 MemoryQueryEvent 事件流
```

即：**每轮对话都重新查询记忆**，不缓存查询结果。这保证了记忆的实时性，但对高频调用场景有额外检索开销。

### ChatCompletionContext 变体

`ChatCompletionContext` 是抽象基类，`get_messages()` 由子类实现：
- `UnboundedChatCompletionContext`：返回全量历史，无截断
- `BufferedChatCompletionContext`：保留最近 N 条消息
- `TokenLimitedChatCompletionContext`：按 token 上限截断（从最旧消息丢弃）

三个变体都不做摘要——截断策略是丢弃，而非压缩。`save_state()` / `load_state()` 支持跨会话持久化。

### 技能概念

AutoGen 没有"技能（Skill）"这一一等概念。最接近的对应物是：
- **Tool**：单次可调用的功能单元，通过 function calling 协议激活
- **Workbench**：工具集合的动态注册表（`list_tools()` 接口）

两者都**不具备**：运行时学习/更新、评分/淘汰机制、跨 agent 共享、版本演化。这是 AutoGen 与 SkillClaw / EvoSkills 等系统的核心差距。

---

## Phase 2 Placeholders

- Evolution Algorithm（AutoGen 无；需分析 MAF 是否引入）
- Skill Data Structure（AutoGen 以 FunctionTool 代替；缺少元数据层）
- Retrieval（Memory.query 接口存在，但无向量存储内置实现）
- Evaluation（无内置指标；max_tool_iterations 是唯一限制器）
- Agent ↔ Skill Boundary（工具归属单一 agent，无跨 agent 共享机制）
- Engineering（Workbench 协议值得深读；MAF 迁移路径待考察）

---

## Borrow Recommendations（Phase 1）

以下建议均面向 cryptotrader-ai 的 4-agent + debate + verdict 架构（spec 017 实施对象）：

**1. 四槽位分离模式（最高优先级）**

AutoGen 将提示的四个来源显式分开：`_system_messages`（静态角色）、`_tools`（工具列表）、`model_context`（历史 + 记忆注入）、`_call_llm` 拼接逻辑（组装器）。cryptotrader-ai 目前将所有内容混写在单一 `ROLE` 字符串中，可按此四槽模型重构为命名区段——这直接支持 D-PA-01。

**2. Memory.update_context() 作为 per-turn 注入钩子**

AutoGen 的 `update_context()` 模式——在每次 LLM 调用前触发、允许记忆层动态修改 context——可作为 cryptotrader-ai `verbal_reinforcement()` 的架构参考。目前 verbal_reinforcement 在 node 层硬连线，可抽象为注入钩子，使 experience memory 与 prompt 组装解耦。

**3. ChatCompletionContext 变体 → 历史管理策略**

`TokenLimitedChatCompletionContext` 的"超限丢弃最旧"策略与 cryptotrader-ai 的 debate 轮次历史管理同构。debate 节点当前没有 token 预算控制；可借鉴此变体模式为每个 debate round 设置独立的上下文窗口，防止多轮辩论溢出。

**4. Workbench 协议 → 动态技能目录接口**

`Workbench.list_tools()` 是 AutoGen 中最接近"技能目录"的接口：agent 不直接持有工具引用，而是按需从 workbench 查询。cryptotrader-ai spec 017 若要实现"可热更新的技能集合"，可参照此接口设计一个 `SkillRegistry.list_skills(agent_id, context)` 方法，将技能选择从 agent 初始化解耦到每轮推理时。

**5. 消息类型化 → debate 通信语义**

AutoGen 使用强类型消息（`TextMessage`、`HandoffMessage`、`ToolCallMessage` 等），每种消息携带语义上下文（`source`、`content`、`context`）。cryptotrader-ai debate 节点目前以自由文本传递 bull/bear 立场，可引入类似类型（如 `DebateMessage(stance, score, reasoning)`）提升 verdict 节点的解析稳定性。

---

## Notes / Open Questions

- AutoGen 已进入维护期（2025-10 公告）；Microsoft Agent Framework 是其继任者，但 MAF 文档尚不完整。本研究的设计模式在两者中均适用。
- `ListMemory` 的 `update_context()` 具体注入位置（UserMessage 前缀 vs 追加到历史末尾）在公开文档中描述模糊，需要阅读源码 `python/packages/autogen-ext/src/autogen_ext/memory/` 确认——Phase 2 补充。
- Workbench 协议的完整 API 未在本次 Phase 1 中读取，仅从 `_assistant_agent.py` 推断接口形状——Phase 2 补充。
- AutoGen 的 `Society of Mind Agent`（`_society_of_mind_agent.py`）内嵌了 Team，可能与 cryptotrader-ai 的 debate subgraph 有结构相似性——Phase 2 值得精读。
- 框架本身**无 Skill 进化机制**——这是预期内的结论：AutoGen 定位是编排框架，不是学习系统。其价值在于提供了清晰的提示组装 + 记忆注入工程范式，而非进化算法。

---

## 摘要（80-150 字）

对 cryptotrader-ai spec 017 最具参考价值的单一模式是 **四槽位分离**：AutoGen 将系统提示（静态角色）、工具列表（function calling schema）、记忆注入（per-turn update_context 钩子）、对话历史（ChatCompletionContext 变体）四条信息流完全解耦，各自独立管理。这与 D-PA-01 的命名区段方案直接对应，是提示外化设计的工业级参考实现。

AutoGen **不具备任何技能进化概念**——工具是静态注册的，记忆层无学习机制，无评分、无版本演化。AutoGen 是纯粹的编排框架，其对 spec 016 的贡献仅限于提示组装与记忆注入的工程规范，不涉及进化算法设计。Phase 2 的"Evolution Algorithm / Skill Data Structure / Retrieval"条目需从其他项目（SkillClaw、EvoSkills、OpenClaw-RL）借鉴。

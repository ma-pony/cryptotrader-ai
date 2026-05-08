---
name: microsoft/autogen
url: https://github.com/microsoft/autogen
license: MIT（代码）/ CC-BY-4.0（文档）
tier: 2
last_accessed: 2026-05-08
phase_1_complete: true
phase_2_complete: true
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

---

## Phase 2：进化算法

### 结论：N/A — AutoGen 是纯编排框架，无任何内置进化机制

AutoGen 框架的定位从未包含"技能演化"。工具（`FunctionTool`）在 agent 初始化时静态注册，运行时不可增删、不会被评分、不存在版本迭代。框架没有任何以下机制：

- **自动改写提示**：系统提示 `_system_messages` 在整个生命周期中不变，框架不提供修改接口
- **工具淘汰 / 晋升**：`_tools` 列表不记录调用成功率或失败率
- **跨会话强化学习**：记忆层（`Memory`）是读写存储，不是学习器；`ListMemory` 按追加顺序存储，无任何梯度更新或规则归纳
- **Chat history compression**：截断策略（`BufferedChatCompletionContext`、`TokenLimitedChatCompletionContext`）是纯丢弃，不做摘要或经验提炼

### Microsoft Agent Framework（MAF）是否引入进化？

2025-10 MAF 公开预览版（AutoGen + Semantic Kernel 合并）的文档（截至 2026-05）中未出现技能演化语义。MAF 主要在以下维度升级 AutoGen：进程外 agent 隔离、gRPC 通信层、企业级可观察性。演化能力仍需由应用层外挂。

**结论**：本章节 N/A。演化算法应从 SkillClaw、EvoSkills、OpenClaw-RL 等学习型系统借鉴。

---

## Phase 2：技能数据结构

AutoGen 没有"技能（Skill）"一等对象，最接近的替代品是 **Tool 层级**（`BaseTool` → `FunctionTool`）和 **Workbench 容器**。

### BaseTool 抽象结构

`python/packages/autogen-core/src/autogen_core/tools/_base.py`（核心类）：

```
BaseTool[ArgsT, ReturnT]
├── _name: str                          # 工具标识符（OpenAI function name）
├── _description: str                   # 传入模型的说明文本
├── _args_type: type[ArgsT]             # Pydantic 模型，定义输入 schema
├── _return_type: type[ReturnT]         # 输出类型（用于序列化）
├── schema: ToolSchema                  # 计算属性，生成 OpenAI-style JSON Schema
│   ├── name: str
│   ├── description: str | None
│   └── parameters: ParametersSchema
│       ├── type: "object"
│       ├── properties: Dict[str, ...]
│       ├── required: List[str]
│       └── additionalProperties: bool
└── run(args: ArgsT, cancellation_token) → ReturnT   # 异步，抽象方法
```

`run_json()` 是公开入口：接收 `Dict`，用 Pydantic `model_validate()` 校验参数，执行 `run()`，同时触发 `trace_tool_span()` 和 `ToolCallEvent` 两个可观察性钩子。

### FunctionTool：从 Python 函数自动推导 schema

`FunctionTool` 是 `BaseTool` 的主要具体实现。它接收一个 `Callable`（普通函数或协程）和 `description` 字符串，通过 `inspect` 模块提取类型注解，自动生成 `_args_type`（合成 Pydantic 模型）。开发者无需手写 JSON Schema，工具的文档字符串和类型注解即完整规格。

```python
# 注册方式示例（AssistantAgent.__init__）：
FunctionTool(
    func=get_market_price,      # async def get_market_price(symbol: str) -> float
    description="查询实时价格"
)
# → 自动生成 schema.parameters.properties = {"symbol": {"type": "string"}}
```

### Workbench：工具集合容器

`Workbench` 是工具集的抽象容器（`python/packages/autogen-core/src/autogen_core/tools/_workbench.py`）：

| 方法 | 签名 | 语义 |
|------|------|------|
| `list_tools()` | `async → List[ToolSchema]` | 返回当前可用工具列表（可动态变化） |
| `call_tool()` | `async (name, arguments, cancellation_token, call_id) → ToolResult` | 按名称执行工具 |
| `start()` / `stop()` | `async → None` | 生命周期管理 |
| `reset()` | `async → None` | 状态重置 |
| `save_state()` / `load_state()` | `async → Mapping` | 持久化 |

`ToolResult` 封装执行结果，内容为 `ResultContent` 联合类型（文本或图像）。

`AssistantAgent` 在 `_call_llm()` 中的工具汇总逻辑：
```
tools = list(self._tools)
for wb in self._workbench:
    tools += [FunctionTool.from_schema(s) for s in await wb.list_tools()]
tools += list(self._handoff_tools)
```

**关键观察**：`list_tools()` 的动态性意味着 Workbench 可以在不重启 agent 的情况下增删工具——这是 AutoGen 中最接近"热更新技能集合"的机制，但 Workbench 本身没有语义分级、评分或检索；它是一个枚举接口，不是语义目录。

### 与 cryptotrader-ai 的对应关系

| AutoGen 概念 | cryptotrader-ai 等价物 | 差距 |
|---|---|---|
| `FunctionTool` | 各 agent 内的工具函数 | autogen 有 Pydantic schema 校验；我们是裸函数 |
| `Workbench.list_tools()` | 无对应物（工具硬编码到 agent） | 我们缺少动态工具目录层 |
| `_name` / `_description` | 无结构化元数据 | 我们的工具无独立 schema 对象 |
| `ToolCallEvent` 遥测 | `@node_logger()` 装饰器 | 功能相近，粒度不同 |

---

## Phase 2：检索机制

### Memory.query() 接口精确语义

`autogen_core.memory.Memory` 的 `query` 方法签名：

```python
async def query(
    self,
    query: str | MemoryContent,
    cancellation_token: CancellationToken | None = None,
    **kwargs: Any
) -> MemoryQueryResult
```

`MemoryQueryResult` 仅包含 `results: List[MemoryContent]`，无排序分数字段——排序/过滤由具体实现决定，抽象接口不暴露相似度值。

### ListMemory（内置，无向量检索）

`ListMemory` 是唯一随框架附带的 `Memory` 实现（`autogen-core` 包内）。其 `update_context()` 算法：

1. 从 `ChatCompletionContext` 提取最后一条消息文本作为 "query"
2. 将 **全部存储条目**按时间顺序返回（无过滤、无相似度排序）
3. 格式化为文本后，以 `UserMessage` 形式追加到 `model_context` 末尾

这等价于"把记忆本全文贴入上下文"，适合小规模（数十条）场景。

### ChromaDB 实现（autogen-ext，向量检索）

`python/packages/autogen-ext/src/autogen_ext/memory/chromadb/` 提供了真正的语义检索，是 AutoGen 的推荐生产记忆后端：

**Embedding 生成**：延迟初始化，支持四种策略：
- `DefaultEmbeddingFunction`（ChromaDB 内置）
- `SentenceTransformer`（本地模型）
- `OpenAI`（API 调用）
- 自定义 Callable

**相似度算法**：
```
cosine 度量：score = 1.0 - (distance / 2.0)
其他度量：  score = 1.0 / (1.0 + distance)
```

支持 `score_threshold` 过滤低质量结果；返回 `k`（默认配置值）条最相关内容。

**`update_context()` 注入位置**：ChromaDB 实现将结果注入为 **`SystemMessage`**（而非 `UserMessage`），文本格式为 `"Relevant memory content:\n{results}"`。这与 `ListMemory` 的 `UserMessage` 注入位置不同——这个不一致是框架设计上的遗留问题，两种实现均通过 `model_context.add_message()` 写入。

**其他后端**：`autogen-ext/memory/` 下还有 `mem0/`（Mem0 托管服务）和 `redis/`（Redis 向量存储），接口与 ChromaDB 实现相同。

### 检索触发时机

无论哪种 `Memory` 实现，注入都发生在 `AssistantAgent.on_messages_stream()` 的固定位置：

```
1. 接收新消息 → 写入 model_context
2. [for mem in self._memory] await mem.update_context(model_context)  ← 检索 + 注入
3. [for wb in self._workbench] 汇总工具列表
4. _call_llm() → model_client.create()
```

每轮强制重新查询，不缓存。`MemoryQueryEvent` 在查询完成后向调用方广播，用于可观察性。

### 与 cryptotrader-ai 的对应

`verbal_reinforcement()` 节点在 `nodes/data.py` 中硬连线地从 `journal/search.py` 检索历史案例，功能上等价于 AutoGen 的 Memory + ChromaDB 组合。差距在于：AutoGen 将检索封装为可替换的 `Memory` 协议，而我们的检索逻辑与提示组装耦合在同一个节点函数内。

---

## Phase 2：评估

### 内置评估能力：极其有限

AutoGen 核心框架几乎不提供 agent 行为评估工具。以下是已确认的存在：

**1. `max_tool_iterations`（唯一内置限制器）**

`AssistantAgent.__init__` 的 `max_tool_iterations: int = 1` 参数控制单次 `on_messages()` 调用中允许的工具调用循环次数。达到上限后 agent 强制返回当前结果。这不是评估指标，而是防止无限循环的保险丝。

**2. 可观察性事件（用于外部评估）**

AutoGen 在每次 LLM 调用和工具执行时发出强类型事件：
- `ToolCallEvent`：工具名称 + 参数 + 结果（字符串化）
- `ToolCallRequestEvent` / `ToolCallExecutionEvent`：执行前后
- `MemoryQueryEvent`：记忆检索结果
- `ThoughtEvent`：推理模型的思考链

这些事件可被调用方收集用于离线分析，但框架本身不处理。

**3. `agbench`：独立 Benchmark 子包**

`python/packages/agbench/` 是 AutoGen 的专用 benchmark 工具（注意：与 AutoGen 0.1/0.2 兼容，0.4 支持状态待确认）。工作原理：

- 任务以 **JSONL 文件**定义，每条记录对应一个任务实例
- 每次运行在**独立 Docker 容器**中执行，保证环境一致性
- 结果存储在层级目录：`./results/[scenario]/[task_id]/[instance_id]/`
- `agbench tabulate` 命令汇总任务完成率
- 评估指标由用户的 `Scripts/` 分析脚本定义，**没有内置指标库**

**4. 无 LangSmith 原生集成**

AutoGen 不内置与 LangSmith 等外部评估平台的适配器。`autogen-ext` 包提供 OpenTelemetry 导出器（`autogen_ext.telemetry`），可将 span 发送至任何 OTel 兼容后端（Jaeger、Azure Monitor 等），但这是分布式追踪而非 AI 评估语义。

### 结论

AutoGen 的评估能力远弱于其编排能力。框架假设评估是应用层职责：开发者收集事件流、接入自己的评估管线。对 cryptotrader-ai 最有价值的是 **ToolCallEvent 的结构化格式**（工具名 + 参数 + 结果三元组），可作为我们 `@node_logger()` 遥测的参考，用于 debate 轮次的工具调用审计。

---

## Phase 2：Agent ↔ Skill 边界（核心章节）

### 三层抽象的精确接口

AutoGen 的 agent 层级：

#### 1. `BaseChatAgent`（基础协议）

所有 agent 的共同接口（`autogen-agentchat/agents/_base_chat_agent.py`）：

```python
class BaseChatAgent(ABC):
    @property
    def name(self) -> str: ...
    @property
    def description(self) -> str: ...
    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]: ...

    @abstractmethod
    async def on_messages(
        self,
        messages: Sequence[BaseChatMessage],
        cancellation_token: CancellationToken
    ) -> Response: ...

    @abstractmethod
    async def on_messages_stream(
        self,
        messages: Sequence[BaseChatMessage],
        cancellation_token: CancellationToken
    ) -> AsyncGenerator[BaseAgentEvent | BaseChatMessage | Response, None]: ...

    @abstractmethod
    async def on_reset(self, cancellation_token: CancellationToken) -> None: ...

    async def save_state(self) -> Mapping[str, Any]: ...
    async def load_state(self, state: Mapping[str, Any]) -> None: ...
```

**关键约定**：调用方每次只传入"自上次调用以来的新消息"；agent 自行维护历史状态。这是**增量消息协议**，与 LangGraph 的全量状态传递不同。

#### 2. `AssistantAgent`（主要具体实现）

在 `BaseChatAgent` 基础上增加完整初始化参数（见 Phase 1）。核心执行流：

```
on_messages_stream()
├── 1. 将新消息追加到 model_context
├── 2. _update_model_context_with_memory()   # Memory 检索注入
├── 3. 汇总工具列表（_tools + workbench tools + handoff_tools）
├── 4. _call_llm()                           # LLM 推理
│   ├── 组装：_system_messages + model_context.get_messages()
│   └── 调用：model_client.create(messages, tools=tools)
├── 5. 若 LLM 输出 ToolCall → _execute_tool_call() → 循环至 max_tool_iterations
└── 6. yield Response(chat_message, inner_messages)
```

#### 3. `Team`（多 agent 编排层）

`Team` 继承自 `BaseGroupChat`，通过 **发布-订阅主题总线** 编排多个 `BaseChatAgent`：

- `group_topic_{team_id}`：广播频道，所有参与者共享上下文
- `{participant_name}_{team_id}`：点对点频道
- `{manager_name}_{team_id}`：管理器频道，负责路由决策

团队运行接口：
```python
async def run(task, *, cancellation_token=None) -> TaskResult
async def run_stream(task) -> AsyncGenerator[BaseAgentEvent | BaseChatMessage | TaskResult, None]
async def reset() -> None
```

### 三种 Team 的协作协议对比

| 特性 | `RoundRobinGroupChat` | `SelectorGroupChat` | `SwarmGroupChat` |
|------|---------------------|--------------------|--------------------|
| 发言人选择 | 轮询（固定顺序） | LLM 推理选择 | Handoff 消息驱动 |
| 选择逻辑 | `_next_speaker_index % n` | 构造含 `{roles}/{history}/{participants}` 的提示，让 LLM 返回 agent 名 | 扫描最后一条 `HandoffMessage.target` |
| 自定义接口 | 无 | `selector_func` / `candidate_func` | 无（完全显式） |
| 重试/回退 | N/A | `max_selector_attempts` → 上次发言人 | N/A |
| 内部 Team 嵌套 | 支持 | 支持 | **不支持** |
| 最适合的场景 | 固定流水线 | 动态任务路由 | 显式状态机 |

**与 cryptotrader-ai 的结构映射**：

```
cryptotrader-ai 架构          ≈  AutoGen 对应
─────────────────────────────────────────────────────
4 分析 agent（并行 gather）   ≈  RoundRobinGroupChat（但我们实际是 asyncio.gather，无中心调度）
debate_gate → debate_round   ≈  SelectorGroupChat（debate_gate_router 是 selector_func）
SwarmTeam                    ≈  verdict 节点（读最后 HandoffMessage → 最终裁决）
SocietyOfMindAgent           ≈  debate subgraph（内嵌 Team，对外呈现单一响应）
```

### `Memory.update_context()` 精确签名与扩展点

完整方法签名（`autogen_core.memory._base_memory.Memory`）：

```python
@abstractmethod
async def update_context(
    self,
    model_context: ChatCompletionContext
) -> UpdateContextResult
```

`UpdateContextResult` 包含 `memories: MemoryQueryResult`（用于回传给调用方做遥测）。

**扩展点分析**：该方法接收 `ChatCompletionContext` 引用，可以对其执行任意 `add_message()` 调用，插入任何类型的 `LLMMessage`（`SystemMessage`、`UserMessage`、`AssistantMessage` 等）。框架对注入内容和位置无约束——这是一个开放钩子，不是受限接口。

ChromaDB 实现选择注入为 `SystemMessage`；`ListMemory` 实现注入为 `UserMessage`；开发者可以在自定义实现中注入带结构标签的 `UserMessage`（如 `<experience>...</experience>`），完全契合 spec 014 的 `SkillsInjectionMiddleware` 设计意图。

### `ChatCompletionContext` 四种变体的精确算法

`python/packages/autogen-core/src/autogen_core/model_context/` 提供以下实现：

#### UnboundedChatCompletionContext
无截断，`get_messages()` 返回 `list(self._messages)`。适合短会话或 token 预算充足场景。

#### BufferedChatCompletionContext
```
构造参数：buffer_size: int（> 0）
get_messages() 算法：
  1. messages = self._messages[-buffer_size:]   # 滑动窗口，保留最近 N 条
  2. if messages[0] is FunctionExecutionResultMessage:
         messages = messages[1:]                # 移除孤立函数结果
  3. return messages
```
**语义**：纯消息数量截断，不感知 token。适合工具调用频繁、每条消息较短的场景。

#### TokenLimitedChatCompletionContext
```
构造参数：token_limit: int | None，model_client（用于 count_tokens）
get_messages() 算法：
  messages = list(self._messages)
  while True:
      token_count = model_client.count_tokens(messages, tools=tool_schema)
      if token_limit is not None:
          if token_count < token_limit: break
      else:
          if model_client.remaining_tokens(messages) >= 0: break
      middle_index = len(messages) // 2
      messages.pop(middle_index)               # ← 删除中间消息（非最旧）
  if messages[0] is FunctionExecutionResultMessage:
      messages = messages[1:]
  return messages
```
**关键细节**：删除的是**中间位置**消息，而非最旧消息。这保留了最早的对话背景（system 近邻）和最近的消息（当前任务），牺牲中间过渡部分。与 Phase 1 描述的"丢弃最旧"不符——需修正。

#### HeadAndTailChatCompletionContext
```
构造参数：head_size: int（> 0），tail_size: int（> 0）
get_messages() 算法：
  head = self._messages[:head_size]
  # 若 head 末尾是包含 FunctionCall 的 AssistantMessage，移除之（避免断裂调用）
  tail = self._messages[-tail_size:]
  # 若 tail 头部是 FunctionExecutionResultMessage，移除之（避免孤立结果）
  if len(self._messages) <= head_size + tail_size:
      return head ∪ tail  # 无重叠，返回全量
  skipped = len(self._messages) - len(head) - len(tail)
  return head + [UserMessage(f"Skipped {skipped} messages")] + tail
```
**语义**：保留最早的 head_size 条（建立背景）和最近的 tail_size 条（维持连贯），中间用占位消息标记跳过数量。这是最适合**长对话摘要**场景的策略。

### SocietyOfMindAgent：debate subgraph 的结构参考

`SocietyOfMindAgent`（`autogen-agentchat/agents/_society_of_mind_agent.py`）是 AutoGen 中与 cryptotrader-ai debate subgraph 最接近的结构：

- 内嵌一个完整 `Team`（可以是 RoundRobin/Selector/Swarm 任意变体）
- 运行内部 Team，收集所有 agent 的 `BaseAgentEvent` 和 `BaseChatMessage`
- 用 LLM **合成**内部对话为单一 `TextMessage` 输出（合成提示模板："Earlier you were asked to fulfill a request... provide a standalone answer"）
- 对外完全兼容 `BaseChatAgent` 接口，调用方不感知内部 Team 存在

**与 cryptotrader-ai debate 的差异**：
- autogen：内部 Team 结果由 LLM 合成为文本摘要
- cryptotrader-ai：debate 结果由 `verdict` 节点解析结构化评分（`bull_score`, `bear_score`）
- autogen：外部不感知内部 agent 身份
- cryptotrader-ai：verdict 节点需要知道是哪个 agent 持哪个立场

---

## Phase 2：工程实现

### python/packages/ 九个子包职责划分

| 子包 | 核心职责 | 主要依赖 |
|------|----------|----------|
| `autogen-core` | 事件驱动消息总线、`BaseChatAgent` 基础类、`Memory` 协议、`ChatCompletionContext` 变体、`BaseTool` / `Workbench` | Python 3.10+, Pydantic v2 |
| `autogen-agentchat` | 高层 API：`AssistantAgent`、`Team`（RoundRobin/Selector/Swarm）、`SocietyOfMindAgent`、消息类型系统 | `autogen-core` |
| `autogen-ext` | LLM 客户端（OpenAI/Azure/Gemini/Anthropic）、Memory 后端（ChromaDB/Mem0/Redis）、工具集成（Docker、MCP、Web Search）、代码执行器 | `autogen-core`、各第三方库 |
| `autogen-studio` | 无代码 GUI（React + FastAPI），可视化构建 agent 团队 | `autogen-agentchat`、React |
| `autogen-magentic-one` | Magentic-One 参考实现：WebSurfer + FileSurfer + Coder + Executor 多 agent 团队 | `autogen-agentchat` |
| `magentic-one-cli` | Magentic-One 的命令行入口 | `autogen-magentic-one` |
| `agbench` | Benchmark 框架（与 AutoGen 0.1/0.2 兼容，0.4 支持待确认） | Docker、独立运行 |
| `autogen-test-utils` | 测试辅助工具（mock model client、消息工厂等） | `autogen-core` |
| `component-schema-gen` | 从 Python 类生成 Component JSON Schema（用于 Studio 可视化配置） | `autogen-core` |
| `pyautogen` | 遗留包（v0.2 入口），重定向至新包，不含新功能 | `autogen-agentchat` |

### 包边界与版本策略

`autogen-core` 是框架的稳定核心：Memory、Tool、Workbench、ChatCompletionContext 等可扩展协议均定义于此，且保证向后兼容。`autogen-agentchat` 和 `autogen-ext` 以 `autogen-core` 为基础扩展，可独立版本化。

这种分层允许：开发者只依赖 `autogen-core` 实现自定义 agent，无需引入 `autogen-agentchat` 的高层抽象——精细依赖管理是大型项目的标准做法。

### .NET 实现

`dotnet/` 下的 C# 实现功能覆盖：
- **已实现**：对话 agent（ConversableAgent）、函数调用、代码执行（`dotnet-interactive`）、双 agent 对话、GroupChat
- **未完成**：Enhanced LLM Inferences（对应 Python 的 prompt caching / streaming 优化）
- **独特优势**：C# Source Generator 在编译时生成类型安全的函数定义，消除 Python 版中运行时 `inspect` 调用的开销
- **包策略双轨**：旧版 `AutoGen.*`（稳定）与新版 `Microsoft.AutoGen.*`（API 不稳定，对齐 MAF）并存

与 Python 的**同步机制**：.NET 不同步 Python 版本的功能——两者共享架构理念（三层抽象、事件驱动、增量消息协议）但独立实现，无自动代码共享。

### Microsoft Agent Framework（MAF）迁移路径

2025-10 MAF 公开预览标志着 AutoGen 进入维护期。对于已有 AutoGen 代码库的影响：

- **继续工作**：`BaseChatAgent`、`AssistantAgent`、所有 `Team` 类型、`Memory` 协议在 MAF 中保留兼容接口
- **主要变化**：MAF 引入进程外 agent（gRPC 通信）、企业认证集成、与 Semantic Kernel PromptTemplate 的互操作
- **迁移优先级**：新功能（自动 Handoff 检测、多语言 agent）只在 MAF 提供；bug fix 和安全补丁仍在 AutoGen
- **时间线**：MAF 1.0 GA 目标 2026 Q1，稳定 API 发布前不建议生产迁移

### 可观察性体系

AutoGen 0.4 在可观察性上有显著投入：
- **OpenTelemetry**：通过 `autogen_ext.telemetry` 导出 span，支持 Azure Monitor、Jaeger 等后端
- **事件流**：`on_messages_stream()` 为每个中间步骤 yield 强类型事件（工具调用、记忆查询、推理块）
- **`ToolCallEvent` 结构**：`{tool_name: str, arguments: Dict, result: str}` 三元组，标准化工具调用审计格式

这套体系与 cryptotrader-ai 的 `MetricsCollector` + `/metrics` 端点（Architecture Review 2026-03-15 引入）定位相同，但 AutoGen 粒度更细（精确到单次工具调用），我们目前粒度在节点级别。

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
- **[Phase 2 已确认]** `ListMemory.update_context()` 注入位置：追加到历史末尾，类型为 `UserMessage`。ChromaDB 实现注入为 `SystemMessage`。两者不一致，是框架遗留问题。
- **[Phase 2 已确认]** Workbench 精确接口：`list_tools() → List[ToolSchema]`，支持动态变化。`call_tool(name, arguments, cancellation_token, call_id) → ToolResult`。生命周期：`start()/stop()/reset()/save_state()/load_state()`。
- **[Phase 2 已确认]** `SocietyOfMindAgent` 确与 debate subgraph 高度同构：内嵌 Team + LLM 合成输出。关键差异：autogen 合成为文本摘要；我们需要结构化评分——不可直接照搬，需适配。
- **[Phase 2 已确认]** `TokenLimitedChatCompletionContext` 删除**中间位置**消息，非最旧消息——Phase 1 描述有误，已在 Phase 2 章节修正。
- **[Phase 2 已确认]** 存在第四种变体 `HeadAndTailChatCompletionContext`（Phase 1 遗漏）：保留 head_size 条最早消息 + tail_size 条最近消息，中间插入占位符。
- 框架本身**无 Skill 进化机制**——这是预期内的结论：AutoGen 定位是编排框架，不是学习系统。其价值在于提供了清晰的提示组装 + 记忆注入工程范式，而非进化算法。
- `agbench` 与 AutoGen 0.4 的兼容性未官方确认（README 明确标注兼容 0.1/0.2）。在 Phase 2 调研中未找到 0.4 适配的证据。

---

## 摘要（Phase 2 更新版）

对 cryptotrader-ai 4-agent + debate + verdict 架构最该借鉴的 AutoGen 抽象是 **`SocietyOfMindAgent` 模式**：内嵌 Team（对应我们的 debate subgraph）+ 对外呈现单一 `BaseChatAgent` 接口，这使 debate 节点对上层图透明，且 `BaseChatAgent` 的增量消息协议（调用方只传新消息，agent 自维护状态）可直接对应 LangGraph 节点间的增量状态更新语义。四槽位分离（系统提示 / 工具 / 记忆注入 / 历史）仍是最高优先级的提示组装范式参考。

**Anthropic prompt cache 场景**的最优 `ChatCompletionContext` 变体是 **`HeadAndTailChatCompletionContext`**：head 保留系统提示附近的最早消息（对应 Anthropic cache breakpoint 的稳定前缀），tail 保留最近消息（对应当前轮次输入），中间部分可对接 prompt cache 的缓存边界——从而最大化缓存命中率。`TokenLimitedChatCompletionContext` 的中间删除策略会破坏 Anthropic cache 要求的"前缀稳定性"，不适合。

**`Memory.update_context()` 钩子可直接对应 spec 014 的 `SkillsInjectionMiddleware`**：两者语义完全同构——在每次 LLM 调用前、接收 `ChatCompletionContext` 引用、可注入任意 `LLMMessage`。区别是 AutoGen 注入的是检索到的记忆内容，spec 014 注入的是匹配的技能 prompt 区段。可以将 `SkillsInjectionMiddleware` 实现为 `Memory` 协议的子类，直接插入 AssistantAgent 的 `memory=` 参数，无需修改 agent 核心代码。

AutoGen **完全没有技能进化机制**（进化算法章节 N/A）。进化能力须从 SkillClaw / EvoSkills / OpenClaw-RL 借鉴。

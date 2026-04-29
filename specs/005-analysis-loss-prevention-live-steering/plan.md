# 技术实施方案：分析防丢失 + Live Steering

**Feature Branch**: `005-analysis-loss-prevention-live-steering`
**对应 Spec**: `specs/005-analysis-loss-prevention-live-steering/spec.md`
**编写日期**: 2026-04-17

---

## 技术上下文

### 现有 `POST /api/chat/stream` 实现分析

`src/api/routes/chat.py` 当前实现：

```
ChatStreamRequest → _run_chat_stream() 生成器
  → build_trading_graph() + build_initial_state()
  → run_graph_traced(graph, state)     # 同步等待，全部完成才返回
  → 逐节点 tool_call + 文字摘要
StreamingResponse 直接包装生成器协程
```

**核心问题**：

- `StreamingResponse` 生命周期与 HTTP 连接强耦合。客户端断连 → FastAPI 取消生成器协程 → `run_graph_traced()` 被中止 → 已完成 90% 的分析结果永久丢失。
- `run_graph_traced()` 以 `graph.astream(stream_mode="updates")` 顺序迭代，返回 `(final_state, node_trace)` 元组，不向外发布细粒度事件。
- 事件类型只有 6 种（`session`/`message_start`/`content_delta`/`tool_call`/`message_end`/`done`），缺乏节点/Agent 粒度进度信息。
- 无 steering 注入点、无中断信号通道。

### 现有基础设施可复用项

| 组件 | 位置 | 可复用方式 |
|------|------|-----------|
| `RedisStateManager` | `src/cryptotrader/risk/state.py` | 继承/扩展，增加 List 操作和 Pub/Sub |
| `_MemoryStore` | `src/cryptotrader/risk/state.py` | 扩展支持 `list` 操作，用于 Redis 不可用时的完整降级 |
| `TaskRegistry` | `src/cryptotrader/task_registry.py` | 已有 `add_background_task()` 防 GC，`BackgroundTaskManager` 在其上叠加会话注册表 |
| `node_logger()` 装饰器 | `src/cryptotrader/tracing.py` | 已为所有节点提供结构化日志，`wrap_node_with_events()` 与其共存 |
| `AppConfig` / TOML | `src/cryptotrader/config.py` | 新增 `ChatConfig` 数据类，`default.toml` 追加 `[chat]` 段 |
| `streamFetch()` | `web/src/lib/stream-fetch.ts` | 已读取 `id:` 字段，已有 `SSEError(status)`；只需补充 410 分支处理 |
| `useChatMessages()` | `web/src/hooks/use-chat-messages.ts` | 扩展 `handleEvent switch`，新增 15 种事件类型分支 |

### `_run_agent()` 调用链分析（Live Steering 注入点）

```
nodes/agents.py：run_agents() → asyncio.gather(4 × _run_agent())
_run_agent(agent_id, state, model, timeout)
  → agent.analyze(snapshot, experience, ...)  # BaseAgent.analyze()
    → build_prompt()
    → llm.ainvoke(messages)                   # LLM 调用
    → _parse_response()
```

Steering 最小侵入注入点：`_run_agent()` 在 `agent.analyze()` 调用前，读取 `state["metadata"].get("steering_queue", {}).get(agent_id)` 并追加到 `experience` 字符串末尾。

---

## 架构决策

### 决策 1：后台任务管理器（BackgroundTaskManager）设计

**选择**：新增 `src/cryptotrader/chat/task_manager.py`，封装单例 `BackgroundTaskManager`。

核心行为：

- 复用 `task_registry.add_background_task()` 防 GC 回收；在其上叠加会话注册表（`dict[str, AnalysisTask]`）和并发上限检查。
- 分析协程（`_run_analysis_and_buffer()`）创建后独立运行，HTTP 连接断开**不取消**后台 Task（Task 已被 `TaskRegistry` 强引用）。
- `POST /api/chat/stream` 改造为两阶段：① 创建/查找后台 Task；② 返回 `StreamingResponse`，SSE 生成器仅从 Redis List 消费事件并转发。

**拒绝方案**：LangGraph 原生 checkpoint/interrupt（`MemorySaver` + `interrupt_before`）侵入式改造大，且 LangGraph interrupt 语义是"挂起等待人工输入"，不匹配"软中断出快速裁决"需求。

### 决策 2：事件缓冲方案（Redis List + 内存降级）

**选择**：`analysis:events:{session_id}` 作为 Redis List；每个元素为序列化后的 SSE 信封 JSON。

- 扩展 `RedisStateManager`，新增 `buffer_push`/`buffer_range`/`buffer_len`/`buffer_set_ttl` 方法；`_MemoryStore` 扩展 `lists: dict[str, list[str]]`，实现等价的内存降级。
- per-session 事件 ID 由 `INCR analysis:event_seq:{session_id}` 生成（原子操作）；内存降级时使用线程安全计数器。
- 超出 `event_buffer_max_size`（默认 500 条）时，`LTRIM` 丢弃最旧事件，并在流中插入 `buffer_overflow` 警告事件。
- TTL 默认 300 秒（`event_buffer_ttl_seconds`），可通过 `[chat]` 配置调整。

### 决策 3：结构化事件总线（EventBus）

**选择**：新增 `src/cryptotrader/chat/event_bus.py`，`EventBus` 持有 `session_id`、`EventBuffer` 引用和进程内 `asyncio.Queue`（实时推送给当前连接的 SSE 生成器）。

- `publish(event_type, data)` 方法：① 生成 `event_id`；② 构造 `SSEEnvelope`；③ 写入 Redis List；④ 同时写入 `asyncio.Queue`（用于实时转发）。
- 节点包装层（`src/cryptotrader/chat/node_events.py`）：`wrap_node_with_events(fn, event_bus)` 在节点执行前后分别发布 `node_started`/`node_done`，与 `@node_logger()` 共存（职责互补：logger 写文件，event_bus 发 SSE）。
- Agent 节点在 `_run_agent()` LLM 调用前后发布 `agent_thinking`/`agent_analysis`，不依赖 `astream` chunk 顺序。

**方案对比**：修改 `tracing.py` 可行但职责混乱；新增 `EventBus` 使事件发布与追踪解耦，单独可测。

### 决策 4：Soft Interrupt 实现（asyncio.Event + 节点间检查）

**选择**：每个 `AnalysisTask` 持有 `interrupt_event: asyncio.Event`；分析协程在 LangGraph `astream` 每个 chunk 之后检查标志。

- 中断生效后：① 已完成 Agent ≥ 1 时，调用 `_make_partial_verdict()` 生成 `verdict_partial`（发布到事件流）；② 完成 Agent < 1 时，发布 `interrupt_rejected`。
- `POST /api/chat/interrupt/{session_id}` 调用 `interrupt_event.set()`，不 `cancel()` Task——保证运行中的 LLM 调用自然完成。
- LangGraph 4 Agent 并行执行时（`asyncio.gather`），中断检查在 gather 完成后生效，即"不强杀运行中 LLM 调用"——与 spec FR-012 完全一致。

### 决策 5：Live Steering 注入点（`_run_agent()` 最小侵入）

**选择**：在 `src/cryptotrader/nodes/agents.py` 的 `_run_agent()` 中，`agent.analyze()` 调用前读取 steering 队列。

- steering 队列键：`steering:{session_id}:{agent_id}`（Redis List，LRANGE + DEL 原子清空）。
- `_run_agent()` 通过 `state["metadata"].get("event_bus")` 获取 `EventBus` 引用；steering 指令追加到 `experience` 末尾（system 上下文扩展）。
- `POST /api/chat/steer/{session_id}`：写入 steering 队列，根据 Agent 是否已在 `completed_agents` 中返回 `steer_queued` 或 `steer_too_late`。
- 单条指令超 500 字符时截断，发出 `steer_truncated` 警告；截断后的指令仍入队。

**设计约束**：`state["metadata"]` 在 LangGraph 中通过 `merge_dicts` 传播（非序列化字段），可安全注入 `EventBus` 引用。

### 决策 6：断线重连协议

**选择**：`POST /api/chat/stream` 通过请求体 `last_event_id` 字段区分新连接与重连。

- 重连时：① 检查 `analysis:events:{session_id}` 是否存在，不存在返回 `410 Gone`；② 存在则发出 `stream_resume` 事件；③ 从 Redis List 线性扫描找到 `last_event_id` 之后的事件批量回放（O(n)，n≤500 可接受）；④ 回放完成后若任务仍在运行，附接实时 `asyncio.Queue`。
- SSE 协议 `id:` 字段与 `event_id` 保持一一对应，前端 `Last-Event-ID` 头和请求体 `last_event_id` 字段均可触发重连。

### 决策 7：Watch 端点（Redis Pub/Sub + 进程内降级）

**选择**：`GET /api/chat/watch` 订阅 Redis Pub/Sub `analysis:new_workflow` 频道；Redis 不可用时降级为进程内 `asyncio.Queue` 广播。

- `BackgroundTaskManager.create()` 在启动后台 Task 后，异步 `PUBLISH analysis:new_workflow <json>`。
- Watch 端点维护模块级 `_watch_subscribers: set[asyncio.Queue]`（进程内，Redis 降级时使用）。
- 多个 watch 客户端同时连接时，Redis Pub/Sub 天然广播；内存模式通过 `for q in _watch_subscribers: q.put_nowait(msg)` 广播。

### 决策 8：前端最小改动原则

**选择**：不重写 `streamFetch`，仅在 `useChatMessages` 扩展事件处理，新增独立 `useAnalysisProgress` hook。

- `stream-fetch.ts`：补充 `SSEError.status === 410` 分支（`retryable = false`）；其余无需改动。
- `useChatMessages.ts`：`handleEvent` 的 `switch` 扩展新事件类型；原有 6 种类型不变（向后兼容）。
- 新增 `useAnalysisProgress.ts`：维护管线节点状态、Agent 卡片状态、断线重连逻辑（持久化 `last_event_id` 到 `sessionStorage`）。
- 新增 `AnalysisProgressPanel`、`AgentCard`、`VerdictCard`、`NodeProgressBar` 组件。

---

## 文件结构（新增/修改）

### 新增文件

```
src/
  cryptotrader/
    chat/
      __init__.py                      # 公开 BackgroundTaskManager, EventBus, EventBuffer
      task_manager.py                  # BackgroundTaskManager + AnalysisTask dataclass
      event_bus.py                     # EventBus（SSEEnvelope 序列化、publish、asyncio.Queue）
      event_buffer.py                  # EventBuffer（Redis List + _MemoryStore 降级）
      node_events.py                   # wrap_node_with_events() 装饰器
      partial_verdict.py               # _make_partial_verdict() 基于已完成 Agent 生成快速裁决

  api/
    routes/
      chat_control.py                  # POST /interrupt/{session_id}
                                       # POST /steer/{session_id}
                                       # GET  /watch

config/
  default.toml                         # 新增 [chat] 段落（见数据模型）

web/
  src/
    hooks/
      use-analysis-progress.ts         # 分析进度状态 hook（节点/Agent/辩论/裁决状态机）
    components/
      analysis/
        AnalysisProgressPanel.tsx      # 管线进度面板（顶层容器）
        AgentCard.tsx                  # 单个 Agent 分析卡片（含 SteeringInput）
        VerdictCard.tsx                # 裁决卡片（支持 is_partial 警示横幅）
        NodeProgressBar.tsx            # 节点水平时间线进度条
        DebateStatus.tsx               # 辩论轮次状态指示器
    types/
      analysis-events.ts               # AnalysisEventType 联合类型 + SSEEnvelope 接口

tests/
  test_chat_task_manager.py            # BackgroundTaskManager 单元测试（创建/并发上限/完成清理）
  test_chat_event_bus.py               # EventBus 单元测试（publish/序列化/asyncio.Queue）
  test_chat_event_buffer.py            # EventBuffer 单元测试（Redis + 内存降级 + TTL + 溢出）
  test_chat_partial_verdict.py         # _make_partial_verdict() 快速裁决逻辑测试
  test_chat_reconnect.py               # 断线重连集成测试（SC-001 ~ SC-003）
  test_chat_interrupt.py               # Soft Interrupt 集成测试（SC-004）
  test_chat_steering.py                # Live Steering 集成测试（SC-005）
  test_api_chat_control.py             # 新 API 端点认证 + 响应格式测试
```

### 修改文件

```
src/
  cryptotrader/
    config.py                          # 新增 ChatConfig dataclass；AppConfig 新增 chat: ChatConfig
    risk/state.py                      # RedisStateManager 新增 buffer_push/buffer_range/buffer_len
                                       # _MemoryStore 扩展 lists dict 支持 list 操作
    nodes/agents.py                    # _run_agent() 新增 steering 队列读取 + event_bus 发布
    tracing.py                         # run_graph_traced() 接受可选 event_bus 参数，注入节点包装

  api/
    routes/chat.py                     # 重构为后台任务模式 + last_event_id 断线重连
    main.py                            # 注册 chat_control router；lifespan 初始化 BackgroundTaskManager

config/
  default.toml                         # [chat] 段落

web/
  src/
    hooks/use-chat-messages.ts         # 扩展 handleEvent 处理新 15 种事件类型
    lib/stream-fetch.ts                # 补充 410 Gone 错误处理分支
    pages/chat/index.tsx               # 嵌入 AnalysisProgressPanel；传入 sessionId
```

---

## 数据模型

### Python 数据模型

```python
# src/cryptotrader/config.py

@dataclass
class ChatConfig:
    event_buffer_ttl_seconds: int = 300          # FR-004
    max_concurrent_tasks: int = 10               # FR-006
    max_steering_instruction_chars: int = 500    # FR-020
    event_buffer_max_size: int = 500             # 边界条件
```

```python
# src/cryptotrader/chat/task_manager.py

@dataclass
class AnalysisTask:
    session_id: str
    pair: str
    trigger_source: str               # "manual" | "scheduler" | "api"
    task: asyncio.Task
    interrupt_event: asyncio.Event
    event_bus: "EventBus"
    created_at: float                 # time.monotonic()
    completed: bool = False
    completed_agents: list[str] = field(default_factory=list)
```

```python
# src/cryptotrader/chat/event_bus.py

@dataclass
class SSEEnvelope:
    event_id: int                     # per-session 自增，用于断线重连定位
    type: str                         # 见 FR-022 事件类型列表
    ts: str                           # ISO8601 UTC
    session_id: str
    data: dict
```

### Redis 键规范

| 键 | 类型 | TTL | 说明 |
|----|------|-----|------|
| `analysis:events:{session_id}` | List | 300s（可配置） | SSE 事件帧缓冲，每条为序列化 SSEEnvelope JSON |
| `analysis:event_seq:{session_id}` | String (int) | 300s | per-session 事件 ID 原子计数器 |
| `analysis:status:{session_id}` | String (JSON) | 300s | 任务元数据：pair, trigger_source, state（running/done/error）|
| `steering:{session_id}:{agent_id}` | List | 300s | steering 指令队列，LRANGE + DEL 原子读取 |
| `analysis:new_workflow` | Pub/Sub 频道 | — | 新任务广播：`{session_id, pair, trigger_source}` |

### `config/default.toml` 新增段落

```toml
[chat]
event_buffer_ttl_seconds = 300
max_concurrent_tasks = 10
max_steering_instruction_chars = 500
event_buffer_max_size = 500
```

### TypeScript 类型定义

```typescript
// web/src/types/analysis-events.ts

export type AnalysisEventType =
  | 'session_start' | 'stream_resume' | 'stream_done' | 'stream_error' | 'session_replaced'
  | 'node_started' | 'node_done'
  | 'agent_thinking' | 'agent_analysis'
  | 'debate_started' | 'debate_round_done'
  | 'verdict_ready' | 'verdict_partial' | 'risk_checked'
  | 'checkpoint_saved' | 'interrupt_rejected'
  | 'steer_queued' | 'steer_too_late' | 'steer_truncated'
  | 'buffer_overflow' | 'new_workflow';

export interface SSEEnvelope<T = unknown> {
  event_id: number;
  type: AnalysisEventType;
  ts: string;
  session_id: string;
  data: T;
}

// 向后兼容：原有 6 种类型继续保持
export type LegacyEventType =
  | 'session' | 'message_start' | 'content_delta' | 'tool_call' | 'message_end' | 'done';
```

---

## API 设计

### 改造现有端点

**`POST /api/chat/stream`**（扩展 `ChatStreamRequest`）

```
新增请求字段：
  last_event_id:  int | null  — 断线重连起点（对应 SSE id: 字段）

响应行为：
  新任务：推送 session_start → 创建后台 Task → SSE 实时转发
  重连（last_event_id 非 null）：推送 stream_resume → 批量回放历史 → 附接实时流
  任务过期：HTTP 410 Gone
  并发超限：HTTP 429 Too Many Requests
  session_id 碰撞（新任务覆盖旧任务）：旧连接收到 session_replaced → 断开
```

### 新增端点

**`POST /api/chat/interrupt/{session_id}`**

```
认证：X-API-Key
成功响应：200 + SSEEnvelope（type: "checkpoint_saved" 或 "interrupt_rejected"）
错误响应：404（session_id 不存在）
```

**`POST /api/chat/steer/{session_id}`**

```
请求体：{ target: str, instruction: str }
响应：200 + SSEEnvelope（type: "steer_queued" | "steer_too_late"）
       + 可选 SSEEnvelope（type: "steer_truncated"，若超长）
错误：404（session_id 不存在）
```

**`GET /api/chat/watch`**

```
认证：X-API-Key
响应类型：text/event-stream（长连接）
事件：new_workflow { session_id, pair, trigger_source }
降级：Redis 不可用时进程内 asyncio.Queue 广播
```

---

## 向后兼容性保证

| 变更点 | 现有调用方 | 保证 |
|--------|-----------|------|
| `ChatStreamRequest` 新增 `last_event_id` | 所有现有前端调用 | `last_event_id: int \| None = None` 默认值，不传时行为与当前完全相同 |
| 原有 6 种 SSE 事件类型 | `useChatMessages` switch、前端集成测试 | 原有生成路径保留（`_run_chat_stream_legacy()` 兼容层），FR-023 过渡期内并存 |
| `RedisStateManager` 新增方法 | 所有风控调用方 | 仅新增方法，不修改现有 `get/set/incr/expire/delete` 签名 |
| `_run_agent()` 新增参数 | `run_agents()` 调用方 | `event_bus: EventBus | None = None` 可选参数；无 event_bus 时行为不变 |
| `run_graph_traced()` 新增参数 | `chat.py`、`scheduler.py`、`backtest/engine.py` | `event_bus: EventBus | None = None` 可选参数；不传时返回值格式不变 |
| `AppConfig.chat` 新字段 | `load_config()` | `chat: ChatConfig = field(default_factory=ChatConfig)`，无 `[chat]` 段时使用全部默认值 |
| 新增 `/api/chat/interrupt`、`/steer`、`/watch` 端点 | 无现有调用 | 纯新增，不影响现有路由 |

---

## 依赖变更

### 后端（`pyproject.toml`）

无需新增外部依赖：

- `redis>=7.3`：已有，`redis.asyncio` 已原生支持 `lpush`/`lrange`/`ltrim`/`publish`/`subscribe`。
- `asyncio`（标准库）：`asyncio.Event`、`asyncio.Queue`、`asyncio.Task` 均已使用。
- `pydantic>=2.12`：已有，用于 `SteerRequest`、`InterruptResponse` 等新请求/响应模型。
- `fakeredis`：测试环境可选（若不安装则集成测试使用内存降级路径）。

### 前端（`web/package.json`）

无需新增依赖：

- React 19 + Zustand + React Query 5 已满足所有 UI 状态管理需求。
- `sessionStorage` 为浏览器原生 API，无需额外库。

---

## 风险与缓解

| 风险 | 概率 | 影响 | 缓解方案 |
|------|------|------|----------|
| **Redis OOM**：高频分析导致事件积压 | 低 | 中 | 单任务最多 500 条事件，超限 FIFO 丢弃（`LTRIM`）+ `buffer_overflow` 警告事件 |
| **后台 Task 内存泄漏**：完成后未从注册表清除 | 低 | 高 | `add_done_callback` 自动清理；24 小时压测验证（SC-008） |
| **steering 竞态**：Agent LLM 调用与 steering 入队同时发生 | 中 | 低 | `_run_agent()` 在 `asyncio.wait_for()` 之前同步读取队列；迟到指令返回 `steer_too_late` |
| **软中断裁决质量差**：仅 1 个 Agent 完成时触发中断 | 中 | 低 | 最少需要 2 个 Agent 完成才生成 `verdict_partial`；否则返回 `interrupt_rejected`（FR-015） |
| **LangGraph 并行块无法逐 Agent 推送 `agent_thinking`** | 高 | 低 | `agent_thinking` 在 `_run_agent()` 内部（LLM 调用前）直接 `publish`，不依赖 `astream` chunk |
| **event_id 回放精度**：Redis List 索引与自增 event_id 偏差 | 低 | 中 | event_id 存储在每个事件 JSON 内，重连时线性扫描匹配（O(n)，n≤500 可接受，P99<200ms） |
| **前端 410 未处理**：现有 `streamFetch` 缺乏 410 分支 | 低 | 低 | `SSEError.retryable = false` + 专用"分析已过期"提示 |
| **CI 无 Redis 导致测试失败** | 中 | 中 | 单元测试优先走内存降级路径；集成测试可选 `fakeredis`；`pytest.mark.redis` 标记隔离 |
| **`session_id` 碰撞覆盖旧任务** | 低 | 低 | 旧连接收到 `session_replaced` 事件后主动断开；由客户端负责生成 UUID4 |

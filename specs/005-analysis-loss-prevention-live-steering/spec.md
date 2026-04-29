# Feature Specification: 分析防丢失 + Live Steering

**Feature Branch**: `005-analysis-loss-prevention-live-steering`
**Created**: 2026-04-17
**Status**: Draft

---

## 背景与问题陈述

CryptoTrader-AI 的多 Agent 分析管线（`build_trading_graph()`）包含数据收集 → 经验注入 → 4 个并行 Agent → 辩论门 → 最多 2 轮辩论 → AI 裁决 → 风控门，总耗时通常在 **30–60 秒**。

当前实现（`POST /api/chat/stream`）直接绑定 `StreamingResponse`，生成器协程与 HTTP 连接生命周期强耦合：

- **网络抖动 → 分析永久丢失**：客户端断连后，服务端协程立即被 GC，已完成 90% 的分析结果不可恢复。
- **无法中途纠偏**：分析进行中无法向 Agent 注入新指令（如"忽略最近的 FUD 新闻"），只能等待完成后重新发起。
- **全量重试成本极高**：一次完整管线调用消耗数百 token；断线重连只能重新触发整个管线。

本 Feature 旨在以最小侵入性解决上述问题，同时在此基础上实现 **Live Steering**（实时引导）能力。

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 — 断线重连不丢失分析结果（Priority: P0）

作为系统用户，当我正在观察 BTC/USDT 分析流式输出时，若因网络波动导致 SSE 连接中断，我希望在 10 秒内重新连接后能从断点处继续接收事件，而不是看到一片空白或被迫重新等待 30–60 秒。

**Why this priority**：分析结果具有时效性（市场窗口有限），断线导致完整重新分析不仅浪费 token，还可能错过交易时机。这是整个 Feature 的核心价值命题，必须 P0 完成。

**Independent Test**：触发一次 BTC/USDT 分析 → 在分析进行至 50% 时（节点追踪显示至少 2 个 Agent 已完成）手动断开网络 → 5 秒后恢复网络并重新连接（携带 `session_id` 和 `last_event_id`）→ 观察是否从断点处接收到后续事件，最终看到完整裁决。

**Acceptance Scenarios**:

1. **Given** 分析任务进行中（已完成 `collect_data` + 2 个 Agent），**When** 客户端断开 SSE 连接，**Then** 服务端后台任务**继续运行**至完成，所有事件持久化到 Redis List
2. **Given** 客户端携带 `session_id` + `last_event_id` 重连，**When** 服务端收到重连请求，**Then** 回放 `last_event_id` 之后的所有已缓冲事件，随后附接实时事件流
3. **Given** 分析在断线期间已全部完成，**When** 客户端重连，**Then** 所有历史事件批量回放，最后发送 `stream_done` 事件
4. **Given** 客户端在任务完成后 5 分钟重连，**When** Redis 缓冲已过期，**Then** 服务端返回 `410 Gone`，前端提示"该分析已过期，请重新发起"
5. **Given** Redis 不可用（降级为内存模式），**When** 客户端断线，**Then** 行为与 Redis 可用时相同（内存缓冲），但重启后不可恢复

---

### User Story 2 — Soft Interrupt（ESC 中断，Priority: P1）

作为用户，我在等待分析时发现市场发生了重大新闻（如大型交易所暴雷），我希望能按下"软中断"键（ESC 或点击"中断"按钮），让系统**保存当前已完成节点的检查点**，同时允许已在运行中的子 Agent 继续完成（不强杀），然后用已有结果生成一个"不完整但立即可用"的快速裁决。

**Why this priority**：强制等待 60 秒的完整分析可能让用户错过最佳干预时机。Soft Interrupt 赋予用户主动权，是从"被动等待"到"主动协作"的关键体验转变。P1。

**Independent Test**：触发分析 → 等待至少 2 个 Agent 完成 → 发送 `interrupt` 消息 → 观察是否收到 `checkpoint_saved` 事件 → 之后收到基于已有分析的快速裁决（`verdict_partial`），其中注明"基于 N/4 Agent 的不完整分析"。

**Acceptance Scenarios**:

1. **Given** 分析正在进行（已完成 ≥ 1 个 Agent），**When** 客户端发送 `interrupt` 消息，**Then** 服务端发出 `checkpoint_saved` 事件，包含已完成节点列表与检查点 ID
2. **Given** 中断信号发出时有 Agent 仍在运行，**When** 信号到达，**Then** 运行中的 Agent **不被强杀**，继续运行至当前 LLM 调用完成
3. **Given** 中断后所有运行中 Agent 完成，**When** 检查点刷写完毕，**Then** 系统用已有 Agent 分析生成 `verdict_partial` 并推送到事件流
4. **Given** 无 Agent 已完成（中断发生在 `collect_data` 阶段），**When** 中断信号到达，**Then** 服务端返回 `interrupt_rejected` 事件，说明"尚无可用分析结果"
5. **Given** 中断产生了 `verdict_partial`，**When** 裁决送达，**Then** 事件中包含 `is_partial: true` 标志，前端在裁决卡片上显示明显的"不完整分析"提示

---

### User Story 3 — Live Steering（实时引导，Priority: P1）

作为研究员，我在分析进行中观察到某 Agent 的推理方向与我的判断相悖（例如 NewsAgent 过度聚焦于一则 FUD），我希望在不中断整体分析的情况下，向该 Agent 的下一次 LLM 调用**注入一条修正指令**（如"请降低 FUD 新闻的权重，聚焦链上资金流向"）。

**Why this priority**：4 个 Agent 并行时，单个 Agent 推理偏差会拉低裁决质量。Live Steering 允许用户在代价最低的时间点（Agent 被调用之前）干预，而不是等全部完成后才发现问题。P1。

**Independent Test**：触发分析 → 在分析开始 5 秒内通过 `POST /api/chat/stream` 发送 steering 消息（格式：`{"type": "steer", "target": "news_agent", "instruction": "降低 FUD 权重"}`）→ 观察后续 NewsAgent 的分析结果中是否体现了该引导（`reasoning` 字段包含对引导的响应）。

**Acceptance Scenarios**:

1. **Given** 分析正在进行，某 Agent 尚未开始 LLM 调用，**When** 客户端发送 `steer` 类型消息（含 `target` + `instruction`），**Then** 服务端发出 `steer_queued` 事件，指令进入该 Agent 的 steering 队列
2. **Given** 指令已进入队列，**When** 目标 Agent 开始 LLM 调用，**Then** 指令被注入到 prompt 的 system 消息中（作为补充上下文）
3. **Given** 目标 Agent 已经完成 LLM 调用，**When** steering 消息到达，**Then** 服务端发出 `steer_too_late` 事件，说明该 Agent 已完成；前端提示"该 Agent 已完成分析，指令未生效"
4. **Given** 指令被注入，**When** Agent 分析完成，**Then** 对应的 `agent_analysis` 事件中包含 `steered: true` 标志
5. **Given** 用户同时向多个 Agent 发送 steering，**When** 各 Agent 依次被调用，**Then** 每个 Agent 独立应用各自的 steering 指令，互不干扰

---

### User Story 4 — 结构化 SSE 事件流（Priority: P1）

作为前端开发者，我需要在分析过程中精确感知各管线节点的进度（哪个 Agent 正在运行、辩论已进入第几轮、当前风控门的状态），而不仅仅是拿到最终的文本摘要。当前 SSE 只有 `session`、`message_start`、`content_delta`、`tool_call`、`message_end`、`done` 等少数类型，无法渲染细粒度进度条与实时 Agent 卡片。

**Why this priority**：细粒度事件是断线重连（Story 1）、Soft Interrupt（Story 2）、Live Steering（Story 3）共同依赖的基础设施，没有结构化事件就无法实现其他能力。P1。

**Independent Test**：触发分析 → 观察 SSE 事件日志 → 确认按顺序收到以下事件类型：`analysis_started` → `node_started`（collect_data）→ `node_done`（collect_data）→ `node_started`（tech_agent）→ `agent_thinking`（tech_agent）→ `agent_analysis`（tech_agent）→ … → `debate_started` → `debate_round_done` → `verdict_ready` → `risk_checked` → `stream_done`。

**Acceptance Scenarios**:

1. **Given** 分析管线启动，**When** 任意 LangGraph 节点开始执行，**Then** 发出 `node_started` 事件（含节点名称、时间戳）
2. **Given** 任意节点完成，**When** 节点返回结果，**Then** 发出 `node_done` 事件（含节点名、耗时 ms）
3. **Given** Agent 开始 LLM 调用，**When** 第一个 token 到达，**Then** 发出 `agent_thinking` 事件（含 agent_id）；LLM 调用结束发出 `agent_analysis` 事件（含完整分析 JSON）
4. **Given** 辩论轮次开始，**When** debate_round 节点启动，**Then** 发出 `debate_started` 事件（含轮次编号）；各 Agent 辩论完成后发出 `debate_round_done`
5. **Given** 裁决完成，**When** make_verdict 节点返回，**Then** 发出 `verdict_ready` 事件（含 action/confidence/position_scale）；风控门完成后发出 `risk_checked` 事件（含 allowed: bool 和拒绝原因）

---

### User Story 5 — Watch 端点（新工作流通知，Priority: P2）

作为外部集成用户（或 CI 工具），我希望通过一个长连接监听所有新分析任务的触发（来自调度器、来自手动调用、来自 API），在新任务启动时立即收到通知（含 session_id），以便决定是否附接该任务的事件流。

**Why this priority**：调度器自动触发的分析任务缺乏通知机制，外部系统无法知晓何时有新分析进行中。P2 是增量能力，不阻塞核心用户故事。

**Independent Test**：客户端连接 `GET /api/chat/watch` → 手动触发一次 `/api/chat/stream` 分析 → 观察 watch 端点是否收到 `new_workflow` 事件（含 session_id）→ 用该 session_id 重连 `/api/chat/stream` 确认能附接进行中的分析。

**Acceptance Scenarios**:

1. **Given** 客户端连接 `GET /api/chat/watch`，**When** 新分析任务启动（无论来源），**Then** Watch 端点推送 `new_workflow` 事件（含 session_id、pair、trigger_source）
2. **Given** 多个 watch 客户端同时连接，**When** 新任务启动，**Then** 所有 watch 客户端都收到通知
3. **Given** 调度器触发的自动分析启动，**When** 事件发布到 Pub/Sub，**Then** watch 端点转发 `new_workflow` 事件，前端可据此弹出"有新分析进行中"提示条

---

### 边界条件

- **Redis OOM**：单个分析任务的事件缓冲不超过 500 条；超出后以 FIFO 丢弃最旧事件，并在流中插入 `buffer_overflow` 警告事件
- **分析任务异常崩溃**：后台任务抛出未捕获异常时，将 `error` 事件写入 Redis 缓冲，重连客户端可收到错误原因
- **重复中断**：对已处于中断状态的任务再次发送 `interrupt`，返回 `interrupt_noop` 事件
- **Steering 注入过长**：单条 steering 指令超过 500 字符时，服务端截断并发出 `steer_truncated` 警告事件
- **session_id 碰撞**：同一 session_id 触发新分析时，旧任务的缓冲被覆盖，旧连接收到 `session_replaced` 事件后自动断开
- **Redis 降级时的 watch 端点**：Redis Pub/Sub 不可用时，watch 端点改为进程内广播（单进程单用户场景下功能等价）

---

## Requirements *(mandatory)*

### Functional Requirements

#### 后台任务管理

- **FR-001**：系统必须提供 `BackgroundTaskManager`，能将 LangGraph 分析任务封装为独立后台协程，使其生命周期与 HTTP 连接解耦
- **FR-002**：后台任务必须保护核心分析协程不被 HTTP 连接断开所取消，确保分析结果不因客户端断连而丢失
- **FR-003**：每个后台任务在 Redis 中拥有独立的事件缓冲（Redis List），键名格式为 `analysis:events:{session_id}`
- **FR-004**：事件缓冲设置 TTL（默认 5 分钟），过期后自动清理；TTL 可通过配置项调整
- **FR-005**：后台任务在完成（正常完成或异常终止）后，必须在事件缓冲中写入终止标记事件（`stream_done` 或 `stream_error`）
- **FR-006**：同一进程内同时运行的后台任务数量不得超过配置上限（默认 10），超出时返回 `429 Too Many Requests`

#### 断线重连

- **FR-007**：`POST /api/chat/stream` 接受可选的 `last_event_id` 请求参数；当参数存在时，服务端从该 ID 之后的事件开始回放
- **FR-008**：回放完成后，若任务仍在进行，服务端自动附接实时事件队列，无缝继续推送后续事件
- **FR-009**：每个 SSE 事件必须携带全局自增的 `id` 字段（与 SSE 协议 `id:` 字段对应），客户端可通过 `Last-Event-ID` HTTP 头或请求体 `last_event_id` 字段传递
- **FR-010**：当 `last_event_id` 对应的任务已过期（TTL 超时）时，服务端返回 `410 Gone` HTTP 状态码

#### Soft Interrupt

- **FR-011**：客户端可在分析进行中通过同一 SSE 连接发送 `interrupt` 控制消息（通过 `POST /api/chat/interrupt/{session_id}` 端点）
- **FR-012**：服务端收到 `interrupt` 后，设置该任务的中断标志位；已在运行中的 LLM 调用**不被强制取消**，而是等待其自然完成
- **FR-013**：中断标志生效后，尚未开始的管线节点（含辩论轮次）不再启动，系统用已完成的 Agent 分析生成 `verdict_partial`
- **FR-014**：中断产生的 `verdict_partial` 事件中必须包含 `is_partial: true` 标志及已完成 Agent 列表
- **FR-015**：中断在无可用 Agent 分析时（例如仅完成了 `collect_data`）返回 `interrupt_rejected` 事件，说明拒绝原因

#### Live Steering

- **FR-016**：客户端可通过 `POST /api/chat/steer/{session_id}` 端点向进行中的分析注入 steering 指令，请求体包含 `target`（Agent 名称）和 `instruction`（自然语言指令）
- **FR-017**：每个 Agent 在 LangGraph 节点执行前通过 `before_model` 钩子检查 steering 队列（Redis List：`steering:{session_id}:{agent_id}`），若有指令则追加到 system prompt
- **FR-018**：steering 指令到达但目标 Agent 已完成时，服务端发出 `steer_too_late` 事件，不修改已完成的分析结果
- **FR-019**：steering 指令注入成功后，对应 Agent 的 `agent_analysis` 事件中包含 `steered: true` 标志
- **FR-020**：单条 steering 指令长度上限为 500 字符；超出时服务端截断并发出 `steer_truncated` 警告

#### 结构化 SSE 事件

- **FR-021**：所有 SSE 事件必须遵循统一信封格式：`{"event_id": int, "type": string, "ts": ISO8601, "session_id": string, "data": object}`
- **FR-022**：系统必须支持以下结构化事件类型，按管线阶段分组：

  **连接与会话类**
  - `session_start`：分析任务启动（含 pair、trigger_source）
  - `stream_resume`：断线重连成功（含回放起始 event_id）
  - `stream_done`：分析正常完成
  - `stream_error`：分析异常终止（含错误摘要）
  - `session_replaced`：同一 session_id 被新任务覆盖

  **节点进度类**
  - `node_started`：LangGraph 节点开始执行（含节点名）
  - `node_done`：节点完成（含节点名、耗时 ms）

  **Agent 分析类**
  - `agent_thinking`：Agent 开始 LLM 调用（含 agent_id）
  - `agent_analysis`：Agent 完成分析（含完整分析 JSON、steered 标志）

  **辩论类**
  - `debate_started`：辩论轮次开始（含轮次编号）
  - `debate_round_done`：单轮辩论完成（含各 Agent 更新后立场）

  **裁决与风控类**
  - `verdict_ready`：正式裁决完成（含 action/confidence/position_scale）
  - `verdict_partial`：软中断产生的不完整裁决（含 is_partial: true）
  - `risk_checked`：风控门完成（含 allowed: bool、flags）

  **控制类**
  - `checkpoint_saved`：软中断检查点已保存（含已完成节点列表）
  - `interrupt_rejected`：中断被拒绝（含原因）
  - `steer_queued`：steering 指令已入队（含 target、queue_position）
  - `steer_too_late`：steering 指令到达时 Agent 已完成
  - `steer_truncated`：steering 指令被截断
  - `buffer_overflow`：事件缓冲溢出警告

- **FR-023**：现有事件类型（`session`、`message_start`、`content_delta`、`tool_call`、`message_end`、`done`）必须在过渡期（该 Feature 发布后 2 个版本内）继续支持，以保持前端向后兼容

#### Watch 端点

- **FR-024**：提供 `GET /api/chat/watch` SSE 端点；客户端连接后，每次有新分析任务启动时推送 `new_workflow` 事件
- **FR-025**：`new_workflow` 事件包含 `session_id`、`pair`、`trigger_source`（`manual` / `scheduler` / `api`）
- **FR-026**：新任务启动时通过 Redis Pub/Sub 频道 `analysis:new_workflow` 广播；Watch 端点订阅此频道转发给已连接客户端
- **FR-027**：Redis Pub/Sub 不可用时，Watch 端点降级为进程内广播（asyncio `Queue`），功能保持等价（单进程场景）

#### 配置

- **FR-028**：新增配置项 `[chat]` 段落，包含：
  - `event_buffer_ttl_seconds`（默认 300）
  - `max_concurrent_tasks`（默认 10）
  - `max_steering_instruction_chars`（默认 500）
  - `event_buffer_max_size`（默认 500）

---

### Non-Functional Requirements

- **NFR-001（延迟）**：断线重连后首个历史事件必须在 200ms 内开始推送（不含网络往返）
- **NFR-002（兼容性）**：所有新端点必须通过现有 `X-API-Key` 认证中间件（与其他 `/api/*` 路由一致）
- **NFR-003（内存）**：单个后台任务的内存占用（含事件缓冲）不得超过 50MB
- **NFR-004（可观测性）**：每个后台任务的生命周期事件（创建/完成/中断/异常）必须写入结构化日志，含 session_id、pair、duration_ms

---

### Key Entities

- **`BackgroundTaskManager`**：管理后台分析任务生命周期的单例组件；负责任务创建、中断标志管理、并发上限控制
- **`AnalysisTask`**：单次分析的运行时表示；包含 session_id、asyncio Task 引用、中断标志、steering 队列引用
- **事件缓冲（Event Buffer）**：Redis List，键名 `analysis:events:{session_id}`；存储序列化的 SSE 事件帧，带 TTL
- **Steering 队列**：Redis List，键名 `steering:{session_id}:{agent_id}`；存储待注入的 steering 指令
- **Watch 频道**：Redis Pub/Sub 频道 `analysis:new_workflow`；广播新任务启动通知
- **`last_event_id`**：事件的全局自增序号（per session），用于断线重连的精确回放定位
- **`verdict_partial`**：软中断场景下基于部分 Agent 分析生成的不完整裁决；携带 `is_partial: true` 标志

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**：网络中断 ≤ 5 分钟时，重连成功率 100%（不丢失已生成事件），通过集成测试验证
- **SC-002**：断线重连后的事件回放延迟 P99 < 500ms（本地 Redis 环境）
- **SC-003**：后台任务在 SSE 连接断开后继续运行完成（验证：连接断开后 60 秒内，Redis 缓冲中存在 `stream_done` 事件）
- **SC-004**：Soft Interrupt 响应时间 < 1 秒（从发送 interrupt 请求到收到 `checkpoint_saved` 事件）
- **SC-005**：Live Steering 指令注入成功率 ≥ 95%（目标 Agent 未完成时）；指令注入失败时 100% 给出明确反馈事件
- **SC-006**：SSE 事件类型扩展为 ≥ 15 种结构化类型，原有 6 种类型继续兼容（过渡期内零破坏性变更）
- **SC-007**：Redis 不可用时，内存降级模式下功能完整（单进程场景），且系统日志中有明确的降级警告
- **SC-008**：新增的后台任务管理不引入新的内存泄漏（通过 24 小时压测验证，任务完成后内存回收）
- **SC-009**：所有新 API 端点通过现有 API Key 认证，未认证请求返回 `401 Unauthorized`
- **SC-010**：测试覆盖率：新增代码的单元测试覆盖率 ≥ 85%；关键路径（断线重连、中断、steering 注入）有端到端集成测试

---

## Assumptions

- Redis 已在生产环境中运行（通过 Docker Compose），`RedisStateManager` 提供的连接模式可复用于事件缓冲
- 系统为单用户自托管场景，无多租户隔离需求；session_id 碰撞由客户端负责生成唯一值（UUID4）
- LangGraph 节点间的进度通知通过在节点函数内显式发布事件实现（不依赖 LangGraph 内置的 streaming 机制）
- 前端 `streamFetch` 采用 POST-based SSE（非 `EventSource`），已支持通过请求体传递 `last_event_id`；无需更改 `streamFetch` 的底层传输机制
- 协程保护机制足以防止分析任务被 HTTP 框架的连接断开取消传播
- Soft Interrupt 产生的 `verdict_partial` 不触发实盘订单执行；执行决策仅基于 `verdict_ready`
- Live Steering 仅影响 prompt 内容，不修改 Agent 的工具调用逻辑或数据收集范围
- steering 指令注入点为 LangGraph 节点 `before_model` 拦截位置，与现有 `@node_logger()` 装饰器模式共存
- 事件缓冲 TTL 默认 5 分钟满足"分析完成后短暂断线重连"的典型场景；长时离线恢复不在本 Feature 范围内

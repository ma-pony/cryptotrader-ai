# 实施任务列表：分析防丢失 + Live Steering

**Feature Branch**: `005-analysis-loss-prevention-live-steering`
**对应 Plan**: `specs/005-analysis-loss-prevention-live-steering/plan.md`
**对应 Spec**: `specs/005-analysis-loss-prevention-live-steering/spec.md`
**编写日期**: 2026-04-17

---

## 任务状态说明

- `[ ]` 待开始　`[x]` 已完成　`[~]` 进行中
- `[P]` 可与同 Phase 内其他 `[P]` 任务并行执行
- 任务编号格式：`T0XX`（按 Phase 分组，从 001 开始）

---

## 任务概览

| 阶段 | 任务编号 | 描述 | 预计耗时 |
|------|----------|------|----------|
| Phase 1 配置与基础设施 | T001–T005 | ChatConfig、Redis List/Pub/Sub 扩展、子包骨架 | 7h |
| Phase 2 后台任务管理 | T006–T010 | EventBuffer、EventBus、BackgroundTaskManager、chat.py 重构 | 10h |
| Phase 3 断线重连 | T011–T013 | 精确回放逻辑、410/session_replaced 边界 | 4h |
| Phase 4 结构化 SSE 事件 | T014–T019 | 节点/Agent/辩论/裁决事件 + 顺序验证 | 7h |
| Phase 5 Soft Interrupt | T020–T023 | 快速裁决、中断协程检查、interrupt 端点 | 5h |
| Phase 6 Live Steering | T024–T026 | 队列注入、steer 端点、并发边界 | 4h |
| Phase 7 Watch 端点 | T027–T028 | Pub/Sub + 进程内广播 | 3h |
| Phase 8 前端集成 | T029–T034 | 类型定义、事件处理、进度面板、断线重连 UX | 9h |
| Phase 9 集成测试 | T035–T042 | 单元测试 + 集成测试 + 全量回归 | 10h |

**总预计耗时**: ~59 小时

---

## Phase 1：配置与基础设施

> 目标：为后续所有 Phase 提供配置 dataclass 和 Redis 扩展能力。无外部依赖，优先完成。

- [X] **T001** — `src/cryptotrader/config.py` 新增 `ChatConfig` dataclass
  - 字段：`event_buffer_ttl_seconds: int = 300`、`max_concurrent_tasks: int = 10`、`max_steering_instruction_chars: int = 500`、`event_buffer_max_size: int = 500`
  - `AppConfig` 新增 `chat: ChatConfig = field(default_factory=ChatConfig)`
  - **关联**：FR-028
  - **验收**：`load_config().chat.event_buffer_ttl_seconds == 300`；无 `[chat]` TOML 段时不报错

- [X] **T002** — `config/default.toml` 新增 `[chat]` 段落
  - 内容：4 个字段及中文注释，与 `ChatConfig` 默认值一致
  - **关联**：FR-028
  - **验收**：`load_config()` 解析后各字段值与 TOML 一致

- [X] **T003** [P] — `src/cryptotrader/risk/state.py` 的 `_MemoryStore` 扩展 List 操作
  - 新增 `lists: dict[str, list[str]]` 字段（带 TTL 支持）
  - 新增方法：`list_rpush(key, value, ex=None)`、`list_lrange(key, start, end) -> list[str]`、`list_llen(key) -> int`、`list_ltrim(key, start, end)`
  - **关联**：FR-003、FR-007（内存降级路径）
  - **验收**：list_rpush 后 list_lrange 能取回；TTL 过期后 list_lrange 返回空列表

- [X] **T004** [P] — `src/cryptotrader/risk/state.py` 的 `RedisStateManager` 扩展 List 和 Pub/Sub
  - 新增 List 方法：`async buffer_push(key, value, max_size, ttl)`（含 LTRIM 溢出保护）、`async buffer_range(key, start, end) -> list[str]`、`async buffer_len(key) -> int`、`async buffer_set_ttl(key, ttl_s)`
  - 新增 Pub/Sub 方法：`async publish(channel, message)`、`subscribe_iter(channel) -> AsyncIterator[str]`
  - 进程内降级：`_in_proc_queues: dict[str, set[asyncio.Queue]]`；Redis 不可用时路由到 `_MemoryStore` list 操作 + `asyncio.Queue` 广播
  - **关联**：FR-003、FR-004、FR-026、FR-027
  - **验收**：Redis 可用时使用真实 Redis；Redis 断开后自动降级；publish/subscribe_iter 在内存模式下等价

- [X] **T005** — 新建 `src/cryptotrader/chat/` 子包骨架
  - 创建 `src/cryptotrader/chat/__init__.py`（空，待后续填充导出）
  - **关联**：整体架构
  - **验收**：`import cryptotrader.chat` 不报错

---

## Phase 2：后台任务管理

> 目标：实现分析任务与 HTTP 连接生命周期解耦的核心机制。

- [X] **T006** [P] — `src/cryptotrader/chat/event_buffer.py` 实现 `EventBuffer`
  - 封装 `RedisStateManager` 的 buffer 操作，提供：
    - `push(envelope: SSEEnvelope)` — 序列化后 rpush，溢出时 ltrim + buffer_overflow 警告
    - `range_after(last_event_id: int) -> list[SSEEnvelope]` — 线性扫描回放
    - `next_event_id() -> int` — Redis INCR `analysis:event_seq:{session_id}`（降级时内存计数器）
    - `set_ttl(ttl_s: int)` — 刷新 TTL
    - `mark_done(event_type: str)` — 写入 `stream_done` 或 `stream_error` 终止标记
  - **关联**：FR-003、FR-004、FR-005、FR-009
  - **验收**：push 后 range_after(0) 能取回所有事件；TTL 过期后 range_after 返回空；内存降级行为等价

- [X] **T007** [P] — `src/cryptotrader/chat/event_bus.py` 实现 `SSEEnvelope` 和 `EventBus`
  - `SSEEnvelope` dataclass：`event_id: int`、`type: str`、`ts: str`（ISO8601 UTC）、`session_id: str`、`data: dict`
  - `EventBus` 类：
    - `publish(event_type, data)` — 生成 event_id → 构造 SSEEnvelope → 写入 EventBuffer → put_nowait 到 `_live_queue: asyncio.Queue`
    - `subscribe() -> asyncio.Queue` — 返回实时队列供 SSE 生成器消费
    - `to_sse_frame(envelope) -> str` — 生成符合 SSE 协议的字符串帧（含 `id:` 字段）
  - **关联**：FR-003、FR-005、FR-009、FR-021
  - **验收**：publish 后 subscribe() 队列立即收到事件；event_id 单调递增；to_sse_frame 输出格式正确

- [X] **T008** — `src/cryptotrader/chat/task_manager.py` 实现 `AnalysisTask` 和 `BackgroundTaskManager`
  - `AnalysisTask` dataclass：`session_id`、`pair`、`trigger_source`、`task: asyncio.Task`、`interrupt_event: asyncio.Event`、`event_bus: EventBus`、`created_at: float`、`completed: bool = False`、`completed_agents: list[str]`
  - `BackgroundTaskManager` 单例：
    - `create(session_id, pair, coro, trigger_source, config) -> AnalysisTask`：检查并发上限，同 session_id 先发 `session_replaced`，复用 `task_registry.add_background_task()` 防 GC
    - `get(session_id) -> AnalysisTask | None`
    - `interrupt(session_id) -> bool`：设置 `interrupt_event`
    - `_on_task_done(session_id)` done_callback：标记 completed，写入结构化日志（NFR-004）
  - **关联**：FR-001、FR-002、FR-005、FR-006
  - **验收**：超并发上限抛 `TooManyTasksError`；Task 完成后注册表最终清除；interrupt 后 `interrupt_event.is_set()` 为 True

- [X] **T009** — `src/cryptotrader/chat/analysis_runner.py` 实现后台分析协程 `run_analysis_and_buffer()`
  - 发布 `session_start` → 写入 `analysis:status:{session_id}` → 构建 `initial_state`（metadata 注入 `event_bus`、`session_id`、`redis_state_manager`）
  - 调用 `run_graph_traced(graph, state, event_bus=event_bus)` 执行管线
  - `astream` 每个 chunk 后检查 `interrupt_event.is_set()`（见 Phase 5 T021）
  - 正常完成发布 `stream_done`；异常发布 `stream_error`（含错误摘要）
  - `finally` 调用 `_on_task_done()`，确保 `asyncio.CancelledError` 不传播给 HTTP 层
  - **关联**：FR-002、FR-005
  - **验收**：HTTP 连接断开后后台 Task 继续运行至发布 `stream_done`

- [X] **T010** — 重构 `src/api/routes/chat.py` 为后台任务 + 事件消费模式，并更新 `src/api/main.py`
  - `ChatStreamRequest` 新增 `last_event_id: int | None = None` 字段
  - 新建流程：`BackgroundTaskManager.create()` → 推送 `session_start` → `StreamingResponse` 消费 `event_bus.subscribe()` 队列
  - 重连流程（`last_event_id` 非 None）：推送 `stream_resume` → 批量回放历史 → 附接实时队列
  - 429（`TooManyTasksError`）和 410（EventBuffer 不存在/过期）错误处理
  - `main.py` lifespan：初始化 `BackgroundTaskManager` 存入 `app.state`，注册 `chat_control` router
  - 原有 6 种事件类型生成路径保留（兼容层，FR-023）
  - **关联**：FR-002、FR-007、FR-008、FR-009、FR-010、FR-023
  - **验收**：新建任务后断开连接，后台 Task 继续；重连后收到 `stream_resume` + 历史事件回放

---

## Phase 3：断线重连

> 目标：实现基于 `last_event_id` 的精确事件回放，验证 SC-001~SC-003。

- [X] **T011** — `EventBuffer.range_after()` 精确回放逻辑
  - 线性扫描 Redis List（最多 500 条），找到 `event_id > last_event_id` 的起始位置，返回后续事件列表
  - P99 延迟目标：< 200ms（本地 Redis）
  - **关联**：FR-007、FR-008、NFR-001
  - **验收**：100 条历史事件中回放 `last_event_id=50` 后精确返回 50 条；无事件时返回空列表

- [X] **T012** — `_sse_consumer_gen()` 中实现回放 → 附接实时流的无缝衔接
  - 回放完成后检查任务是否仍在运行：运行中则附接 `event_bus.subscribe()` 队列；已完成则发送 `stream_done`
  - **关联**：FR-008、SC-003
  - **验收**：断线期间任务已完成时，重连后批量收到所有历史事件 + `stream_done`

- [X] **T013** — 处理 `410 Gone` 和 `session_id` 碰撞边界条件
  - TTL 过期时 `buffer_range()` 返回空且 status key 不存在 → HTTP 410 Gone
  - 同一 `session_id` 触发新分析时，旧连接的实时队列收到 `session_replaced` 事件后自动断开
  - **关联**：FR-010；边界条件（session_id 碰撞）
  - **验收**：过期后重连返回 410；新任务覆盖时旧连接收到 `session_replaced`

---

## Phase 4：结构化 SSE 事件

> 目标：实现 FR-022 的 20+ 种结构化事件类型，为 Phase 5/6 提供基础设施。

- [X] **T014** [P] — `src/cryptotrader/chat/node_events.py` 实现 `node_event_scope()` 上下文管理器
  - `wrap_node_with_events(fn, event_bus) -> async fn`：节点执行前发布 `node_started {node_name, ts}`，执行后发布 `node_done {node_name, duration_ms}`
  - 与 `@node_logger()` 共存（职责互补，不冲突）
  - **关联**：FR-022（node_started、node_done）
  - **验收**：包装后的节点正确发布两个事件；`duration_ms` 为正整数

- [X] **T015** [P] — `src/cryptotrader/tracing.py` 的 `run_graph_traced()` 接受可选 `event_bus` 参数
  - 新增 `event_bus: EventBus | None = None` 参数
  - 若非 None，在 `astream` chunk 循环中调用 `event_bus.publish("node_started"/"node_done", ...)`
  - **关联**：FR-022
  - **验收**：传入 `event_bus` 后，订阅队列能收到各节点的 `node_started`/`node_done`；不传时行为完全不变

- [X] **T016** — `src/cryptotrader/nodes/agents.py` 的 `_run_agent()` 发布 Agent 类事件
  - 从 `state["metadata"].get("event_bus")` 获取可选 `event_bus`
  - LLM 调用前发布 `agent_thinking {agent_id}`
  - LLM 调用完成后发布 `agent_analysis {agent_id, direction, confidence, reasoning, key_factors, risk_flags, steered}`
  - Agent 完成后，通过 `state["metadata"].get("session_id")` 获取 session_id，再调用 `BackgroundTaskManager.get_task(session_id)` 取得 `AnalysisTask` 引用，将 `agent_id` 追加到 `task.completed_agents`（不使用 `event_bus.parent_task`，避免循环引用）
  - **关联**：FR-022（agent_thinking、agent_analysis）
  - **验收**：4 个并行 Agent 各自独立发布事件；无 event_bus 时行为不变

- [X] **T017** [P] — `src/cryptotrader/nodes/debate.py` 发布辩论类事件
  - `debate_round()` 开始时发布 `debate_started {round_number}`
  - 所有辩论 Agent 完成后发布 `debate_round_done {round_number, updated_positions}`
  - **关联**：FR-022（debate_started、debate_round_done）
  - **验收**：两轮辩论各自发布一对事件；轮次编号正确

- [X] **T018** [P] — `src/cryptotrader/nodes/verdict.py` 和风控节点发布裁决类事件
  - 裁决完成后发布 `verdict_ready {action, confidence, position_scale, reasoning}`
  - 风控门完成后发布 `risk_checked {allowed: bool, flags, reason}`
  - **关联**：FR-022（verdict_ready、risk_checked）
  - **验收**：完整管线运行后，EventBuffer 中包含 `verdict_ready` 和 `risk_checked` 事件

- [X] **T019** — 编写完整事件序列顺序验证测试
  - 触发完整管线（mock LLM），断言收到的事件类型序列符合 FR-022 描述的顺序
  - 序列起点为 `session_start`（注意：spec.md 示例中出现的 `analysis_started` 为旧命名，以 `session_start` 为准，与 FR-022 和 `AnalysisEventType` 定义一致）
  - 断言 event_id 单调递增，信封格式符合 FR-021
  - **关联**：FR-021、FR-022、SC-006
  - **验收**：≥15 种事件类型全部出现；序列首事件为 `session_start`；event_id 严格单调递增

---

## Phase 5：Soft Interrupt

> 目标：实现 ESC 软中断，在已完成 ≥1 个 Agent 时生成快速裁决，验证 SC-004。

- [X] **T020** — `src/cryptotrader/chat/partial_verdict.py` 实现 `make_partial_verdict()`
  - 基于已完成 Agent 分析生成快速裁决：加权平均置信度、多数投票方向
  - 结果携带 `is_partial: True`、`completed_agents` 列表、`missing_agents` 列表
  - **关联**：FR-013、FR-014
  - **验收**：2 个相反方向 Agent 时 `action` 为 `hold` 或 `neutral`；`is_partial: true` 字段必须存在

- [X] **T021** — `run_analysis_and_buffer()` 添加中断检查点
  - `astream` 每个 chunk 后检查 `interrupt_event.is_set()`
  - `completed_agents >= 1` → 调用 `_make_partial_verdict()`，发布 `checkpoint_saved + verdict_partial` → 写入 `stream_done`，退出循环
  - `completed_agents == 0`（为空）→ 发布 `interrupt_rejected {reason: "尚无可用 Agent 分析结果"}`，**继续运行**（不退出）
  - **关联**：FR-012、FR-013、FR-014、FR-015
  - **验收**：中断时运行中 LLM 调用自然完成（不被强杀）；已完成 Agent 分析不丢失

- [X] **T022** — `src/api/routes/chat_control.py` 实现 `POST /api/chat/interrupt/{session_id}`
  - 获取 `task_manager.get(session_id)` → 404 if None
  - 任务已完成 → 返回 `interrupt_noop`
  - 已处于中断状态 → 返回 `interrupt_noop`
  - 否则 `task_manager.interrupt(session_id)` → 返回 `200 {type: "interrupt_received"}`（实际 checkpoint_saved 通过 SSE 流推送）
  - 需通过 `verify_api_key` 依赖（NFR-002）
  - **关联**：FR-011、NFR-002、SC-004
  - **验收**：从发送 interrupt 到 EventBuffer 出现 `checkpoint_saved`，端到端 < 1 秒

- [X] **T023** — 处理重复中断和无效中断边界条件
  - 已中断状态再次 interrupt → `interrupt_noop`
  - Task 已完成时 interrupt → `interrupt_noop`（不是 404）
  - **关联**：边界条件（重复中断）
  - **验收**：重复中断返回 `interrupt_noop` 而非 500 错误

---

## Phase 6：Live Steering

> 目标：实现向进行中 Agent 注入实时 steering 指令，验证 SC-005。

- [X] **T024** — `src/cryptotrader/nodes/agents.py` 的 `_run_agent()` 读取并注入 steering 队列
  - 在 `agent.analyze()` 调用前：从 `state["metadata"]` 获取 `session_id` 和 `redis_state_manager`
  - 读取 `steering:{session_id}:{agent_id}`（`buffer_range(0, -1)` 后删除键）
  - 若队列非空：拼接到 `experience` 末尾（格式：`\n\n[用户实时引导]\n{instructions}`），设置 `steered = True`
  - `agent_analysis` 事件中携带 `steered: true`
  - **关联**：FR-017、FR-019
  - **验收**：steering 指令注入后 `agent_analysis` 包含 `steered: true`；队列为空时 `steered: false`

- [X] **T025** — `src/api/routes/chat_control.py` 实现 `POST /api/chat/steer/{session_id}`
  - 请求体：`SteerRequest(target: str, instruction: str)`
  - 校验 `target` 为合法 Agent 名 → 422 if 无效
  - 超过 `max_steering_instruction_chars` 时截断，发布 `steer_truncated` 警告
  - 检查 `target` 是否在 `task.completed_agents` → `steer_too_late` if 已完成
  - 写入 `steering:{session_id}:{target}` 队列（`RPUSH`），发布 `steer_queued {target, queue_position}`
  - **关联**：FR-016、FR-018、FR-019、FR-020、NFR-002
  - **验收**：Agent 未完成时成功入队；超长指令截断并警告；Agent 已完成时返回 `steer_too_late`

- [X] **T026** — 验证并发 steering 互不干扰
  - 编写测试：同时向 `tech_agent` 和 `news_agent` 注入不同指令，验证各自独立应用，互不覆盖
  - **关联**：User Story 3 验收场景 5
  - **验收**：两个 Agent 各自收到且仅收到自己的 steering 指令

---

## Phase 7：Watch 端点

> 目标：实现新工作流通知端点，为外部系统提供分析任务启动广播，验证 User Story 5。

- [X] **T027** — `src/api/routes/chat_control.py` 实现 `GET /api/chat/watch` SSE 端点
  - 连接时订阅 Redis Pub/Sub `analysis:new_workflow`（`RedisStateManager.subscribe_iter`）
  - Redis 不可用时降级为进程内 `asyncio.Queue` 广播（`subscribe_in_proc`）
  - 每次收到消息时推送 `new_workflow {session_id, pair, trigger_source}` SSE 事件
  - 连接断开时清理订阅
  - 需通过 `verify_api_key` 依赖（NFR-002）
  - **关联**：FR-024、FR-025、FR-026、FR-027、NFR-002
  - **验收**：连接后触发新分析，watch 端点收到 `new_workflow` 事件；`session_id` 与新分析一致

- [X] **T028** — `BackgroundTaskManager.create()` 发布新任务通知
  - 成功创建后台 Task 后，调用 `redis_state_manager.publish("analysis:new_workflow", json_payload)`
  - 同时广播到所有进程内订阅者（降级路径）
  - **关联**：FR-026、FR-027
  - **验收**：多个 watch 客户端同时连接时，都能收到 `new_workflow` 事件

---

## Phase 8：前端集成

> 目标：扩展前端以正确处理所有新事件类型，实现断线重连 UI 和 steering 输入。

- [X] **T029** [P] — 新建 `web/src/types/analysis-events.ts`
  - `AnalysisEventType` 联合类型（21 种新事件）
  - 各事件 `data` 字段接口：`NodeStartedData`、`AgentAnalysisData`、`VerdictPartialData` 等
  - `SSEEnvelope<T>` 泛型接口
  - `LegacyEventType` 联合类型（保留原有 6 种，向后兼容）
  - **关联**：FR-021、FR-022、FR-023
  - **验收**：TypeScript `strict` + `exactOptionalPropertyTypes` 编译无错误

- [X] **T030** [P] — `web/src/lib/stream-fetch.ts` 410 Gone 已由现有逻辑处理（不在 RETRYABLE_STATUSES 中）
  - 确认 `RETRYABLE_STATUSES` 不含 410
  - 在调用层通过 `SSEError.status === 410` 判断，`retryable = false`
  - **关联**：FR-010
  - **验收**：模拟 410 响应时 `SSEError.retryable === false`；上层不触发重试

- [X] **T031** [P] — `web/src/hooks/use-chat-messages.ts` 扩展 `handleEvent` 处理新事件类型
  - 新增 15 种事件类型的 `case` 分支（原有 6 种保持不变）
  - `stream_resume`：显示"已从断点恢复"提示；`session_replaced`：提示覆盖；`stream_error`：展示错误摘要
  - 控制反馈类事件（`checkpoint_saved`、`steer_queued`、`steer_too_late` 等）：显示对应提示
  - 确保文件行数 ≤ 500 行（NFR-M-007），必要时提取 helper 函数
  - **关联**：FR-022、FR-023
  - **验收**：原有 6 种事件处理行为无变化；新事件不落入 `default` 分支

- [X] **T032** [P] — 新建 `web/src/hooks/use-analysis-progress.ts`
  - 状态机：`nodes: Record<string, {status, duration_ms}>`、`agents: Record<string, {status, analysis, steered}>`、`debateRound`、`verdict`、`riskCheck`、`interrupted`、`lastEventId`
  - `handleProgressEvent(envelope: SSEEnvelope)` 方法：根据事件类型更新对应状态
  - 断线重连：持久化 `lastEventId` 到 `sessionStorage`；`SSEError` 时自动重连（携带 `last_event_id`）；410 时清除缓存提示重新发起
  - `sendInterrupt()` 和 `sendSteer(target, instruction)` 方法（调用新 API 端点）
  - **关联**：FR-007、FR-008、FR-010；User Story 1 验收场景 1~5
  - **验收**：模拟断线后重连请求体中 `last_event_id` 与断线前最后收到的 event_id 一致

- [X] **T033** [P] — 新建前端分析进度 UI 组件
  - `web/src/components/analysis/AnalysisProgressPanel.tsx`：顶层容器，接收 `useAnalysisProgress` 状态
  - `web/src/components/analysis/NodeProgressBar.tsx`：水平时间线，各节点显示 `pending`/`running`/`done`
  - `web/src/components/analysis/AgentCard.tsx`：Agent 卡片，含分析摘要、`steered` 标记 Badge、`SteeringInput` 输入框（仅分析中可见）
  - `web/src/components/analysis/VerdictCard.tsx`：裁决卡片，`is_partial` 时显示橙色警示横幅
  - `web/src/components/analysis/DebateStatus.tsx`：辩论轮次状态指示器
  - **关联**：User Story 2（is_partial 标记）、User Story 3（steering 输入）、User Story 4（进度面板）
  - **验收**：TypeScript 编译无错误；`is_partial: true` 时警示横幅正确渲染；steering 输入框在 Agent 完成后禁用

- [X] **T034** — `web/src/pages/chat/index.tsx` 嵌入 `AnalysisProgressPanel`
  - 将 `useAnalysisProgress` 状态传入 `AnalysisProgressPanel`
  - **关联**：User Story 4
  - **验收**：开发环境访问 `/chat` 页面可见进度面板骨架；TypeScript 编译无错误

---

## Phase 9：集成测试

> 目标：覆盖 SC-001~SC-010，确保关键路径端到端验证，确保无回归。

- [X] **T035** [P] — `tests/test_chat_event_buffer.py` EventBuffer 单元测试
  - 测试场景：push/range_after/TTL 过期/溢出保护（FIFO + `buffer_overflow` 警告）/内存降级路径
  - **关联**：FR-003、FR-004、FR-005；SC-007
  - **验收**：覆盖率 ≥ 90%；内存降级与 Redis 路径行为一致

- [X] **T036** [P] — `tests/test_chat_task_manager.py` BackgroundTaskManager 单元测试
  - 测试场景：创建/并发上限拒绝/session_id 碰撞（发 session_replaced）/Task 完成自动清理/interrupt_event 设置
  - **关联**：FR-001、FR-002、FR-006；SC-008
  - **验收**：Task 完成后从注册表清除，无内存泄漏；日志含 `session_id`、`pair`、`duration_ms`

- [X] **T037** [P] — `tests/test_chat_reconnect.py` 断线重连集成测试
  - 测试场景：① 分析进行中断线 → 后台 Task 继续 → 重连从断点回放；② 分析已完成时重连 → 批量回放；③ TTL 过期重连 → 410 Gone
  - 使用内存降级模式（minimize mocks 原则：mock Redis I/O，不 mock 业务逻辑）
  - **关联**：FR-007、FR-008、FR-009、FR-010；SC-001、SC-002、SC-003
  - **验收**：重连后事件序列完整无遗漏；回放延迟 P99 < 500ms

- [X] **T038** [P] — `tests/test_chat_interrupt.py` Soft Interrupt 集成测试
  - 测试场景：① ≥1 Agent 完成时中断 → `checkpoint_saved + verdict_partial(is_partial=true)`；② 0 Agent 完成时中断 → `interrupt_rejected`（继续运行）；③ 重复中断 → `interrupt_noop`；④ 中断时运行中 LLM 调用不被强杀
  - **关联**：FR-011~FR-015；SC-004
  - **验收**：从 interrupt 请求到 `checkpoint_saved` < 1 秒；`verdict_partial` 中 `is_partial: true`

- [X] **T039** [P] — `tests/test_chat_steering.py` Live Steering 集成测试
  - 测试场景：① 目标 Agent 未完成时入队 → `steer_queued` + `agent_analysis steered=true`；② 目标 Agent 已完成 → `steer_too_late`；③ 超长指令 → `steer_truncated` + 截断入队；④ 多 Agent 并发注入 → 各自独立应用
  - **关联**：FR-016~FR-020；SC-005
  - **验收**：指令注入成功率 100%（目标未完成时）；`steer_too_late` 不修改已完成分析

- [X] **T040** [P] — `tests/test_api_chat_control.py` 新 API 端点测试
  - 所有新端点（interrupt / steer / watch）的认证测试（无 API Key → 401）
  - 正常流程响应格式验证（200 + 正确事件类型 JSON）
  - **关联**：NFR-002；SC-009
  - **验收**：未认证请求返回 401；有效请求返回正确 SSEEnvelope 格式

- [X] **T041** [P] — `tests/test_chat_partial_verdict.py` 快速裁决逻辑测试
  - 测试场景：① 2 个相反方向 Agent → `action=hold`；② 3 个同向 + 1 个反向 → 多数方向；③ 0 个 Agent → 发布 `interrupt_rejected`
  - **关联**：FR-013、FR-014
  - **验收**：`is_partial: true`；`completed_agents` 列表与输入一致；置信度为加权平均

- [X] **T042** — 全量测试运行与代码质量检查
  - 运行 `pytest` 确保原有 742+ 个测试全部通过
  - 统计新增代码覆盖率（目标 ≥ 85%）
  - 运行 `ruff check src/ tests/`（零 lint 错误，禁止 `# noqa`）
  - **关联**：SC-010
  - **验收**：零测试失败；新增代码覆盖率 ≥ 85%；零 lint 错误

---

## 验收矩阵

| 成功标准 | 关联任务 |
|---------|---------|
| SC-001 重连成功率 100% | T011、T012、T037 |
| SC-002 回放延迟 P99 < 500ms | T011、T037 |
| SC-003 后台 Task 继续运行至完成 | T009、T010、T036 |
| SC-004 Soft Interrupt 端到端 < 1 秒 | T021、T022、T038 |
| SC-005 Steering 注入成功率 ≥ 95% | T024、T025、T039 |
| SC-006 ≥15 种结构化事件类型，原 6 种兼容 | T014~T018、T019 |
| SC-007 Redis 降级功能完整 | T003、T004、T006、T035 |
| SC-008 无内存泄漏 | T008（done_callback）、T036 |
| SC-009 新端点 API Key 认证正确 | T022、T025、T027、T040 |
| SC-010 新增代码覆盖率 ≥ 85% | T035~T042 |

---

## 代码质量检查清单

- [X] 零 lint 错误（`ruff check src/ tests/`）
- [X] 禁止 `# noqa` 注释（per 用户偏好）
- [X] 全部现有测试仍通过（742+ tests）
- [X] 新增测试全部通过
- [X] 新增代码行数 ≤ 500 行/文件（NFR-M-007）
- [X] 所有新 dataclass/Pydantic 模型有类型注解

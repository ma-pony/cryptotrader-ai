# 功能规格说明：人工审批节点 + Telegram 远程控制（HITL）

**Feature Branch**: `006-hitl-approval-telegram-control`
**Created**: 2026-04-17
**Status**: Draft

---

## 背景与动机

CryptoTrader-AI 当前是全自动流水线：数据采集 → 4 Agent 并行分析 → 辩论门控 → 2 轮辩论 → AI 裁决 → 风控门控 → 执行下单。整条链路没有人工介入点。

在以下场景中，完全自动化存在明显风险：

1. **大额仓位操作**：`position_scale > 0.5` 时，单笔交易可能占用投资组合 5% 以上资金。
2. **Agent 严重分歧**：4 个 Agent 中 2 多空对立，`divergence_scores` 高，AI 裁决质量存疑。
3. **风控熔断重置**：`circuit_breaker` 触发后需人工判断是否满足重启条件。
4. **冷启动期**：系统刚上线或参数大幅调整后，首批交易需要人工见证。

本功能在 LangGraph 图的 `risk_gate` 前插入可选的人工审批节点，并通过 Telegram Bot 提供移动端远程审批通道。审批规则完全可配置；不配置时，系统仍以纯自动模式运行。

---

## 用户故事与验收测试

### 用户故事 1 — 大额交易需确认（优先级：P0）

作为 CryptoTrader-AI 的自托管用户，我希望当系统准备开一笔超过仓位阈值的大单时，在下单前收到通知并能批准或拒绝，以便防止系统在我不知情的情况下建立过大仓位。

**为什么是 P0**：资金安全是核心需求。大额单笔失误可能造成不可恢复的亏损。

**独立验收测试**：可以在隔离环境中单独测试审批触发逻辑，无需运行完整交易流水线。

**验收场景**：

1. **Given** 裁决结果为 `action=long, position_scale=0.75`，审批配置 `min_position_scale_for_approval=0.5`，**When** 图运行至 `hitl_gate` 节点，**Then** 图通过 `interrupt()` 暂停，向 Web 前端推送一条待审批卡片，同时（若 Telegram 已配置）向 Telegram 发送包含完整裁决摘要的审批消息，等待人工响应。

2. **Given** 审批请求已发出，用户在 Web 前端点击"批准"，**When** 前端通过 `POST /api/hitl/{approval_id}/respond` 提交 `{"decision": "approve"}`，**Then** LangGraph 图从暂停点恢复，继续执行 `risk_gate` → `execute`，日志记录审批人和时间戳。

3. **Given** 审批请求已发出，超时时间（`approval_timeout_seconds`）到达，没有任何人工响应，**When** 超时 coroutine 触发，**Then** 图以自动拒绝（`decision=reject`）恢复，交易取消，日志记录超时拒绝原因，不发送任何交易到交易所。

---

### 用户故事 2 — Agent 分歧时人工裁决（优先级：P1）

作为用户，当 4 个分析 Agent 存在严重观点分歧时（即辩论后 `divergence_score` 仍高于阈值），我希望能看到完整的分歧摘要并决定是否允许系统继续下单，以便在 AI 自身不确定的情况下由人工把关。

**为什么是 P1**：Agent 高分歧是系统不确定性的关键信号，但这是可接受的偶发场景，不影响主流程。

**独立验收测试**：可以构造高 `divergence_score` 的 `ArenaState` 触发该路径。

**验收场景**：

1. **Given** 辩论结束后 `divergence_scores[-1] > hitl.divergence_threshold`（默认 0.6），且裁决 `action != "hold"`，**When** 图到达 `hitl_gate`，**Then** 触发人工审批，审批请求包含所有 4 个 Agent 的 `direction/confidence/reasoning` 摘要，以及辩论轮次后的分歧分数变化趋势。

2. **Given** 用户通过 Telegram 回复"拒绝"，**When** Telegram Bot 处理该消息，**Then** 图从暂停点恢复并将交易标记为拒绝，日志同步显示决策来源为 `telegram`。

3. **Given** `hitl.enabled=false`，**When** 图运行至 `hitl_gate` 节点，**Then** 节点直接透传，不触发任何审批，不向任何渠道发送消息。

---

### 用户故事 3 — Telegram 远程批准（优先级：P1）

作为用户，当我不在电脑旁时，我希望能直接在手机上的 Telegram 频道中看到交易提案，通过内联按钮批准或拒绝，以便在移动场景下也能参与审批，不因不在线而被迫错过时间窗口（超时自动拒绝）。

**为什么是 P1**：Telegram 解决"移动端不在线"场景，与 Web 审批并列但不互斥。

**独立验收测试**：Telegram Bot 通知和内联按钮响应可以 Mock `telegram.Bot` 单独测试。

**验收场景**：

1. **Given** `hitl.telegram.enabled=true` 且已配置 `bot_token` 和 `chat_id`，审批请求触发，**When** `hitl_gate` 节点执行，**Then** Telegram Bot 向指定 `chat_id` 发送包含以下内容的消息：币对、裁决动作、仓位比例、置信度、触发原因，以及"批准"/"拒绝"两个内联按钮。

2. **Given** Telegram 消息已发出，用户点击"批准"内联按钮，**When** Telegram Webhook 接收到 `callback_query`，**Then** Bot 处理回调，调用 `POST /api/hitl/{approval_id}/respond`，图从暂停点恢复，Telegram 消息更新为"已批准 ✓ by Telegram"。

3. **Given** 审批请求超时，**When** 超时触发自动拒绝，**Then** 如果 Telegram 消息仍处于待响应状态，Bot 更新该消息为"已超时 — 自动拒绝"，两个内联按钮变为不可用。

---

### 用户故事 4 — 风控熔断重置的人工确认（优先级：P2）

作为用户，在熔断器被触发后，当我通过 Web 界面发起重置时，我希望系统记录重置时间、触发原因和操作人，以便后续审计。

**为什么是 P2**：现有 `/api/risk/circuit-breaker/reset` 端点已可用，此为增强而非阻塞功能。

**独立验收测试**：可通过 API 测试单独验证日志记录逻辑。

**验收场景**：

1. **Given** 熔断器处于活跃状态，用户发起重置，**When** `POST /api/risk/circuit-breaker/reset` 成功，**Then** 系统在日志中记录结构化事件 `{event: "circuit_breaker_reset", triggered_at, reset_at, reason}`，并通过 Notifier 发送通知（若 `notifications.enabled=true`）。

---

### 边界条件

- **Telegram 未配置时降级**：`hitl.telegram.enabled=false` 时，仅使用 Web 审批。Telegram 相关代码不执行，无任何报错。
- **HITL 完全禁用时透传**：`hitl.enabled=false` 时，`hitl_gate` 节点作为透明节点直接传递状态，零副作用。
- **Web 和 Telegram 并发响应**：若 Web 和 Telegram 几乎同时收到响应，以先到达 `/api/hitl/{approval_id}/respond` 的为准；第二次响应返回 `409 Conflict`。
- **图在暂停期间重启**：系统重启后，所有待决审批请求应通过持久化存储恢复，超时机制继续生效。
- **回测模式下强制跳过**：`backtest_mode=true` 时，`hitl_gate` 节点必须无条件跳过，审批从不触发。
- **只有 `hold` 动作时不触发**：裁决为 `hold` 时不需要审批，直接继续（持仓不变，无资金风险）。
- **多币对并发**：不同币对的分析循环相互独立，A 币对的审批挂起不影响 B 币对的分析和执行。

---

## 功能需求

### FR-001：人工审批门控节点（HITL Gate Node）

系统必须在 LangGraph 全量图（`build_trading_graph()`）的 `verdict` 节点之后、`risk_gate` 节点之前，插入一个新的 `hitl_gate` 节点。该节点使用 LangGraph `interrupt()` 实现暂停，而不是阻塞线程。

### FR-002：审批触发条件（可配置）

系统必须支持以下可配置触发条件，满足任意一项即触发审批请求：

- **FR-002a 仓位规模阈值**：裁决 `position_scale >= hitl.min_position_scale`（默认值：`0.5`，即占最大单笔仓位限额的 50% 以上）。
- **FR-002b Agent 分歧阈值**：辩论后 `divergence_scores[-1] >= hitl.divergence_threshold`（默认值：`0.6`）。
- **FR-002c 冷启动周期**：系统生命周期内已完成的交易总数小于 `hitl.cold_start_min_trades`（默认值：`5`），每笔交易都需要审批。

### FR-003：超时自动拒绝（fail-closed）

审批请求发出后，若在 `hitl.approval_timeout_seconds`（默认值：`300`，即 5 分钟）内未收到人工响应，系统必须自动以 `reject` 决策恢复图执行，取消该笔交易。系统绝对不能因为人工不在线而自动放行交易。

### FR-004：审批请求持久化

每条审批请求必须持久化存储（SQLite）并包含以下字段：

- `approval_id`（UUID）
- `pair`（交易对）
- `created_at`（创建时间，UTC）
- `expires_at`（超时时间，UTC）
- `trigger_reason`（触发原因：`position_scale | divergence | cold_start`）
- `verdict_snapshot`（裁决全量快照，JSON）
- `agent_analyses_snapshot`（4 个 Agent 分析摘要，JSON）
- `status`（`pending | approved | rejected | expired`）
- `decision_by`（决策来源：`web | telegram | timeout`）
- `decided_at`（决策时间，UTC，nullable）

### FR-005：Web 审批 API

系统必须提供以下 RESTful API 端点：

- `GET /api/hitl/pending` — 列出所有状态为 `pending` 的审批请求（含完整 `verdict_snapshot` 和 `agent_analyses_snapshot`）。
- `GET /api/hitl/{approval_id}` — 查询单条审批请求详情。
- `POST /api/hitl/{approval_id}/respond` — 提交审批决策（`{"decision": "approve" | "reject", "comment": "可选备注"}`）。若请求已过期或已决策，返回 `409 Conflict`。

### FR-006：前端审批卡片（Approval Card）

Web 前端的风控页面（`/risk`）必须显示待审批请求列表。每条请求渲染为一张审批卡片，展示：

- 币对、触发原因（中文标签）
- 裁决动作（颜色编码：多头绿色、空头红色、平仓橙色）
- 仓位比例（`position_scale`）、置信度
- 4 个 Agent 方向摘要（bullish/bearish/neutral + 置信度）
- 分歧分数（divergence score）
- 剩余超时倒计时（实时更新）
- "批准"和"拒绝"操作按钮

前端必须通过轮询（每 5 秒）或 SSE 实时刷新待审批列表。

### FR-007：Telegram Bot 通知与响应

当 `hitl.telegram.enabled=true` 时，系统必须：

- 在审批请求创建时，向 `hitl.telegram.chat_id` 发送包含完整摘要和"批准"/"拒绝"内联按钮的 Telegram 消息。
- 通过 Webhook 或长轮询接收 Telegram `callback_query`，将响应转发至内部 `POST /api/hitl/{approval_id}/respond`。
- 在决策完成（批准、拒绝、超时）后，编辑原始 Telegram 消息，更新为决策结果和决策来源。

Telegram Bot 必须是可选组件：未配置 `bot_token` 时，不初始化 Bot，不影响 Web 审批功能。

### FR-008：审批日志与 Notifier 集成

每次审批决策（包括超时）必须：

- 写入结构化日志（`structlog`），包含 `approval_id, pair, trigger_reason, decision, decision_by, latency_seconds`。
- 通过现有 `Notifier` 系统发送 `hitl_decision` 事件通知（如 Webhook 已配置）。

### FR-009：回测模式强制跳过

`hitl_gate` 节点在检测到 `state["metadata"]["backtest_mode"] == True` 时，必须立即返回，不触发任何审批、通知或持久化操作。

### FR-010：配置项

新增 `[hitl]` 配置块，支持以下字段（均有默认值，向后兼容）：

- `enabled: bool = false` — 全局开关；`false` 时整个功能静默。
- `min_position_scale: float = 0.5` — 触发审批的仓位比例下限。
- `divergence_threshold: float = 0.6` — 触发审批的分歧分数下限。
- `cold_start_min_trades: int = 5` — 冷启动保护期内最少完成交易数。
- `approval_timeout_seconds: int = 300` — 审批超时时长（秒）。
- `[hitl.telegram]` 子块：
  - `enabled: bool = false`
  - `bot_token: str = ""`
  - `chat_id: str = ""`

### FR-011：并发安全

同一 `approval_id` 的响应必须是幂等安全的：并发写入时，系统通过数据库层乐观锁或状态检查确保最终只有一次决策生效，第二次写入返回 `409 Conflict`。

### FR-012：图变更限制

`hitl_gate` 节点仅插入 `build_trading_graph()`（全量图）。`build_lite_graph()`、`build_backtest_graph()`、`build_debate_graph()` 不变。`build_backtest_graph()` 内部已有 `risk_gate` 节点，该节点不受影响。

---

## 关键实体

- **ApprovalRequest（审批请求）**：单次人工审批的完整上下文，包含裁决快照、Agent 分析摘要、状态机（pending/approved/rejected/expired）。
- **HitlConfig（HITL 配置）**：描述触发条件、超时、Telegram 设置的配置数据类，属于 `AppConfig` 的子配置。
- **TelegramApprovalBot**：封装 Telegram Bot API 调用的可选组件，负责发送通知消息和处理回调响应。
- **HitlGateNode（hitl_gate 节点）**：LangGraph 节点函数，评估触发条件，触发时调用 `interrupt()`，不触发时透传。
- **ApprovalStore（审批持久化）**：SQLite 表，存储 `ApprovalRequest` 的完整生命周期记录。

---

## 成功标准

### 可测量的结果

- **SC-001**：当 `hitl.enabled=false`（默认值），全量图的 E2E 测试通过率不变（零回归）。

- **SC-002**：当 `hitl.enabled=true` 且 `position_scale >= min_position_scale`，图必须在 `hitl_gate` 处暂停，持续时间 ≥ 0 秒，直到收到响应或超时。

- **SC-003**：超时场景：`approval_timeout_seconds=5`（测试配置），若 5 秒内无响应，图以 `reject` 恢复，`ApprovalRequest.status` 更新为 `expired`，不发出任何下单请求。

- **SC-004**：Telegram 配置有效时，审批请求创建后 ≤ 3 秒内，Telegram Bot 消息可查询（Mock Bot API 验证）。

- **SC-005**：Web 前端 `GET /api/hitl/pending` 返回的待审批列表中，每条记录的 `verdict_snapshot` 包含完整的 `action, position_scale, confidence, reasoning` 字段。

- **SC-006**：并发安全：同一 `approval_id` 的两个并发 `POST respond` 请求，恰好一个返回 200，另一个返回 409。

- **SC-007**：回测模式：`backtest_mode=true` 时，`hitl_gate` 节点耗时 < 1ms（无任何 I/O），`ApprovalStore` 零写入。

- **SC-008**：`hitl.enabled=true` 的全量图，平均每个审批请求的端到端延迟（从 `hitl_gate` 触发到恢复）等于人工响应时间，系统本身引入的额外延迟 < 200ms。

---

## 假设

1. **LangGraph `interrupt()` 可用**：当前 `langgraph>=1.0.10` 版本已支持 `interrupt()` 语义（Human-in-the-loop checkpointing）；若实际版本不支持，需升级 LangGraph 至支持版本（预计 1.2+）。

2. **单用户自托管**：系统没有多用户权限控制需求，审批 API 端点依赖现有的 `X-API-Key` 验证即可。

3. **Telegram Bot 轮询或 Webhook**：实现可选择长轮询（polling，适合自托管）或 Webhook（适合云部署），具体选择在设计阶段决定；本 spec 不预设实现方式。

4. **SQLite 持久化已就绪**：系统已有 SQLite 基础设施（`db.py` 共享工厂）；`ApprovalStore` 直接在同一数据库文件中新增表，无需引入新依赖。

5. **不阻塞其他币对**：每个币对的分析周期运行于独立的 LangGraph 图实例，`hitl_gate` 挂起仅影响当前图实例，不影响其他币对的并发运行。

6. **`hitl.enabled` 默认为 `false`**：新功能对所有现有用户完全透明，只有显式开启才生效。这确保了零回归风险。

7. **前端审批卡片属于 `/risk` 页面**：不新增路由，审批卡片作为 `RiskPage` 的新区块嵌入；若待审批列表为空，该区块不显示。

8. **超时必须 fail-closed**：这是硬约束，不可通过配置修改。超时后唯一可能的行为是拒绝，不允许配置"超时后自动批准"。

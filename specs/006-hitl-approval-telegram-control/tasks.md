# 实施任务清单：人工审批节点 + Telegram 远程控制（HITL）

**Feature Branch**: `006-hitl-approval-telegram-control`
**版本**: 1.0（2026-04-17）
**预估总任务数**: 40 个

---

## 任务格式说明

- `[P]` = 可与同阶段其他 `[P]` 任务并行执行
- 无标记 = 需按序执行（依赖前置任务）
- 每个任务粒度约 1–3 小时

---

## Phase 1：配置层（Config）

> 目标：新增 `HitlConfig` / `TelegramConfig` dataclass，向后兼容，零回归。

- [X] T001 在 `src/cryptotrader/config.py` 中新增 `TelegramConfig` dataclass，字段：`enabled: bool = False`、`bot_token: str = ""`、`chat_id: str = ""`
- [X] T002 在 `src/cryptotrader/config.py` 中新增 `HitlConfig` dataclass，字段：`enabled`、`min_position_scale`、`divergence_threshold`、`cold_start_min_trades`、`approval_timeout_seconds`、`telegram: TelegramConfig`，所有字段含默认值
- [X] T003 在 `AppConfig` 中新增 `hitl: HitlConfig = field(default_factory=HitlConfig)` 字段（`src/cryptotrader/config.py`）
- [X] T004 在 `config/default.toml` 中新增 `[hitl]` 和 `[hitl.telegram]` 配置段，值与 dataclass 默认值一致
- [X] T005 [P] 在 `src/cryptotrader/config.py` 的 `_build_config()` 函数中新增 `[hitl]` 段的解析逻辑，支持 `[hitl.telegram]` 嵌套；验证 `hitl.min_position_scale` 在 `(0, 1)` 区间内（扩展 `validate_config()`）
- [X] T006 [P] 编写 `tests/test_hitl_config.py`：验证默认值加载、`local.toml` 无 `[hitl]` 段时降级、env override `CRYPTOTRADER_HITL__ENABLED=true` 生效、`validate_config()` 对越界 `min_position_scale` 抛出 `ConfigurationError`

---

## Phase 2：持久化层（ApprovalStore）

> 目标：新增 `hitl_approvals` SQLite 表，实现 CRUD 和并发安全决策。

- [X] T007 在 `src/cryptotrader/hitl/__init__.py` 中创建空 `hitl` 子包
- [X] T008 在 `src/cryptotrader/hitl/store.py` 中定义 `ApprovalRequest` SQLAlchemy 模型（字段见 plan.md 数据模型节），使用 `DeclarativeBase`，沿用 `journal/store.py` 的 `_sa_models()` 延迟加载模式
- [X] T009 在 `src/cryptotrader/hitl/store.py` 中实现 `ApprovalStore` 类，方法：
  - `async def ensure_table(db_url: str)` — 建表（幂等）
  - `async def create(db_url, approval_id, pair, expires_at, trigger_reason, verdict_snapshot, agent_analyses_snapshot, thread_id) -> ApprovalRequest`
  - `async def get(db_url, approval_id) -> ApprovalRequest | None`
  - `async def list_pending(db_url) -> list[ApprovalRequest]`
  - `async def decide(db_url, approval_id, status, decision_by, comment) -> bool` — CAS 写法，返回 `False` 表示并发冲突
  - `async def set_telegram_message_id(db_url, approval_id, message_id)`
  - `async def expire_stale(db_url) -> int` — 启动时扫描并过期已超时 pending 记录，返回处理数量
- [X] T010 编写 `tests/test_hitl_store.py`：
  - `test_create_and_get` — 创建后读取字段正确
  - `test_list_pending` — 仅返回 pending 状态记录
  - `test_decide_approve` — 状态机正确流转
  - `test_decide_concurrent_409` — 两个并发 `decide()` 调用，恰好一个返回 `True`，一个返回 `False`
  - `test_expire_stale` — 过期 pending 记录被标记为 `expired`，有效期内记录不受影响

---

## Phase 3：`hitl_gate` LangGraph 节点 + 图接线

> 目标：实现核心 HITL 节点，插入全量图，超时自动拒绝。

- [X] T000 **（前置验证）** 验证 `from langgraph.types import interrupt` 和 `from langgraph.types import Command` 可用性：
  - 在 Python REPL 中执行 `from langgraph.types import interrupt, Command` 确认无 `ImportError`
  - 若当前已安装版本不支持（`langgraph < 1.2`），在 `pyproject.toml` 中将约束升级为 `langgraph>=1.2`，执行 `uv sync`，确认现有 `pytest tests/ -x -q` 全部通过后再继续

- [X] T011 在 `src/cryptotrader/state.py` 中新增 `HitlState(TypedDict, total=False)` 类型定义，字段：`approval_id`、`decision`、`trigger_reason`、`skipped`；在 `ArenaState` 中新增 `hitl: HitlState` 可选字段
- [X] T012 在 `src/cryptotrader/hitl/gate.py` 中实现触发条件检测函数 `_should_trigger(state: ArenaState, config: HitlConfig) -> tuple[bool, str]`，返回 `(should_trigger, trigger_reason)`，检查：
  - `action == "hold"` → 不触发
  - `backtest_mode == True` → 不触发
  - `hitl.enabled == False` → 不触发
  - `position_scale >= min_position_scale` → 触发，reason = `"position_scale"`
  - `divergence_scores[-1] >= divergence_threshold` → 触发，reason = `"divergence"`
  - 已完成交易数 < `cold_start_min_trades` → 触发，reason = `"cold_start"`
  - **冷启动数据来源**：调用 `await ApprovalStore.get_completed_trades_count(db_url)`，通过 `SELECT COUNT(*) FROM decisions WHERE action != 'hold'` 查询 journal 表，而非 `hitl_approvals` 表（journal 包含所有历史执行交易，能准确反映系统实际冷启动状态）
- [X] T013 在 `src/cryptotrader/hitl/gate.py` 中实现 `_timeout_reject(approval_id, timeout_s, compiled_graph, thread_id, db_url)` 协程：等待 `timeout_s` 秒，检查 DB 状态，若仍为 `pending` 则调用 `ApprovalStore.decide()` 写入 `expired`，然后通过 `compiled_graph.ainvoke(None, config={"configurable": {"thread_id": thread_id}})` 恢复图
- [X] T014 在 `src/cryptotrader/hitl/gate.py` 中实现 `hitl_gate(state: ArenaState) -> dict` 异步节点函数：
  - 不触发时返回 `{"hitl": {"skipped": True, "decision": "approve"}}`
  - 触发时：创建 `ApprovalRequest`，调用 `interrupt(approval_id)`，恢复后从 `state["hitl"]["decision"]` 读取决策结果
  - 启动 `_timeout_reject` task
- [X] T015 在 `src/cryptotrader/hitl/gate.py` 中实现 `_hitl_router(state: ArenaState) -> str`：`decision in ("approve", "")` → `"pass"`；`decision in ("reject", "expired")` → `"rejected"`
- [X] T016 修改 `src/cryptotrader/graph.py`：
  - **仅修改 `_build_full_graph()`**：将 `graph.add_edge("verdict", "risk_gate")` 替换为：添加 `hitl_gate` 节点，添加 `"verdict" → "hitl_gate"` 边，添加 `add_conditional_edges("hitl_gate", _hitl_router, {"pass": "risk_gate", "rejected": "record_rejection"})`
  - **仅在 `_build_full_graph()` 的 `graph.compile()` 调用** 改为 `graph.compile(checkpointer=MemorySaver())`（从 `langgraph.checkpoint.memory` 导入）
  - **`build_lite_graph()` 和 `build_backtest_graph()` 不做任何修改**：这两个图变体无 `risk_gate`，FR-009/FR-012 明确禁止插入 `hitl_gate`，`compile()` 调用保持原样
  - 在 `__all__` 中导出 `hitl_gate`、`_hitl_router`
- [X] T016b 搜索所有 `build_trading_graph()` 调用点，为每个调用点补充 `thread_id` config 注入：
  - 搜索范围：`src/cryptotrader/scheduler/scheduler.py`、`src/api/routes/`（`analyze`、`backtest` 等路由）、`src/cli/main.py`、`tests/`
  - 对每个调用点，在 `.invoke()` / `.ainvoke()` 时传入 `config={"configurable": {"thread_id": str(uuid4())}}`
  - 同时将 `thread_id` 写入 `state["metadata"]["thread_id"]`（供 `hitl_gate` 读取并存入 `ApprovalRequest`）
  - 确认 `build_lite_graph()` 和 `build_backtest_graph()` 的调用点**不需要**此修改（这两个图变体未插入 `hitl_gate`，无 checkpointer）

- [X] T017 编写 `tests/test_hitl_gate.py`：
  - `test_passthrough_disabled` — `hitl.enabled=False` 时节点透传，`decision="approve"，skipped=True`
  - `test_passthrough_backtest` — `backtest_mode=True` 时节点透传
  - `test_passthrough_hold_action` — `action="hold"` 时不触发
  - `test_trigger_position_scale` — `position_scale >= min_position_scale` 触发，`trigger_reason="position_scale"`
  - `test_trigger_divergence` — `divergence_scores[-1] >= threshold` 触发
  - `test_trigger_cold_start` — 已完成交易数不足时触发
  - `test_timeout_auto_reject` — 超时后 `decision="expired"`，`ApprovalStore` 状态为 `expired`（SC-003）
  - `test_hitl_router_pass` — `decision="approve"` 路由到 `"pass"`
  - `test_hitl_router_rejected` — `decision="reject"` / `"expired"` 路由到 `"rejected"`

---

## Phase 4：Web API 端点

> 目标：实现三个 HITL REST API 端点，接入 ApprovalStore 和图恢复逻辑。

- [X] T018 在 `src/api/routes/hitl.py` 中定义 Pydantic 模型：`ApprovalRequestOut`、`HitlRespondIn`、`HitlRespondOut`（字段见 plan.md）
- [X] T019 在 `src/api/routes/hitl.py` 中实现 `GET /api/hitl/pending` 端点：调用 `ApprovalStore.list_pending()`，返回 `list[ApprovalRequestOut]`；`verdict_snapshot` 和 `agent_analyses_snapshot` 反序列化为 dict/list
- [X] T020 在 `src/api/routes/hitl.py` 中实现 `GET /api/hitl/{approval_id}` 端点：调用 `ApprovalStore.get()`；不存在时返回 `404 Not Found`
- [X] T021 在 `src/api/routes/hitl.py` 中实现 `POST /api/hitl/{approval_id}/respond` 端点：
  - 调用 `ApprovalStore.decide()`；`False` 时返回 `409 Conflict`
  - 通过全局图实例引用（或应用状态）找到对应 `thread_id`，调用 `compiled_graph.ainvoke(None, config={"configurable": {"thread_id": thread_id}}, command=Command(resume={"decision": decision}))`
  - 通过 `HitlNotifier` 发送 `hitl_decision` 事件
  - 写入 structlog 结构化日志（`approval_id, pair, trigger_reason, decision, decision_by, latency_seconds`）
- [X] T022 在 `src/api/main.py` 中直接注册 `hitl.router`：
  - 在文件顶部 `from src.api.routes import hitl`（或 `from src.api.routes.hitl import router as hitl_router`）
  - 使用 `app.include_router(hitl.router, dependencies=[Depends(verify_api_key)])` 注册，与现有 `health`、`risk`、`scheduler` 等 router 的注册模式完全一致
  - **不在 `src/api/routes/__init__.py` 中重导出**（现有 router 均未经过 `__init__.py` 统一导出，保持一致性）
- [X] T023 编写 `tests/test_hitl_api.py`：
  - `test_get_pending_empty` — 无 pending 时返回空列表
  - `test_get_pending_returns_full_snapshot` — verdict_snapshot 包含 `action, position_scale, confidence, reasoning`（SC-005）
  - `test_get_by_id_found` — 返回正确字段
  - `test_get_by_id_not_found` — 返回 404
  - `test_respond_approve` — 状态流转为 `approved`，图恢复（Mock graph.ainvoke）
  - `test_respond_reject` — 状态流转为 `rejected`
  - `test_respond_conflict_409` — 第二次响应返回 409（SC-006）
  - `test_respond_expired_409` — 已过期请求返回 409

---

## Phase 5：前端审批卡片

> 目标：在 `/risk` 页面展示待审批列表，支持实时倒计时和一键审批/拒绝。

- [X] T024 [P] 在 `web/src/types/api.schema.ts` 中新增 `AgentAnalysisSummarySchema`、`ApprovalRequestSchema`、`HitlPendingListSchema`、`HitlRespondSchema`（使用 Zod，字段见 plan.md）
- [X] T025 [P] 新建 `web/src/hooks/use-hitl-approvals.ts`，实现：
  - `useHitlPending()` — `useQuery`，`queryKey: ['hitl-pending']`，`refetchInterval: 5000`，调用 `GET /api/hitl/pending`
  - `useHitlRespond()` — `useMutation`，调用 `POST /api/hitl/{approval_id}/respond`，成功后 `invalidateQueries(['hitl-pending'])`
- [X] T026 新建 `web/src/pages/risk/components/approval-item.tsx`，渲染单条审批卡片：
  - 顶部：币对名、触发原因（中文标签：大额仓位 / Agent 分歧 / 冷启动保护）
  - 裁决区：颜色编码动作（多头绿/空头红/平仓橙）、position_scale（百分比）、置信度
  - Agent 分析区：4 个 Agent 方向图标 + 置信度数值
  - 倒计时：基于 `expires_at` 实时更新（`useEffect` + `setInterval(1000)`），剩余 < 60 秒时红色警告
  - 操作区：批准（绿色按钮）/ 拒绝（红色按钮），点击时显示确认 Dialog，提交时调用 `useHitlRespond()`
- [X] T027 新建 `web/src/pages/risk/components/approval-queue-card.tsx`：使用 `useHitlPending()`，pending 列表为空时不渲染该区块（无占位空间），非空时渲染卡片标题 + `ApprovalItem` 列表
- [X] T028 修改 `web/src/pages/risk/index.tsx`：在 `<ThresholdsCard />` 之后条件渲染 `<ApprovalQueueCard />`；添加对应 i18n key（`risk.hitl.*`）
- [X] T029 在 `web/src/locales/zh-CN/risk.json` 和 `web/src/locales/en-US/risk.json` 中新增 HITL 相关 i18n 文本：触发原因标签、卡片标题、按钮文本、确认对话框文本

---

## Phase 6：Telegram Bot

> 目标：实现可选 Telegram Bot，发送审批通知并处理内联按钮回调。

- [X] T030 在 `src/cryptotrader/hitl/telegram.py` 中实现 `TelegramApprovalBot` 类：
  - `__init__(bot_token, chat_id, approval_store_ref, compiled_graph_ref, db_url)` — 延迟 import `python-telegram-bot`
  - `async def send_approval_request(approval: ApprovalRequest) -> int` — 发送包含摘要和"批准"/"拒绝"内联按钮的 Telegram 消息，返回 `message_id`；调用后将 `message_id` 写入 DB
  - `async def _handle_callback(update, context)` — 解析 `callback_data`（格式：`hitl:{approval_id}:{decision}`），调用 `ApprovalStore.decide()`，恢复图，编辑原消息为决策结果
  - `async def update_message_decided(message_id, decision, decision_by)` — 编辑已发送消息内容为"已批准 ✓" 或 "已拒绝 ✗" 或 "已超时 — 自动拒绝"
  - `async def start()` / `async def stop()` — 启动/停止长轮询 Application
- [X] T031 在 `src/cryptotrader/hitl/notifier.py` 中实现 `HitlNotifier`，将 `hitl_decision` 事件通过现有 `Notifier`（`notifications.webhook_url`）发送，事件格式：`{event: "hitl_decision", approval_id, pair, trigger_reason, decision, decision_by, latency_seconds}`
- [X] T032 修改 `src/api/main.py` 的 `lifespan()` 函数：
  - 启动时：若 `config.hitl.telegram.enabled` 且 `bot_token` 非空，则实例化 `TelegramApprovalBot` 并调用 `start()`，存入 `app.state.telegram_bot`
  - 启动时：调用 `ApprovalStore.expire_stale(db_url)` 处理进程重启后的残留过期记录
  - 关闭时：若 `app.state.telegram_bot` 存在，调用 `stop()`
- [X] T033 编写 `tests/test_hitl_telegram.py`（Mock `telegram.Bot` 和 `telegram.ext.Application`）：
  - `test_send_approval_request` — Bot 发送消息，返回 message_id，DB 中 `telegram_message_id` 更新
  - `test_callback_approve` — `callback_query` 触发，`ApprovalStore.decide()` 被调用，消息更新为"已批准"
  - `test_callback_reject` — 拒绝回调流程
  - `test_callback_already_decided_409` — 重复回调不再更新
  - `test_timeout_updates_telegram_message` — 超时后 Telegram 消息更新为"已超时"

---

## Phase 7：集成测试

> 目标：端到端验证完整 HITL 流程，覆盖所有成功标准（SC-001 ~ SC-008）。

- [X] T034 [P] 编写 `tests/test_hitl_integration.py::test_full_graph_hitl_disabled`：`hitl.enabled=False` 时全量图 E2E 通过，无任何 HITL 副作用（SC-001）
- [X] T035 [P] 编写 `tests/test_hitl_integration.py::test_full_graph_hitl_triggers_and_approves`：`hitl.enabled=True`，构造 `position_scale=0.75` 的 ArenaState，图在 `hitl_gate` 暂停，API `POST /respond` 触发 `approve`，图恢复到 `risk_gate`（SC-002）
- [X] T036 [P] 编写 `tests/test_hitl_integration.py::test_full_graph_hitl_timeout_reject`：`approval_timeout_seconds=2`（测试配置），2 秒内无响应，图以 `reject` 恢复，`ApprovalRequest.status == "expired"`，无下单请求（SC-003）
- [X] T037 [P] 编写 `tests/test_hitl_integration.py::test_concurrent_respond_409`：同一 `approval_id` 的两个并发 `POST respond`，一个 200 一个 409（SC-006）
- [X] T038 [P] 编写 `tests/test_hitl_integration.py::test_backtest_mode_zero_io`：`backtest_mode=True` 时 `hitl_gate` 耗时 < 1ms，`ApprovalStore` 零写入（SC-007）
- [X] T039 [P] 编写 `tests/test_hitl_integration.py::test_pending_api_returns_full_snapshot`：`GET /api/hitl/pending` 返回 `verdict_snapshot` 包含完整 `action, position_scale, confidence, reasoning` 字段（SC-005）
- [X] T040 修改 `tests/test_integration.py` 中的全量图测试：确认所有现有测试在 `hitl.enabled=False`（默认）下仍全部通过，运行 `pytest tests/test_integration.py -v` 确认零回归（SC-001）

---

## 任务依赖关系

```
Phase 1 (T001-T006)
    └─→ Phase 2 (T007-T010)
            └─→ Phase 3 (T011-T017)
                    ├─→ Phase 4 (T018-T023)
                    │       └─→ Phase 7 T034-T040
                    ├─→ Phase 5 (T024-T029) [可与 Phase 4 并行]
                    └─→ Phase 6 (T030-T033) [可与 Phase 4、5 并行]
```

**Phase 5 前置**：T024（Zod schema）和 T025（hooks）可在 Phase 4 API 端点确定后立即并行开始；T026-T028 依赖 T024+T025。

**Phase 6 前置**：T030-T031 依赖 T009（ApprovalStore 接口）和 T014（hitl_gate 触发逻辑）；T032 依赖 T030+T031；T033 依赖 T030-T032。

---

## 完成标准检查清单

每个 Phase 完成后运行：

```bash
# 零 lint 错误（禁止 noqa）
uv run ruff check src/ tests/

# 全量测试通过
uv run pytest tests/ -x -q

# HITL 专项测试覆盖率
uv run pytest tests/test_hitl_*.py -v --tb=short
```

所有成功标准（SC-001 ~ SC-008）须有对应自动化测试覆盖，实现完成前不关闭任务。

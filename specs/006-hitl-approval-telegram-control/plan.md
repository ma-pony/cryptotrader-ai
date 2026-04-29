# 技术实施方案：人工审批节点 + Telegram 远程控制（HITL）

**Feature Branch**: `006-hitl-approval-telegram-control`
**版本**: 1.0（2026-04-17）

---

## 技术上下文

### 现有图拓扑分析

`build_trading_graph()` 的完整流水线（定义于 `src/cryptotrader/graph.py`）：

```
START → collect_data → update_pnl → stop_loss_check
  → [continue] → inject_experience → 4 agents（并行）→ debate_gate
  → [debate]   → debate_round_1 → debate_round_2 → enrich_context
  → [skip]     → enrich_context
  → verdict → risk_gate
  → [approved] → execute → record_trade → END
  → [rejected] → record_rejection → END
```

**插入点**：`hitl_gate` 节点置于 `verdict` 之后、`risk_gate` 之前。原有边 `verdict → risk_gate` 替换为：

```
verdict → hitl_gate → [pass]     → risk_gate
                    → [rejected] → record_rejection
```

**其他三个图变体不受影响**：
- `build_lite_graph()`：无 `risk_gate`，不插入 `hitl_gate`
- `build_backtest_graph()`：FR-009 + FR-012 明确禁止插入，节点在 `backtest_mode=True` 时立即透传
- `build_debate_graph()`：无 `risk_gate`，不插入 `hitl_gate`

### LangGraph `interrupt()` 可用性

`langgraph>=1.0.10`（已在 `pyproject.toml`）支持 `interrupt()` 原语（Human-in-the-loop checkpointing）。核心行为：

- 节点内调用 `interrupt(value)` 时，图抛出 `GraphInterrupt` 异常并序列化当前 state 到 checkpointer
- 通过 `graph.invoke(None, config={"configurable": {"thread_id": "..."}})` 恢复执行
- **前置条件**：`build_trading_graph()` 编译时必须传入 checkpointer：`graph.compile(checkpointer=checkpointer)`；现有编译调用为 `graph.compile()`，需修改

### 现有 DB 模式

`src/cryptotrader/db.py` 提供 `get_async_session(database_url)` 工厂，按 `(url, loop_id)` 缓存引擎；`journal/store.py` 等模块直接使用该工厂。`ApprovalStore` 沿用相同模式，在同一 SQLite 数据库文件中新增 `hitl_approvals` 表，无需引入新基础设施。

### 现有 API 路由模式

`src/api/main.py` 中所有受保护路由通过 `app.include_router(router, dependencies=[Depends(verify_api_key)])` 注册。新增 HITL 路由遵循相同模式：在 `src/api/routes/hitl.py` 中定义 `router = APIRouter(prefix="/api/hitl", tags=["hitl"])`，在 `main.py` 中注册。

### 现有前端模式

Risk 页面（`web/src/pages/risk/index.tsx`）：
- 使用 `useRiskStatus()` Hook，每 5 秒轮询 `/api/risk/status`
- 已有 `CircuitBreakerCard`、`ThresholdsCard` 两个子组件
- 新增 `ApprovalQueueCard` 作为第三个子组件，置于已有卡片之下

Hook 模式（`web/src/hooks/use-risk-status.ts`）：
- `useQuery` + `refetchInterval` 轮询
- Zod schema 验证（`web/src/types/api.schema.ts`）
- `apiClient.get / post` 调用

---

## 架构决策

### 决策 1：使用 `interrupt()` + 外部等待，不使用轮询阻塞

**选择**：`hitl_gate` 节点调用 `interrupt(approval_id)` 挂起图。图暂停后，API 服务器在 `POST /api/hitl/{id}/respond` 接收响应后通过 `graph.invoke(None, ...)` 恢复图。超时通过独立 `asyncio.Task` 驱动，到期后以 `reject` 恢复图。

**拒绝备选方案**：节点内 `while True: await asyncio.sleep(1); check_db()` 轮询——会占用 event loop，与 LangGraph 异步图执行语义冲突。`interrupt()` 是 LangGraph 官方 HITL 机制，不阻塞 event loop，是唯一正确选择。

### 决策 2：每个图实例使用独立 thread_id，存储于 ArenaState.metadata

**选择**：`build_trading_graph()` 编译时传入 `MemorySaver`（LangGraph 内置内存 checkpointer）。每次图调用通过 `config={"configurable": {"thread_id": f"{pair}-{timestamp}"}}` 注入唯一标识，同时存入 `state["metadata"]["thread_id"]`。`hitl_gate` 节点从 `state["metadata"]` 读取 `thread_id` 并写入 `ApprovalRequest`，用于后续图恢复。

**理由**：不同币对并发运行时各自 `thread_id` 互不干扰，满足"A 币对审批挂起不影响 B 币对"要求（FR-012 + 边界条件）。

### 决策 3：SQLite 审批表使用乐观锁（CAS 状态检查）

**选择**：`ApprovalStore.decide()` 执行 `UPDATE hitl_approvals SET status=? WHERE approval_id=? AND status='pending'`；若受影响行数为 0，判定为并发冲突，返回 `409 Conflict`。不引入 Redis 分布式锁。

**理由**：SQLite WAL 模式保证单写安全；WHERE 条件的 CAS 写法满足 SC-006 并发安全要求，且无需 Redis（自托管场景中 Redis 可能不可用）。

### 决策 4：超时机制使用 `asyncio.Task`，不使用 APScheduler

**选择**：`hitl_gate` 节点触发审批后，通过 `loop.create_task(_timeout_reject(approval_id, timeout_s, compiled_graph, thread_id))` 创建独立异步任务。该任务 `await asyncio.sleep(timeout_s)` 后检查 DB 状态，若仍为 `pending` 则写入 `expired` 并恢复图。

**进程重启恢复**：应用启动时扫描 DB 中 `status='pending'` 且 `expires_at < NOW()` 的记录，自动标记为 `expired` 并写入拒绝日志；仍在有效期内的 pending 记录写入 WARNING 日志（需人工通过 API 处理）。

**理由**：APScheduler 适合持久化跨进程调度；HITL 超时生命周期与图实例绑定，进程内的 `asyncio.Task` 更轻量直接。

### 决策 5：Telegram Bot 使用长轮询（polling）模式，不使用 Webhook

**选择**：使用 `python-telegram-bot>=21.0` 的 `Application` 异步接口，在 FastAPI lifespan 管理的独立 `asyncio.Task` 中运行长轮询。`callback_query` 处理函数直接调用 `ApprovalStore.decide()`，再触发图恢复。

**理由**：自托管场景通常无公网 Webhook URL；长轮询对自托管用户零配置、开箱即用。Webhook 可作为未来增强（通过配置 `polling: false` 切换）。

### 决策 6：Telegram Bot 作为可选组件，在 FastAPI lifespan 中懒加载

**选择**：`lifespan()` 检查 `config.hitl.telegram.enabled` 和 `bot_token` 是否非空，满足时才实例化并启动 `TelegramApprovalBot`；否则跳过。`python-telegram-bot` 在 `pyproject.toml` 中标注为 optional extra（`[telegram]`）；`hitl/telegram.py` 使用延迟 import 保护，未安装时给出明确错误提示。

### 决策 7：`hitl_gate` 节点路由器返回 `"pass"` / `"rejected"` 两条路径

**选择**：`hitl_gate` 节点函数负责判断触发条件、挂起图、恢复后读取决策。图使用 `add_conditional_edges` 路由：
- `hitl.enabled=false`（透传）、`decision="approve"` → `"pass"` → `risk_gate`
- `decision="reject"` 或 `decision="expired"` → `"rejected"` → `record_rejection`

**理由**：复用现有 `record_rejection` 节点的完整拒绝记录逻辑，避免重复实现。

### 决策 8：HITL 作为独立子包，不混入现有 nodes/ 文件

**选择**：在 `src/cryptotrader/hitl/` 新建子包，包含 `gate.py`（LangGraph 节点函数）、`store.py`（`ApprovalStore`）、`telegram.py`（`TelegramApprovalBot`）、`notifier.py`（Notifier 集成）。

**理由**：HITL 是完整子系统，与现有节点层次平行。独立子包便于测试和维护，遵循 `learning/`、`debate/`、`risk/` 等现有子包的架构模式。

---

## 文件结构（新增 / 修改）

```
config/
  default.toml                          # 修改：新增 [hitl] 和 [hitl.telegram] 配置段

src/cryptotrader/
  config.py                             # 修改：新增 TelegramConfig、HitlConfig；AppConfig.hitl 字段

  hitl/                                 # 新增子包
    __init__.py
    store.py                            # ApprovalStore：SQLAlchemy 模型 + CRUD（create/get/decide/list_pending）
    gate.py                             # hitl_gate 节点函数、_hitl_router、_timeout_reject 协程
    telegram.py                         # TelegramApprovalBot（可选，延迟 import）
    notifier.py                         # hitl_decision 事件 → 现有 Notifier 集成

  graph.py                              # 修改：_build_full_graph() 插入 hitl_gate + 条件边；compile(checkpointer=)
  state.py                              # 修改：ArenaState 新增 hitl: HitlState 可选字段

src/api/
  routes/
    hitl.py                             # 新增：3 个端点（GET pending、GET {id}、POST {id}/respond）
    __init__.py                         # 修改：导出 hitl router
  main.py                               # 修改：注册 hitl router；lifespan 启动/关闭 TelegramApprovalBot

web/src/
  types/
    api.schema.ts                       # 修改：新增 ApprovalRequestSchema、HitlPendingListSchema、HitlRespondSchema
  hooks/
    use-hitl-approvals.ts               # 新增：useHitlPending（轮询）、useHitlRespond（mutation）
  pages/
    risk/
      components/
        approval-queue-card.tsx         # 新增：审批队列卡片（列表容器）
        approval-item.tsx               # 新增：单条审批卡片（倒计时 + 批准/拒绝按钮）
      index.tsx                         # 修改：条件渲染 ApprovalQueueCard

tests/
  test_hitl_store.py                    # 新增：CRUD、并发 CAS 409、超时过期扫描
  test_hitl_gate.py                     # 新增：触发条件逻辑、透传路径、backtest 跳过
  test_hitl_api.py                      # 新增：3 个 API 端点、409 冲突、404 不存在
  test_hitl_telegram.py                 # 新增：Mock Bot API 发送、callback_query 处理
  test_hitl_integration.py             # 新增：端到端：图暂停 → API 响应 → 图恢复
```

---

## 数据模型

### `config.py` 新增 Dataclass

```python
@dataclass
class TelegramConfig:
    enabled: bool = False
    bot_token: str = ""
    chat_id: str = ""


@dataclass
class HitlConfig:
    enabled: bool = False
    min_position_scale: float = 0.5
    divergence_threshold: float = 0.6
    cold_start_min_trades: int = 5
    approval_timeout_seconds: int = 300
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
```

`AppConfig` 新增字段：
```python
hitl: HitlConfig = field(default_factory=HitlConfig)
```

### `config/default.toml` 新增段

```toml
[hitl]
enabled = false
min_position_scale = 0.5
divergence_threshold = 0.6
cold_start_min_trades = 5
approval_timeout_seconds = 300

[hitl.telegram]
enabled = false
bot_token = ""
chat_id = ""
```

### SQLAlchemy 模型（`src/cryptotrader/hitl/store.py`）

```python
class ApprovalRequest(Base):
    __tablename__ = "hitl_approvals"

    approval_id             = Column(String(36),  primary_key=True)   # UUID4
    pair                    = Column(String(20),  nullable=False, index=True)
    created_at              = Column(DateTime(timezone=True), nullable=False)
    expires_at              = Column(DateTime(timezone=True), nullable=False)
    trigger_reason          = Column(String(50),  nullable=False)     # position_scale | divergence | cold_start
    verdict_snapshot        = Column(Text,        nullable=False)     # JSON 字符串
    agent_analyses_snapshot = Column(Text,        nullable=False)     # JSON 字符串
    status                  = Column(String(20),  nullable=False, default="pending", index=True)
    decision_by             = Column(String(20),  nullable=True)      # web | telegram | timeout
    decided_at              = Column(DateTime(timezone=True), nullable=True)
    comment                 = Column(Text,        nullable=True)
    thread_id               = Column(String(100), nullable=False)     # LangGraph thread_id，用于恢复图
    telegram_message_id     = Column(Integer,     nullable=True)      # 用于编辑 Telegram 消息
```

### 冷启动检测：`completed_trades_count` 数据来源

`hitl_gate` 触发条件中"已完成交易数 < `cold_start_min_trades`"所需的计数，通过查询 journal 表获取：

```python
# src/cryptotrader/hitl/store.py — ApprovalStore 提供辅助查询方法
async def get_completed_trades_count(db_url: str) -> int:
    async with get_async_session(db_url) as session:
        result = await session.execute(
            text("SELECT COUNT(*) FROM decisions WHERE action != 'hold'")
        )
        return result.scalar_one()
```

`hitl_gate` 节点在 `_should_trigger()` 中调用此方法。选择 journal 表而非 `hitl_approvals` 表的原因：journal 表记录所有实际执行的交易（包括 HITL 介入前系统已自动执行的历史交易），能更准确地反映系统实际冷启动状态。

### `state.py` 新增 HitlState TypedDict

```python
class HitlState(TypedDict, total=False):
    approval_id:    str   # 当前审批请求 UUID
    decision:       str   # "approve" | "reject" | "expired" | ""
    trigger_reason: str   # "position_scale" | "divergence" | "cold_start" | ""
    skipped:        bool  # True = hitl.enabled=false 或 backtest_mode=True
```

`ArenaState` 新增：
```python
hitl: HitlState
```

### Pydantic API 响应模型（`src/api/routes/hitl.py`）

```python
class ApprovalRequestOut(BaseModel):
    approval_id:             str
    pair:                    str
    created_at:              str
    expires_at:              str
    trigger_reason:          str
    verdict_snapshot:        dict
    agent_analyses_snapshot: list[dict]
    status:                  Literal["pending", "approved", "rejected", "expired"]
    decision_by:             str | None
    decided_at:              str | None

class HitlRespondIn(BaseModel):
    decision: Literal["approve", "reject"]
    comment:  str = ""

class HitlRespondOut(BaseModel):
    approval_id: str
    status:      str
    message:     str
```

### 前端 Zod Schema（`web/src/types/api.schema.ts`）

```typescript
export const AgentAnalysisSummarySchema = z.object({
  agent:      z.string(),
  direction:  z.enum(['bullish', 'bearish', 'neutral']),
  confidence: z.number(),
  reasoning:  z.string().optional(),
});

export const ApprovalRequestSchema = z.object({
  approval_id:    z.string(),
  pair:           z.string(),
  created_at:     z.string(),
  expires_at:     z.string(),
  trigger_reason: z.enum(['position_scale', 'divergence', 'cold_start']),
  verdict_snapshot: z.object({
    action:         z.string(),
    position_scale: z.number(),
    confidence:     z.number(),
    reasoning:      z.string().optional(),
  }),
  agent_analyses_snapshot: z.array(AgentAnalysisSummarySchema),
  status:      z.enum(['pending', 'approved', 'rejected', 'expired']),
  decision_by: z.string().nullable(),
  decided_at:  z.string().nullable(),
});

export const HitlPendingListSchema = z.array(ApprovalRequestSchema);

export const HitlRespondSchema = z.object({
  approval_id: z.string(),
  status:      z.string(),
  message:     z.string(),
});
```

---

## 向后兼容性保证

| 变更点 | 现有调用方 | 保证 |
|--------|-----------|------|
| `AppConfig` 新增 `hitl` 字段 | `load_config()`、所有使用 config 的模块 | `HitlConfig` 全字段有默认值；旧 `local.toml` 无 `[hitl]` 段时降级使用默认值（`enabled=false`） |
| `build_trading_graph()` 插入 `hitl_gate` | scheduler、CLI、analyze 路由 | `hitl.enabled=false`（默认值）时 `hitl_gate` 直接透传，执行路径零变化；SC-001 保证零回归 |
| `graph.compile(checkpointer=MemorySaver())` | `test_integration.py` 等直接调用图的测试 | `MemorySaver` 无外部依赖；测试无需额外配置 |
| `ArenaState` 新增 `hitl` 字段 | 所有使用 ArenaState 的节点和测试 | `TypedDict(total=False)` 可选字段；现有代码不访问 `hitl` 键时零影响 |
| `src/api/main.py` 新增 router 和 lifespan 逻辑 | 现有 API 测试 | Telegram lifespan 逻辑被 `if config.hitl.telegram.enabled` 保护，默认不执行 |
| `python-telegram-bot` 新增 optional 依赖 | 不使用 Telegram 的环境 | 标注为 `[telegram]` optional extra；代码通过延迟 import 保护，未安装时不影响任何其他功能 |

---

## 依赖变更

### Python（可选）

```toml
# pyproject.toml
[project.optional-dependencies]
telegram = ["python-telegram-bot>=21.0"]
```

安装命令（仅需 Telegram 功能时）：
```
uv add "python-telegram-bot>=21.0"
```

`hitl/telegram.py` 通过延迟 import 保护：

```python
def _require_telegram():
    try:
        import telegram
        return telegram
    except ImportError as exc:
        raise RuntimeError(
            "python-telegram-bot 未安装。请运行: uv add 'python-telegram-bot>=21.0'"
        ) from exc
```

### 前端

无新 npm 依赖。审批卡片倒计时使用 `useState` + `useEffect` + `setInterval` 实现，无需额外时间库。

---

## 风险与缓解

| 风险 | 严重度 | 缓解措施 |
|------|--------|---------|
| LangGraph `interrupt()` 在 v1.0.10 行为与预期不符 | 高 | Phase 3 开始前编写最小化 `interrupt()` 验证测试；若失败则升级至 LangGraph 最新版本 |
| `MemorySaver` 进程重启后丢失挂起 state | 中 | 应用启动时扫描 DB 中已过期 pending 记录并自动拒绝；有效期内的 pending 记录写入 WARNING 日志 |
| 同一币对高频触发 HITL 导致审批队列积压 | 中 | `hitl_gate` 检测到同币对已有 `pending` 审批时直接拒绝新裁决（fail-closed），写入 `hitl_queue_full` 结构化日志 |
| Telegram Bot 长轮询阻塞 FastAPI event loop | 中 | `TelegramApprovalBot` 在独立 `asyncio.Task` 中运行，使用 `python-telegram-bot` 的原生 async API，不阻塞主循环 |
| 并发 `approve` + `reject` 请求下 SQLite 写锁竞争 | 中 | `UPDATE WHERE status='pending'` CAS 写法 + 检查受影响行数；行数为 0 则返回 `409 Conflict` |
| `hitl.enabled=false` 透传路径引入额外延迟 | 低 | 透传为纯 Python 条件判断，无 I/O；backtest 模式同样满足 SC-007（< 1ms）要求 |
| `bot_token` 明文存储于 `local.toml` | 中 | 可通过 `CRYPTOTRADER_HITL__TELEGRAM__BOT_TOKEN` 环境变量覆盖（沿用现有 env override 机制）；`local.toml` 已在 `.gitignore` 中 |
| 前端倒计时与服务器时间不同步 | 低 | 倒计时基于 `expires_at`（UTC ISO 字符串）与本地 `Date.now()` 差值，仅作展示；实际状态以 API `status` 字段为准，5 秒轮询及时刷新 |

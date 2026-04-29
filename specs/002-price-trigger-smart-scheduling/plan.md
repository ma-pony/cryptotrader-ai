# 技术实施方案：价格触发器 + 智能调度

## 技术上下文

### 现有技术栈

**后端**
- Python 3.12+，FastAPI 0.135+，APScheduler 3.x（AsyncIOScheduler）
- SQLAlchemy 2.x 异步（asyncpg 驱动），SQLite（本地）/ Postgres（生产）
- Redis 7.3（可选，降级为内存 `_MemoryStore`）
- LangGraph 1.x，LangChain 1.2+
- httpx、structlog、prometheus-client

**前端**
- React 19.2 + Vite 7 + TypeScript 5.9 strict（`exactOptionalPropertyTypes`）
- React Query 5、React Hook Form + `@hookform/resolvers`、Zod 3、i18next
- Radix UI（已有 Dialog / Dropdown / Popover / Switch 等组件）
- 包管理：pnpm 10

### 相关现有模块

| 模块 | 路径 | 与本功能的关系 |
|------|------|----------------|
| APScheduler 调度器 | `src/cryptotrader/scheduler.py` | 现有调度宿主，需扩展以集成触发引擎 |
| AppConfig / SchedulerConfig | `src/cryptotrader/config.py` | 需新增 `TriggersConfig`、`TelegramConfig` 字段 |
| TOML 配置 | `config/default.toml` | 新增 `[triggers]`、`[notifications.telegram]` 节 |
| RedisStateManager | `src/cryptotrader/risk/state.py` | 冷却期 TTL 存储的现成实现，直接复用 `set(key, "1", ex=)` 模式 |
| Notifier（webhook） | `src/cryptotrader/notifications.py` | 扩展为多 Backend，新增 TelegramBackend |
| 调度器 API 路由 | `src/api/routes/scheduler.py` | 已有 `GET /api/scheduler/status`，本功能在同文件追加 CRUD 端点 |
| 共享 DB 会话工厂 | `src/cryptotrader/db.py` | 触发规则持久化直接调用 `get_async_session()` |
| FastAPI 入口 | `src/api/main.py` | 注册新路由、挂载触发引擎 lifespan |
| App.tsx + sidebar.tsx | `web/src/` | 新增 `/scheduler` 路由和导航条目 |
| i18n 体系 | `web/src/locales/` | 新增 `scheduler` 命名空间（zh-CN / en-US） |
| api.schema.ts / api.ts | `web/src/types/` | 新增调度器 CRUD 和历史 Zod Schema |
| apiClient | `web/src/lib/api-client.ts` | 现成的带 X-API-Key 认证的 fetch 封装，无需修改 |
| SchedulerCard | `web/src/pages/dashboard/components/scheduler-card.tsx` | 保留，管理页另建 |

### 约束条件

1. **单用户自托管**：无多租户，无用户维度权限；API Key 认证复用现有 `verify_api_key`。
2. **零 lint 错误**：ruff select 包含 ASYNC/TID251/S/N 等严格规则；新文件需加入 `pyproject.toml` 的 `per-file-ignores` TID251 豁免（触发引擎作为入口层调用 nodes/graph）。
3. **禁止 noqa**：所有 lint 问题必须从源头修正。
4. **测试覆盖率 ≥ 85%（新增模块）**：WebSocket 和 Telegram API 通过 mock 测试，不依赖外网。
5. **现有 742+ 测试全部通过**：本功能不得破坏任何已有测试。
6. **WebSocket 依赖 Binance 公共 API**：与 `exchange_id` 解耦，始终连接 Binance 公共 ticker stream。
7. **资金费率轮询（非 WebSocket）**：Binance 资金费率通过 REST 轮询（每 5 分钟），不需要永久连接。

---

## 架构决策

### 决策 1：触发引擎独立模块，不嵌入 Scheduler 类
- **选择**：新建 `src/cryptotrader/triggers/` 目录，`PriceTriggerEngine` 独立运行
- **理由**：`Scheduler` 负责 APScheduler 任务管理，`PriceTriggerEngine` 负责 WebSocket 监听和条件匹配，两者职责不同。分离后可以独立测试，也支持未来 API-only 模式中单独启动触发引擎（无需 APScheduler）。触发引擎通过回调注入 `run_pair` 函数，与调度器解耦。

### 决策 2：触发规则持久化用 SQLite（复用现有数据库），不新建独立存储
- **选择**：新增 `schedule_rules` 和 `trigger_events` 两张表，通过 SQLAlchemy ORM 管理，复用 `db.py` 的 `get_async_session()`
- **理由**：项目已使用 SQLAlchemy + alembic；SQLite 对单用户场景完全足够。避免引入新的存储依赖（MongoDB、独立 SQLite 文件等）。表结构通过 alembic migration 管理，保持迁移一致性。

### 决策 3：冷却期优先 Redis TTL，降级内存——复用 RedisStateManager
- **选择**：`PriceTriggerEngine` 持有 `RedisStateManager` 实例；冷却键格式 `trigger:cooldown:{rule_id}`
- **理由**：`RedisStateManager` 已经封装了 Redis/内存双模式，且有成熟的 `set(key, "1", ex=seconds)` 接口，无需重新造轮子。系统重启后 Redis TTL 自动恢复冷却状态，符合 SC-005 要求。

### 决策 4：Notifier 改造为多 Backend 模式（不替换接口）
- **选择**：`Notifier` 新增 `backends: list[NotifierBackend]`，`WebhookBackend` 封装原逻辑，`TelegramBackend` 新增；`notify()` 方法向所有启用的 backend 广播
- **理由**：向后兼容——现有所有调用 `Notifier(webhook_url=...).notify(...)` 的代码无需修改，`WebhookBackend` 自动注册当且仅当 `webhook_url` 非空。`TelegramBackend` 单独初始化，支持入站命令（`/status`）通过 polling 处理。

### 决策 5：Telegram Bot 使用 long-polling（不用 webhook 服务器）
- **选择**：`TelegramBackend` 开启一个后台 asyncio 任务做 getUpdates long-polling，只处理 `/status` 命令
- **理由**：单用户自托管场景下不需要公网 webhook URL；long-polling 实现简单，无需额外 HTTP 服务器端口，符合"自托管最简化"原则。依赖 `httpx.AsyncClient` 而非新的 Telegram SDK，避免引入大型依赖。

### 决策 6：规则 CRUD API 扩展到现有 `src/api/routes/scheduler.py`
- **选择**：在同文件的 `api_router`（prefix=/api/scheduler）中追加 `/rules` 和 `/triggers` 端点，而不是新建路由文件
- **理由**：所有调度相关端点聚合在一处，路由 prefix 语义明确（`/api/scheduler/*`）；`main.py` 已注册 `api_router`，无需修改路由注册代码。

### 决策 7：前端 `/scheduler` 页面复用现有 UI 组件，不引入新 UI 库
- **选择**：使用已有的 Radix UI Dialog（规则编辑弹窗）、shadcn Switch（启用/禁用开关）、Table（规则列表/历史）；Cron 说明用纯 JS 逻辑生成，不引入 cron 解析包
- **理由**：保持 bundle 轻量（目标 gzip <300 KB），避免新依赖的版本冲突。已有 `react-hook-form` + `zod` 处理表单验证，现成可用。

### 决策 8：Agent 自我调度（P2）通过 nodes/verdict.py 后置钩子实现
- **选择**：在 `nodes/verdict.py` 的 `run_verdict_node()` 后追加 `_process_schedule_follow_up()` 函数，检查分析结论中的 `schedule_follow_up` 字段并调用规则创建服务
- **理由**：verdict 节点是分析流程的终点，所有路径（AI verdict / weighted verdict）都会经过；在此处插入钩子逻辑最小化侵入面，且 `schedule_depth` 字段可在 `ArenaState.metadata` 中传递，不污染数据模型。

---

## 文件结构（新增 / 修改）

```
src/cryptotrader/
├── triggers/                        ← 新建目录（价格触发引擎）
│   ├── __init__.py                  ← 导出 PriceTriggerEngine、TriggerRule、TriggerEvent
│   ├── engine.py                    ← PriceTriggerEngine：WebSocket 监听 + 条件匹配
│   ├── models.py                    ← SQLAlchemy ORM 模型：ScheduleRule、TriggerEventRecord
│   ├── store.py                     ← 规则 CRUD 服务（async，依赖 db.py）
│   └── conditions.py                ← 四种触发条件判断逻辑（纯函数，易测试）
├── notifications.py                 ← 修改：多 Backend 架构 + TelegramBackend
├── config.py                        ← 修改：新增 TriggersConfig、TelegramConfig
└── scheduler.py                     ← 修改：集成 PriceTriggerEngine lifecycle

src/api/routes/
└── scheduler.py                     ← 修改：追加 /rules CRUD + /triggers 历史端点

src/api/main.py                      ← 修改：触发引擎 lifespan 集成

config/default.toml                  ← 修改：新增 [triggers]、[notifications.telegram] 节

web/src/
├── pages/scheduler/                 ← 新建目录（调度管理页）
│   ├── index.tsx                    ← 页面入口（React.lazy）
│   ├── components/
│   │   ├── rule-table.tsx           ← 规则列表表格
│   │   ├── rule-form-dialog.tsx     ← 规则编辑弹窗（含四类触发类型表单）
│   │   ├── template-selector.tsx    ← 模板卡片选择器（≥4 个预设模板）
│   │   ├── cron-editor.tsx          ← Cron 表达式编辑器（实时中文说明）
│   │   └── trigger-history-table.tsx← 触发历史表格（含展开详情）
│   └── hooks/
│       ├── use-rules.ts             ← React Query hooks（list/create/update/delete）
│       └── use-trigger-history.ts   ← React Query hook（分页历史查询）
├── types/
│   ├── api.schema.ts                ← 修改：新增 ScheduleRule/TriggerEvent Zod Schema
│   └── api.ts                       ← 修改：导出新类型
├── App.tsx                          ← 修改：新增 /scheduler 路由
├── components/layout/sidebar.tsx    ← 修改：新增调度器导航项
├── locales/zh-CN/
│   └── scheduler.json               ← 新建：中文翻译键
└── locales/en-US/
    └── scheduler.json               ← 新建：英文翻译键
```

**测试文件（新增）**
```
tests/
├── test_trigger_engine.py           ← PriceTriggerEngine 单测（mock WebSocket）
├── test_trigger_conditions.py       ← 四种条件判断纯函数单测
├── test_trigger_store.py            ← 规则 CRUD 服务单测（SQLite in-memory）
├── test_telegram_notifier.py        ← TelegramBackend 单测（mock httpx）
└── test_scheduler_api_rules.py      ← /api/scheduler/rules CRUD API 集成测试
```

---

## 数据模型

### ScheduleRule 表（`schedule_rules`）

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | UUID / str（主键） | 规则唯一标识 |
| `name` | str | 用户可读名称 |
| `trigger_type` | str ENUM | `price_threshold` / `pct_change` / `candle_pattern` / `funding_rate` |
| `pair` | str | 交易对，如 `BTC/USDT` |
| `parameters` | JSON | 触发参数（见下方说明） |
| `cooldown_minutes` | int | 冷却期分钟数，默认 30 |
| `enabled` | bool | 是否启用 |
| `ttl_expires_at` | datetime nullable | Agent 临时规则专用，过期后自动失效 |
| `created_by` | str | `user` 或 `agent` |
| `schedule_depth` | int | Agent 自我调度递归深度，默认 0 |
| `created_at` | datetime | 创建时间（UTC） |
| `updated_at` | datetime | 最后更新时间（UTC） |

**parameters 字段示例（按触发类型）**

```json
// price_threshold
{"direction": "below", "price": 80000.0}

// pct_change
{"window_minutes": 15, "threshold_pct": 3.0}

// candle_pattern
{"timeframe": "1h", "candle_count": 3, "direction": "bearish"}

// funding_rate
{"threshold_pct": 0.1}
```

### TriggerEventRecord 表（`trigger_events`）

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | UUID / str（主键） | 事件唯一标识 |
| `rule_id` | str（外键） | 关联的规则 ID |
| `triggered_at` | datetime | 触发时间（UTC） |
| `trigger_reason` | str | 人可读的触发原因，如 `BTC/USDT 价格跌至 79,800.00（阈值: 80,000）` |
| `price_snapshot` | JSON | 触发时的价格快照 `{pair, price, ts}` |
| `analysis_commit_id` | str nullable | 关联的分析记录 commit hash |
| `schedule_depth` | int | Agent 调度深度（0 = 用户规则触发） |
| `cooldown_skipped` | bool | 是否因冷却期而跳过（true = 冷却中未触发，仅记录） |

---

## API 设计

### 扩展 `/api/scheduler` 路由（在现有 `api_router` 上追加）

所有端点需 X-API-Key 认证（已在 `main.py` 中通过 `Depends(verify_api_key)` 保护）。

```
GET    /api/scheduler/rules                    ← 列出所有规则（支持 ?enabled=true 过滤）
POST   /api/scheduler/rules                    ← 创建规则（返回 201 + 规则对象）
GET    /api/scheduler/rules/{rule_id}          ← 获取单条规则
PUT    /api/scheduler/rules/{rule_id}          ← 全量更新规则
PATCH  /api/scheduler/rules/{rule_id}/toggle   ← 切换 enabled 状态（无需发送完整对象）
DELETE /api/scheduler/rules/{rule_id}          ← 删除规则（返回 204）
GET    /api/scheduler/triggers                 ← 查询触发历史（?page=&size=&rule_id=）
GET    /api/scheduler/triggers/{event_id}      ← 获取单条触发事件详情
```

**ScheduleRuleOut 响应模型（Pydantic）**

```python
class ScheduleRuleOut(BaseModel):
    id: str
    name: str
    trigger_type: str
    pair: str
    parameters: dict
    cooldown_minutes: int
    enabled: bool
    ttl_expires_at: datetime | None
    created_by: str
    schedule_depth: int
    created_at: datetime
    updated_at: datetime
    # 运行时字段（查询时计算）
    in_cooldown: bool           # 当前是否在冷却期
    last_triggered_at: datetime | None  # 最近一次触发时间
```

**TriggerEventOut 响应模型**

```python
class TriggerEventOut(BaseModel):
    id: str
    rule_id: str
    triggered_at: datetime
    trigger_reason: str
    price_snapshot: dict
    analysis_commit_id: str | None
    schedule_depth: int
    cooldown_skipped: bool
```

---

## 前端设计

### 新增页面：`/scheduler`

**组件树**

```
SchedulerPage (pages/scheduler/index.tsx)
├── PageHeader（标题 + "创建规则"按钮 + 模板选择按钮）
├── Tabs（"规则列表" / "触发历史"）
│   ├── RuleTable（规则列表表格，含 Switch 切换启用状态）
│   └── TriggerHistoryTable（触发历史，可展开详情行）
└── RuleFormDialog（弹窗，含 Cron 编辑器或触发参数表单）
    └── TemplateSelector（模板卡片选择器，4+ 模板）
```

**预置模板（≥4 个）**
1. BTC 价格跌破提醒（`price_threshold` / below / 用户填入价格）
2. ETH 短期异常波动（`pct_change` / 15 分钟 / 3%）
3. 资金费率异常（`funding_rate` / 0.1%）
4. BTC 连续阴线信号（`candle_pattern` / 1h / 3 根）

**Cron 表达式编辑器**：用户输入 cron 字符串 → 前端 JS 解析 → 实时展示中文说明（如"每 4 小时执行一次"）+ 下次触发时间（UTC）。用简单规则解析常见模式，复杂 cron 显示"自定义表达式"。不引入外部 cron-parser 包。

**React Query 轮询**：规则列表和状态通过 `useQuery` 每 5 秒刷新（`refetchInterval: 5000`）。

**路由**：`/scheduler`，在 `App.tsx` 通过 `React.lazy` 代码拆分，bundle 不影响主入口。

### 修改 Sidebar

新增导航项：
- 路由：`/scheduler`
- labelKey：`nav.scheduler`（新增翻译键）
- 图标：`CalendarClock`（lucide-react，已在依赖中）

---

## 依赖变更

### Python 后端

无需新增外部 pip 依赖：
- WebSocket 客户端：使用 `httpx`（已有）的 `httpx.AsyncClient` 与 Binance WebSocket 连接，或使用标准库 `asyncio` + `websockets`。
- **决策**：引入 `websockets>=13.0` 作为专用 WebSocket 库（更稳定的断线重连支持，且 `pyproject.toml` 中已有 `httpx`，但 `httpx` 的 WebSocket 支持为 experimental）。添加到 `pyproject.toml` dependencies。
- Telegram API：直接使用 `httpx.AsyncClient` 调用 Telegram Bot API（无需 `python-telegram-bot` 大型 SDK）

```toml
# pyproject.toml 新增
"websockets>=13.0",
```

### 前端（npm/pnpm）

不引入新包，使用已有：
- `react-hook-form` + `zod`：表单验证
- `@radix-ui/react-switch`：启用/禁用开关（**需新增**，目前 package.json 未包含）
- `@radix-ui/react-collapsible`：触发历史展开行（可用已有 Tabs 组件替代，或用原生 `<details>`）

```json
// package.json 新增
"@radix-ui/react-switch": "^1.1.2",
"@radix-ui/react-collapsible": "^1.1.2"
```

---

## 风险与缓解

| 风险 | 影响 | 缓解方案 |
|------|------|----------|
| Binance WebSocket 公共 API 限制 | 单个连接最多订阅 200 个 stream；初始规则 ≤50 条但仍需合并订阅 | 使用 Binance combined stream 端点（一个连接多 symbol）；`PriceTriggerEngine` 维护 `subscribed_pairs set`，按需订阅 |
| WebSocket 断线时规则匹配中断 | P0 触发功能完全失效 | 指数退避重连（1s→2s→4s→最大 60s）；重连后从 DB 重新加载所有启用规则；重连事件通知（`ws_reconnect` 事件类型） |
| 资金费率轮询与 Binance API 速率限制 | 轮询过频触发 429 | 默认 5 分钟轮询间隔（Binance 资金费率更新频率为 8 小时，5 分钟完全安全）；轮询 job 通过 APScheduler `IntervalTrigger` 管理 |
| SQLAlchemy alembic migration 版本冲突 | 部署时迁移失败 | 新增 migration 文件使用有意义的 revision ID 前缀；本地测试用 SQLite in-memory + `create_all()` 绕过 alembic |
| Telegram Bot Token 泄露 | 安全风险 | Token 只存 `config/local.toml`（gitignored）或环境变量 `CRYPTOTRADER_NOTIFICATIONS__TELEGRAM__BOT_TOKEN`；代码中不出现默认值 |
| Agent 自我调度递归爆炸 | 无限触发、DB 数据膨胀 | `schedule_depth` 计数器硬限 3 次；`verdict.py` 钩子检查深度上限后不注册新规则；TTL 上限 72 小时 |
| 前端 Cron 解析器不完整 | 部分表达式显示"自定义表达式"而非中文说明 | 覆盖最常用的 10 种模式（每 N 分钟、每小时、每 N 小时、每天、每周等）；复杂表达式 fallback 为原始字符串显示 |
| ruff TID251 规则：triggers/ 属于入口层 | 直接导入 `nodes/` 或 `graph.py` 时 lint 报错 | 在 `pyproject.toml` 中将 `src/cryptotrader/triggers/*.py` 添加到 TID251 豁免列表（与 `scheduler.py` 相同处理） |

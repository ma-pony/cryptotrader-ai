# 实施任务：价格触发器 + 智能调度

## Phase 1：基础设施——触发规则持久化 + DB 模型

- [X] T001 新建 `src/cryptotrader/triggers/__init__.py`，导出 `PriceTriggerEngine`、`TriggerRuleStore`，添加空包标识
- [X] T002 [P] 实现 ORM 模型在 `src/cryptotrader/triggers/models.py`：`ScheduleRule`（含 `id/name/trigger_type/pair/parameters/cooldown_minutes/enabled/ttl_expires_at/created_by/schedule_depth/created_at/updated_at`）和 `TriggerEventRecord`（含 `id/rule_id/triggered_at/trigger_reason/price_snapshot/analysis_commit_id/schedule_depth/cooldown_skipped`），使用 SQLAlchemy 2.x 声明式 ORM
- [X] T002a [P] 初始化数据库迁移方案（选择 create_all 方案，在 lifespan 中建表）：检查项目根目录是否已存在 `alembic/` 目录。**若不存在**：运行 `alembic init alembic` 生成目录结构，修改 `alembic/env.py` 使其从 `src/cryptotrader/db.py` 读取 engine URL（替换默认 `sqlalchemy.url` 配置）；更新 `alembic.ini` 的 `script_location` 指向正确路径。**若现有代码库倾向使用 `Base.metadata.create_all()` 方案**（与 `db.py` 的 `get_async_session()` 模式一致）：改为在 `src/cryptotrader/triggers/models.py` 模块加载时通过 `Base.metadata.create_all(bind=engine_sync)` 或在应用启动的 lifespan 中调用 `async_engine.run_sync(Base.metadata.create_all)` 创建表；在代码注释中说明生产迁移方案（alembic）待后续解耦 Sprint 补齐
- [X] T003 [P] 跳过（已选 create_all 方案） `alembic/versions/xxx_add_schedule_rules_and_trigger_events.py`，创建 `schedule_rules` 和 `trigger_events` 两张表（含外键约束和索引）；若 T002a 已选择 `create_all()` 方案，则跳过本任务，在 T002a 的 lifespan 逻辑中合并建表
- [X] T004 实现规则 CRUD 服务在 `src/cryptotrader/triggers/store.py`：`TriggerRuleStore` 类，方法包括 `list_rules(enabled_only=False)`、`get_rule(rule_id)`、`create_rule(data)`、`update_rule(rule_id, data)`、`toggle_rule(rule_id)`、`delete_rule(rule_id)`、`record_event(event_data)`、`list_events(page, size, rule_id=None)`、`cleanup_expired_rules()`；内部调用 `db.py` 的 `get_async_session()`
- [X] T005 [P] 编写单测 `tests/test_trigger_store.py`：使用 SQLite in-memory（`create_all()`）覆盖所有 CRUD 方法，验证过期规则清理逻辑，覆盖率 ≥ 85%

## Phase 2：核心功能——触发条件引擎

- [X] T006 [US1][US2][US3][US4] 实现四种触发条件判断函数在 `src/cryptotrader/triggers/conditions.py`：
  - `check_price_threshold(current_price, parameters)` → `bool`
  - `check_pct_change(current_price, reference_price, parameters)` → `bool`
  - `check_candle_pattern(candles, parameters)` → `bool`
  - `check_funding_rate(funding_rate, parameters)` → `bool`
  全部为纯函数，无 I/O，无副作用，参数格式与 `ScheduleRule.parameters` JSON 对应
- [X] T007 [P] 编写单测 `tests/test_trigger_conditions.py`：用表格驱动测试覆盖所有条件类型的正向、边界、负向场景（阈值临界值、零变化、不足 N 根 K 线等），覆盖率 100%
- [X] T008 [US1][US2][US3][US4] 实现 `PriceTriggerEngine` 核心类在 `src/cryptotrader/triggers/engine.py`：
  - `__init__(store, redis_state_manager, run_pair_callback, config)`
  - `start()` / `stop()` 异步生命周期方法
  - `_ws_connect()` 连接 Binance combined stream（`wss://stream.binance.com/stream?streams=btcusdt@ticker/...`）
  - `_ws_listen_loop()` 消息处理循环：解析 ticker，匹配所有绑定该交易对的启用规则
  - `_reconnect_with_backoff()` 指数退避重连（1s→2s→4s→最大 60s）
  - `_check_cooldown(rule_id)` / `_set_cooldown(rule_id, minutes)` 通过 `RedisStateManager` 管理冷却期键 `trigger:cooldown:{rule_id}`
  - `_dispatch(rule, price_snapshot)` 记录事件到 DB、通知 `run_pair_callback`
  - `reload_rules()` 从 DB 重新加载规则（断线重连后调用）
  - `_poll_funding_rates()` 通过 Binance REST 轮询资金费率，返回 `dict[pair, float]`
- [X] T009 [P] 编写单测 `tests/test_trigger_engine.py`：使用 `pytest-asyncio` + `unittest.mock`，mock WebSocket 消息流和 `RedisStateManager`，验证：触发时机（≤5s 响应）、冷却期防重复、断线重连规则不丢失、多交易对并发触发互不阻塞，覆盖率 ≥ 85%

## Phase 3：配置体系扩展

- [X] T010 [P] 修改 `src/cryptotrader/config.py`：
  - 新增 `TriggersConfig` dataclass（`enabled: bool`、`max_rules: int = 50`、`ws_reconnect_max_s: int = 60`、`funding_rate_poll_interval_minutes: int = 5`）
  - 新增 `TelegramConfig` dataclass（`bot_token: str = ""`、`chat_id: str = ""`、`enabled: bool = False`）
  - 在 `NotificationsConfig` 中新增 `telegram: TelegramConfig` 字段（默认实例化）
  - 在 `AppConfig` 中新增 `triggers: TriggersConfig` 字段
  - 在 `_build_config()` 中解析 `[triggers]` 和 `[notifications.telegram]` 节
- [X] T011 [P] 修改 `config/default.toml`：新增 `[triggers]` 配置节（`enabled = false`、`max_rules = 50`、`ws_reconnect_max_s = 60`、`funding_rate_poll_interval_minutes = 5`）；在 `[notifications]` 下新增 `[notifications.telegram]` 节（`bot_token = ""`、`chat_id = ""`、`enabled = false`）
- [X] T012 修改 `pyproject.toml`：
  - 在 `dependencies` 中新增 `"websockets>=13.0"`
  - 在 `[tool.ruff.lint.per-file-ignores]` 中为 `"src/cryptotrader/triggers/*.py"` 添加 `["TID251"]` 豁免（触发引擎作为入口层）

## Phase 4：调度器集成

- [X] T013 修改 `src/cryptotrader/scheduler.py`：
  - 在 `Scheduler.__init__` 中接受可选的 `trigger_engine: PriceTriggerEngine | None = None`
  - 在 `start()` 中调用 `self._trigger_engine.start()` 启动触发引擎
  - 在 `stop()` / 优雅关闭路径中调用 `self._trigger_engine.stop()`
  - 新增 APScheduler `IntervalTrigger` 任务用于 `_poll_funding_rates()`（间隔由 `config.triggers.funding_rate_poll_interval_minutes` 决定）
  - 在 `pyproject.toml` TID251 豁免注释中补充 `triggers` 导入说明
- [X] T014 修改 `src/api/main.py`：在 `lifespan()` 上下文管理器的 startup 阶段初始化 `PriceTriggerEngine` 并挂载到 `app.state.trigger_engine`；shutdown 阶段调用 `trigger_engine.stop()`。当 `config.triggers.enabled` 为 false 时跳过初始化（不报错）

## Phase 5：通知系统扩展（Telegram）

- [X] T015 [US5] 重构 `src/cryptotrader/notifications.py`：
  - 抽象 `NotifierBackend` 协议（含 `async def send(event, data) -> None`）
  - 将原 webhook 逻辑提取为 `WebhookBackend`
  - 新增 `TelegramBackend`：`send()` 使用 `httpx.AsyncClient` 调用 `sendMessage` API，中文富文本 Markdown 格式，含指数退避重试（最多 3 次）
  - `TelegramBackend._start_polling()` 启动 getUpdates long-polling 后台任务，处理 `/status` 命令（回复调度器状态快照）
  - 修改 `Notifier.__init__` 为多 Backend 模式（向后兼容：`webhook_url` 非空时自动注册 `WebhookBackend`）
  - `notify()` 广播到所有 backend；发送失败的 backend 只记录 warning，不阻塞主流程
  - 新增支持事件类型：`price_trigger`（含 pair/trigger_type/price/trigger_reason）
- [X] T016 [P][US5] 编写单测 `tests/test_telegram_notifier.py`：mock `httpx.AsyncClient.post` 和 getUpdates 响应，验证：消息格式（含中文字段）、重试机制（3 次后放弃）、`/status` 命令响应、发送失败不抛异常，覆盖率 ≥ 85%

## Phase 6：CRUD API 端点

- [X] T017 [US6] 修改 `src/api/routes/scheduler.py`，在 `api_router`（prefix=/api/scheduler）中追加：
  - **鉴权说明**：新端点必须与现有调度器端点保持相同的认证保护。若 `api_router` 在 `main.py` 注册时已通过 `dependencies=[Depends(verify_api_key)]` 统一覆盖，则新端点自动继承；若未统一注册，需在创建 `APIRouter` 时声明 `dependencies=[Depends(verify_api_key)]`，或在 `main.py` 注册新路由时显式传入，确保所有 `/api/scheduler/rules` 和 `/api/scheduler/triggers` 端点均受 X-API-Key 鉴权保护
  - 新增 Pydantic 模型：`ScheduleRuleIn`（创建/更新请求体）、`ScheduleRuleOut`（响应，含 `in_cooldown`、`last_triggered_at` 运行时字段）、`TriggerEventOut`、`PaginatedTriggerEvents`
  - `GET /rules`：返回 `list[ScheduleRuleOut]`，支持 `?enabled=true` 过滤；从 `request.app.state.trigger_store` 获取 `TriggerRuleStore` 实例
  - `POST /rules`：创建规则，验证规则数量上限（`config.triggers.max_rules`），返回 201
  - `GET /rules/{rule_id}`：返回单条规则或 404
  - `PUT /rules/{rule_id}`：全量更新规则，触发引擎调用 `reload_rules()`
  - `PATCH /rules/{rule_id}/toggle`：切换启用状态，触发引擎调用 `reload_rules()`
  - `DELETE /rules/{rule_id}`：删除规则，返回 204
  - `GET /triggers`：分页查询触发历史，支持 `?rule_id=&page=&size=`
  - `GET /triggers/{event_id}`：单条触发事件详情或 404
- [X] T018 [P] 编写集成测试 `tests/test_scheduler_api_rules.py`：使用 FastAPI `TestClient`，mock `TriggerRuleStore`，覆盖所有端点的成功路径和错误路径（404/422/超上限），覆盖率 ≥ 85%

## Phase 7：前端调度管理页——基础结构

- [X] T019 [US6] 安装 pnpm 新依赖：在 `web/` 目录执行 `pnpm add @radix-ui/react-switch @radix-ui/react-collapsible`，更新 `package.json` 和 `pnpm-lock.yaml`
- [X] T020 [P][US6] 新增前端 Zod Schema 在 `web/src/types/api.schema.ts`（追加）：
  - `TriggerTypeSchema`（z.enum）
  - `ScheduleRuleSchema`（含所有字段和 `in_cooldown`/`last_triggered_at` 运行时字段）
  - `ScheduleRuleListSchema`（数组）
  - `TriggerEventSchema`（含展开详情字段）
  - `PaginatedTriggerEventsSchema`
- [X] T021 [P][US6] 在 `web/src/types/api.ts` 追加新类型导出：`TriggerType`、`ScheduleRule`、`ScheduleRuleList`、`TriggerEvent`、`PaginatedTriggerEvents`
- [X] T022 [US6] 新建 `web/src/pages/scheduler/hooks/use-rules.ts`：React Query hooks 封装：
  - `useRules()` — `useQuery`，`refetchInterval: 5000`，调用 `GET /api/scheduler/rules`
  - `useCreateRule()` — `useMutation`，调用 `POST /api/scheduler/rules`，成功后 invalidate 规则列表
  - `useUpdateRule()` — `useMutation`，调用 `PUT /api/scheduler/rules/{id}`
  - `useToggleRule()` — `useMutation`，调用 `PATCH /api/scheduler/rules/{id}/toggle`
  - `useDeleteRule()` — `useMutation`，调用 `DELETE /api/scheduler/rules/{id}`
- [X] T023 [P][US6] 新建 `web/src/pages/scheduler/hooks/use-trigger-history.ts`：
  - `useTriggerHistory(ruleId?, page, size)` — `useQuery`，调用 `GET /api/scheduler/triggers`

## Phase 8：前端调度管理页——组件实现

- [X] T024 [US6] 实现 `web/src/pages/scheduler/components/cron-editor.tsx`：受控组件，接受 `value/onChange` props；内置简单 cron 解析（覆盖 10 种常用模式）；实时展示中文自然语言说明和下次触发时间（UTC 格式）；输入无法识别时显示"自定义表达式"原文
- [X] T025 [US6] 实现 `web/src/pages/scheduler/components/template-selector.tsx`：展示 ≥4 张模板卡片（BTC 价格突破、ETH 异常波动、资金费率异常、BTC 连续阴线）；点击后通过 `onSelect(templateData)` 回调预填表单；使用 Radix UI Popover 或 Dialog 展示选择器
- [X] T026 [US6] 实现 `web/src/pages/scheduler/components/rule-form-dialog.tsx`：
  - Radix UI Dialog 弹窗
  - `react-hook-form` + Zod 验证，`resolver` 使用 `zodResolver`
  - 触发类型下拉选择（修改时动态切换参数字段组）
  - 四种类型参数表单区域：`price_threshold`（方向/价格）、`pct_change`（窗口/幅度）、`candle_pattern`（时间框架/根数/方向）、`funding_rate`（阈值）
  - 通用字段：名称、交易对、冷却分钟数
  - 提交调用 `useCreateRule` 或 `useUpdateRule`，成功后关闭弹窗
- [X] T027 [US6] 实现 `web/src/pages/scheduler/components/rule-table.tsx`：
  - 规则列表表格（列：名称、类型徽章、交易对、状态 Switch、冷却状态、最近触发、操作）
  - `@radix-ui/react-switch` 控制启用/禁用，调用 `useToggleRule()`
  - 操作列：编辑按钮（触发 `RuleFormDialog`）、删除按钮（二次确认 Dialog）
  - 空态显示：无规则时引导用户创建
- [X] T028 [US6] 实现 `web/src/pages/scheduler/components/trigger-history-table.tsx`：
  - 触发历史表格（列：触发时间、规则名称、类型、交易对、触发原因摘要、状态）
  - 点击行展开详情（使用 `@radix-ui/react-collapsible`）：展示完整触发原因、价格快照 JSON、关联分析记录 commit hash
  - 分页控制（上一页/下一页，显示总条数）
- [X] T029 [US6] 实现页面入口 `web/src/pages/scheduler/index.tsx`：
  - 页面标题 + 操作区（"创建规则"按钮 + "从模板创建"按钮）
  - Radix UI Tabs：`规则列表`（`RuleTable`） / `触发历史`（`TriggerHistoryTable`）
  - 错误边界和加载骨架（复用现有 `Skeleton` 组件）
  - 导出 default export 供 `App.tsx` lazy import

## Phase 9：前端路由与 i18n 集成

- [X] T030 [US6] 修改 `web/src/App.tsx`：
  - 新增 `const SchedulerPage = lazy(() => import('@/pages/scheduler'))`
  - 在 `Routes` 中新增 `<Route path="scheduler" element={<SchedulerPage />} />`
- [X] T031 [P][US6] 修改 `web/src/components/layout/sidebar.tsx`：
  - 在 `NavItem` 类型的 `labelKey` 联合类型中新增 `'nav.scheduler'`
  - 在 `NAV_ITEMS` 数组中插入 `{ to: '/scheduler', labelKey: 'nav.scheduler', icon: CalendarClock }`（导入 `CalendarClock` from lucide-react）
- [X] T032 [P][US6] 新建中文翻译文件 `web/src/locales/zh-CN/scheduler.json`（所有页面/组件文本键的中文值：规则类型名称、表格列名、状态文字、模板名称、表单标签、错误提示等）
- [X] T033 [P][US6] 新建英文翻译文件 `web/src/locales/en-US/scheduler.json`（对应英文翻译值）
- [X] T034 修改 `web/src/lib/i18n.ts`：
  - 在文件顶部新增导入语句：`import schedulerZh from '@/locales/zh-CN/scheduler.json'` 和 `import schedulerEn from '@/locales/en-US/scheduler.json'`
  - 在 `resources['zh-CN']` 对象中追加 `scheduler: schedulerZh`
  - 在 `resources['en-US']` 对象中追加 `scheduler: schedulerEn`
  - 在 `ns` 数组中追加 `'scheduler'`

## Phase 10：Agent 自我调度（P2）

- [X] T035 [US7] 修改 `src/cryptotrader/nodes/verdict.py`：
  - 新增 `_process_schedule_follow_up(state, verdict_data)` 函数：从 `verdict_data` 中提取 `schedule_follow_up` 字段（若存在）；检查 `state["metadata"].get("schedule_depth", 0) < 3`（防止递归超限）；调用 `TriggerRuleStore.create_rule()` 注册临时规则（`created_by="agent"`, `ttl_expires_at` 最大 72 小时）
  - 在 `run_verdict_node()` 返回前调用 `_process_schedule_follow_up()`；异常只记录 warning，不阻塞
  - 在 `pyproject.toml` per-file-ignores 为 `nodes/verdict.py` 确认 TID251 已豁免
- [X] T035a [P][US7] 修改 `src/cryptotrader/state.py` 的 `build_initial_state()`：在 `metadata` 字典中新增 `schedule_depth: int = 0` 字段支持（通过可选的 `schedule_depth` 参数传入，默认 0）；确保所有调用 `build_initial_state()` 的现有站点（scheduler、analyze route、backtest）无需改动即可保持向后兼容（参数有默认值）
- [X] T036 [P][US7] 在 `src/cryptotrader/scheduler.py` 的 `_run_pair()` 中：在构建 `initial_state` 时通过 `build_initial_state(schedule_depth=...)` 将 `schedule_depth` 从父级 metadata 传递（默认 0）；在 `_dispatch()` 触发的新分析中传入 `schedule_depth + 1`
- [X] T037 [P][US7] 编写单测覆盖 Agent 自我调度：`tests/test_agent_self_scheduling.py` 15 个测试覆盖 depth limit、TTL capping、field propagation、error handling

## Phase 11：TTL 规则清理任务

- [X] T038 修改 `src/cryptotrader/scheduler.py`：在 `start()` 中注册一个 `CronTrigger(minute=0)` 的 APScheduler 任务 `_cleanup_expired_rules()`，每小时调用 `TriggerRuleStore.cleanup_expired_rules()`（删除 `ttl_expires_at < now()` 且 `created_by="agent"` 的规则），并记录清理日志

## Phase 12：收尾与验证

- [X] T039 运行 `ruff check src/ tests/` 验证零 lint 错误（无 noqa 注释）；修复所有发现的问题
- [X] T040 运行完整测试套件：1197 passed, 6 failed (pre-existing), 1 skipped。新增模块覆盖率: conditions 100%, models 100%, store 96%, engine 69%, notifications 76%
- [ ] T041 手动验证 WebSocket 触发端到端流程：启动 `arena scheduler start`，配置一条测试规则，观察 Binance ticker 变化是否在 5 秒内触发（使用 mock 价格或沙盒环境）
- [ ] T042 验证 Telegram 通知：配置 `bot_token` 和 `chat_id`，触发 `price_trigger` 事件，确认 10 秒内收到中文格式消息；测试 `/status` 命令回复

---

## 依赖关系

```
T002, T002a 依赖 T001
T003 依赖 T002a（迁移方案确定后再写 migration 文件；若选 create_all() 则跳过 T003）
T004 依赖 T002, T002a
T005 依赖 T004
T006 独立（纯函数，无依赖）
T007 依赖 T006
T008 依赖 T004, T006（conditions）
T009 依赖 T008
T010, T011, T012 可并行，互不依赖
T013 依赖 T008（engine）, T010（config）
T014 依赖 T008（engine）, T010（config）, T004（store）
T015 依赖 T010（TelegramConfig）
T016 依赖 T015
T017 依赖 T004（store）, T010（config）, T015（notifier）
T018 依赖 T017
T019 独立（pnpm install）
T020, T021 可并行，依赖 spec.md 数据模型定义
T022, T023 依赖 T020, T021
T024, T025 独立（纯组件，无 hooks 依赖）
T026 依赖 T022（useCreateRule/useUpdateRule）, T025（模板）
T027 依赖 T022（useToggleRule/useDeleteRule）, T026（RuleFormDialog 作为弹窗触发）
T028 依赖 T023（useTriggerHistory）
T029 依赖 T027, T028, T024（CronEditor，用于定时规则展示）
T030 依赖 T029
T031 依赖 T030（确认路由可用后再加侧边栏）
T032, T033 可并行
T034 依赖 T032, T033
T035 依赖 T004（TriggerRuleStore）, T013（schedule_depth 传递）
T035a 依赖 T001（state.py 改动独立，但需在 T036 前完成）
T036 依赖 T035, T035a
T037 依赖 T035, T036
T038 依赖 T004（cleanup_expired_rules 方法）, T013
T039, T040 依赖所有前序任务
T041, T042 依赖 T040
```

---

## 任务工时估计

| 阶段 | 任务范围 | 估计工时 |
|------|----------|----------|
| Phase 1（基础设施） | T001–T005 | 0.5 天 |
| Phase 2（触发条件引擎） | T006–T009 | 1.5 天 |
| Phase 3（配置扩展） | T010–T012 | 0.5 天 |
| Phase 4（调度器集成） | T013–T014 | 0.5 天 |
| Phase 5（Telegram 通知） | T015–T016 | 1 天 |
| Phase 6（CRUD API） | T017–T018 | 1 天 |
| Phase 7（前端基础结构） | T019–T023 | 0.5 天 |
| Phase 8（前端组件） | T024–T029 | 2 天 |
| Phase 9（路由与 i18n） | T030–T034 | 0.5 天 |
| Phase 10（Agent 自调度） | T035–T037 | 1 天 |
| Phase 11（TTL 清理） | T038 | 0.25 天 |
| Phase 12（收尾验证） | T039–T042 | 0.75 天 |
| **合计** | | **~10 天** |

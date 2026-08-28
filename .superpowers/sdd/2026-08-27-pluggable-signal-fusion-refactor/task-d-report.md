# Task D：发布审计缺口修正报告

## 范围与交付边界

- 按 `task-d-release-audit-gaps.md` 修正四类缺口：周期编排 fail-closed、共享启动校验、风控目标审计、执行安全事实展示。
- 保持现有工业/审计台视觉体系，不更换主题、字体或交互范式，不新增动画。
- 未做旧 Journal/API schema 兼容层，当前合同直接升级。
- 未调用真实模型、真实交易所或 live 下单；未 push。

## RED 证据

### 后端：启动后普通异常缺少统一终态

命令：

```bash
rtk uv run pytest tests/test_trading_cycle.py tests/test_bootstrap.py tests/test_cycle_journal_store.py tests/test_api_decisions_detail.py --no-cov -q
```

结果：`9 failed, 45 passed`。失败覆盖 profile 边界错误、context/fusion/decision/exit 普通异常向外传播、approval 创建异常、default profile 缺组件仍可构建、Journal `error` 丢失、API 不接受 `cycle_failed`。

### 后端：持久化 active profile 未走共享启动校验

执行新增 shared helper、既有 API persisted-invalid、scheduler startup 三条聚焦测试，结果：`3 failed`。失败分别证明共享 helper 尚不存在、API 启动未拒绝数据库中的失效组件、scheduler 启动未 fail-fast。

### 后端：approval 清理异常遮蔽原始失败

命令：

```bash
rtk uv run pytest tests/test_trading_cycle.py::test_approval_cleanup_failure_cannot_hide_started_cycle_terminal --no-cov -q
```

结果：`1 failed`。`cancel_pending()` 的普通异常向外传播，导致原始异常被遮蔽且没有 `cycle_failed` 终态。

### 前端：cap 来源和执行安全事实被 schema/视图丢弃

命令：

```bash
rtk pnpm --dir web test -- src/pages/decisions/decisions-cycle-record.test.tsx
```

结果：`3 failed, 119 passed`。失败证明 Zod 丢弃 `cap_source`，中文风险/执行审计事实未渲染，英文 locale 对应文案未渲染。

## GREEN 实现

### 周期 fail-closed

- 新增单一领域终态 `cycle_failed`，Journal 与 API 增加单一 nullable `error`；不把普通编排异常伪装成 `component_failed`、`risk_rejected` 或 `execution_failed`。
- profile 读取、profile × Registry 校验、enabled components 与 requirements 合并作为 preflight，全部位于 `cycle_started` 之前。
- 一旦发布 `cycle_started`，context/fusion/decision/exit/approval 等普通异常统一写一次 `cycle_failed` Journal 和 terminal event，不进入执行；保留异常发生前已有的 context/signals/fused/plan 快照。
- `error` 采用 `TypeName: message`；approval 清理失败以附加说明保留，不遮蔽原始错误和终态。
- `ComponentRunError` 与 `asyncio.CancelledError` 保持既有专属语义。

### 共享 startup assembly

- `build_trading_cycle()` 在 Registry 装配后同步校验 config default profile，纯装配和 CLI/API/scheduler 默认配置一致 fail-fast。
- 新增共享 async `initialize_trading_cycle()`，按实际 Registry 校验持久化 active profile；API lifespan、standalone CLI、注入 cycle 的 scheduler 以及 scheduler lazy assembly 共用同一入口。
- helper 以 cycle 实例为边界幂等，保留 custom factory 仅 startup 加载与实例复用语义。
- 删除 API 自己实现的重复校验，但保留 API 对 persisted active profile 的 fail-fast 能力。

### 决策详情审计

- `CycleRiskResultSchema` 直接要求 `cap_source`；Decision list/detail 当前合同直接要求 nullable `error`。
- 风控区并列显示原始 target、risk-adjusted target 与 cap source。
- 非空执行结果显示 order intent、status、intent amount、filled amount、exchange order ID、`algo_id`、`retained_algo_ids`，以及 protection trigger reason/price/order ID/algo ID。
- 空 orders/protection facts 不展开附加审计块，保持紧凑。
- 新信息使用响应式卡片/网格、`min-w-0` 与断词规则，不引入移动端宽表；中英文 locale 对应一致。

## 变更文件

### 后端实现

- `src/cryptotrader/trading_cycle.py`
- `src/cryptotrader/bootstrap.py`
- `src/cryptotrader/scheduler.py`
- `src/cli/main.py`
- `src/api/main.py`
- `src/cryptotrader/decision/models.py`
- `src/cryptotrader/journal/models.py`
- `src/cryptotrader/journal/store.py`
- `src/api/routes/decisions.py`

### 后端测试

- `tests/test_trading_cycle.py`
- `tests/test_bootstrap.py`
- `tests/test_scheduler.py`
- `tests/test_cycle_journal_store.py`
- `tests/test_api_decisions_detail.py`

### 前端实现与合同

- `web/src/components/decision-detail/decision-detail-panel.tsx`
- `web/src/types/api.schema.ts`
- `web/src/pages/decisions/components/decisions-filter-bar.tsx`
- `web/src/locales/zh-CN/decisions.json`
- `web/src/locales/en-US/decisions.json`

### 前端测试

- `web/src/pages/decisions/decisions-cycle-record.test.tsx`
- `web/tests/unit/schema-contract.test.ts`

## 最终验证

- 后端聚焦/相邻回归：先 `106 passed`；最终修正并格式化后，相关五个文件 `77 passed, 1 warning`。
- 后端全量：`1841 passed, 36 warnings`，coverage `74.60%`，pytest exit `0`。
- 前端全量：`20 passed` test files，`123 passed` tests。
- `rtk pnpm --dir web typecheck`：通过。
- `rtk pnpm --dir web lint`：通过。
- `rtk pnpm --dir web build`：通过。
- `rtk uv run ruff check src tests`：通过。
- `rtk uv run ruff format --check src tests`：`353 files already formatted`。
- `rtk git diff --check` 与 staged diff check：通过。
- architecture `rg` 扫描 legacy graph/verdict/steering 标识：零命中（`rg` exit `1` 为无匹配）。
- shared startup `rg` 确认 API、CLI、scheduler 均调用 `initialize_trading_cycle()`。

## 假设与裁定

- 领域模型只新增一个通用 `cycle_failed`，不扩展阶段状态枚举。
- profile load/validate/requirements 属于运行前 preflight；未发布 `cycle_started` 时不需要伪造终态。
- default profile 与 persisted active profile 是两层独立 startup 不变量，必须都按 actual Registry 校验。
- 现有 Redis durable ambiguous-ack/cancellation 议题属于原分支已记录的跨任务残余，本 Task 不扩范围处理。

## 残余

- Task D 要求范围内无已知功能残余。
- 全量后端退出码为 `0`；测试进程关闭后 OTEL exporter 尝试连接本机未运行的 `localhost:4317`，产生一条外部遥测日志，不影响测试结果或产品行为。
- 未执行真实 provider/model/exchange/live 验证，符合本 Task 的离线 release-audit 范围。

## 实现提交

`a927b54ae94128a45f5c6896a97a6963eadb3e7a` (`fix: close release audit gaps`)

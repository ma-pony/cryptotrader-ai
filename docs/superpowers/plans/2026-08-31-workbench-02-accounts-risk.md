# 第二批：账户与风险 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 网页可核对全部账户资产与成交，在整池风险约束下执行，并可安全停止和退出指定持仓。

**Architecture:** 平台提供账户级标准事实，独立同步服务持久化账本。交易、审批和人工退出共享资金池串行边界，风险计算与页面读取同一份持久化状态。

**Tech Stack:** 现有交易适配器、SQLAlchemy async、Decimal、pytest、React／TanStack Query；不新增消息队列或会计系统。

**Spec:** [设计规格](/Users/rccpony/Projects/cryptotrader-ai/docs/superpowers/specs/2026-08-31-trading-workbench-design.md)，依赖[第一批](/Users/rccpony/Projects/cryptotrader-ai/docs/superpowers/plans/2026-08-31-workbench-01-extensions-decisions.md) F1–F5。

## Global Constraints

- 模拟账户和真实账户可以同时存在，不设置全局互斥的“模拟／实盘模式”。真实下单另有明确授权。
- 已实现收益从成交或平台明确提供的结算收益计算；本期本地计算采用移动平均持仓成本。
- 不可获得或不可核对的数据返回空值和原因，不能填零，也不能把权益差额直接当作交易收益。
- 退出完成后资金池保持停用，只有用户明确恢复后策略才能重新开仓。
- 直接替换旧流程，不长期维护双套页面、双套业务规则或兼容路由。
- 本批仅使用假平台／本地测试账户；运行库、真实账户、外部模拟盘均不在自动验证范围。

---

## 文件责任图

| 文件 | 操作及责任 |
| --- | --- |
| `src/cryptotrader/accounts/__init__.py`、`models.py`、`store.py`、`sync.py`、`income.py`、`operations.py` | Create；标准事实、持久化、同步、收益、独立退出 |
| `src/cryptotrader/venues/protocol.py`、`models.py`、`paper.py`、`okx.py`、`bybit.py` | Modify；完整账户读取、品种及账本能力 |
| `src/cryptotrader/configuration/catalog.py`、`fields.py` | Modify；账户读取／退出能力声明 |
| `src/cryptotrader/portfolio/models.py`、`aggregator.py` | Modify；执行快照由全账户事实投影，不丢其他品种 |
| `src/cryptotrader/risk/book_state.py` | Create；资金池风险状态和峰值持久化；不覆盖现有 Redis `risk/state.py` |
| `src/cryptotrader/risk/models.py`、`gate.py`、`execution_ownership.py`、`cycle_lock.py` | Modify；整池预算和按池执行租约 |
| `src/cryptotrader/trading_cycle.py`、`execution/service.py`、`runtime.py`、`runtime_config/models.py`、`repository.py` | Modify；同步、风控、审批、停用与配置约束接线 |
| `src/api/routes/accounts.py`、`account_operations.py` | Create；独立读账户、刷新与退出接口 |
| `src/api/routes/portfolio_books.py`、`hitl.py`、`venues.py`、`config.py`、`main.py` | Modify；新读模型、审批复核、删除／分配保护和生命周期 |
| `src/cryptotrader/migrations/workbench.py` | Modify；新表及已有配置的归属快照 |
| `web/src/hooks/use-accounts.ts`、`use-account-operations.ts` | Create；账户查询与退出状态 |
| `web/src/hooks/use-portfolio-books.ts`、`use-hitl-approvals.ts` | Modify；同源风险与审批 |
| `web/src/pages/accounts/index.tsx`、`connection-detail.tsx`、`book-detail.tsx`、`exit-dialog.tsx`、`book-form.tsx` | Create；聚合账户、四页签、资金池与退出；迁入已有 book-form 的有效逻辑 |
| `web/src/types/api.schema.ts`、`api.ts`、`web/src/locales/zh-CN/configuration.json` | Modify；有币种／未知原因的 DTO 与中文业务状态 |

## B1：全账户协议及平台适配

**Files:** accounts/models、venues 五文件、catalog／fields、portfolio/models。Test：Create `tests/test_account_read_contract.py`；Modify `tests/test_okx_venue_adapter.py`、`test_bybit_venue_adapter.py`、`test_paper_venue_adapter.py`、`tests/factories/workbench_extensions.py`。

**Interfaces:** Consumes F1 的环境、凭据和平台注册。Produces 下列不可变模型与扩展协议：

```python
class VenueSession(Protocol):
    async def list_instruments(self) -> tuple[Instrument, ...]: ...
    async def fetch_account(self) -> AccountSnapshot: ...
    async def fetch_fills(self, cursor: str | None) -> FillPage: ...
    async def fetch_funding(self, cursor: str | None) -> FundingPage: ...
```

此处 `...` 是 Python Protocol 方法声明，不代表缺省实现；内置三个适配器和测试平台必须在本任务实现这些方法。现有 quote／order／protection 执行方法保留。

- `Money(amount: Decimal | None, currency: str, unavailable_reason: str | None)`；未知金额必须解释原因。
- `Instrument(venue_symbol, pair: Pair | None, market_type, tradable, reason)`；无法规范化的外部品种仍展示 venue_symbol，不能操作。
- `AccountPosition(instrument, signed_amount, available_amount, signed_notional: Money, entry_price, unrealized_pnl: Money)`；支持一个账户多个品种，禁止 dict 只按 pair 合并不同账户。
- `AccountOrder(connection_id, venue_order_id, instrument, side, order_type, amount, filled_amount, average_price, status, reduce_only, protection, client_order_id, observed_at)`；普通与保护单分组，未知品种也保留。
- `AccountSnapshot(connection_id, observed_at, capital_scope, equity: Money, balances, positions, orders, used_margin: Money, available_margin: Money, completeness)`；completeness 列出读取不到的事实。
- `Fill(connection_id, venue_fill_id, venue_order_id, instrument, side, amount, price, occurred_at, fee: Money, realized_pnl: Money, source)`；source 区分平台提供／本地计算，决策与历史资金池归属在 B2 写账时补充。
- `FundingEntry(connection_id, venue_entry_id, instrument, amount: Money, occurred_at)`；`FillPage(items, next_cursor, complete)`、`FundingPage(items, next_cursor, complete)`，complete 表示该次范围已读完。

- [ ] 先写协议测试，假平台同时返回 BTC／ETH 持仓、普通／保护单、手续费和资金费；另有无法映射的品种仍可见。读取必须不调用 place_order／cancel_order。

```python
snapshot = await session.fetch_account()
assert {p.instrument.venue_symbol for p in snapshot.positions} == {"BTCUSDT", "ETHUSDT", "UNKNOWN"}
assert snapshot.positions[-1].instrument.tradable is False
assert snapshot.used_margin.amount is None
assert snapshot.used_margin.unavailable_reason
assert session.write_calls == []
```

- [ ] 红测：`rtk proxy env -u DATABASE_URL -u CONFIG_MASTER_KEY .venv/bin/python -m pytest --no-cov tests/test_account_read_contract.py -q`。
- [ ] 为 OKX／Bybit 使用已安装 SDK/CCXT 的账户级读取和分页成交能力，按固定脱敏返回夹具编写 mapper。资金费或已实现收益不可得时明确未知；不能回退成名义金额／权益差。能力目录声明哪些读项及退出操作可用，前端只消费声明。
- [ ] Paper 实现同一协议，从实际成交和余额读取。账户身份绑定 connection_id，生命周期重建会话不得按 initial_equity 再充值；首次资金初始化仅发生一次，B2 持久化其账户状态。保留成交平台 ID 和订单 client ID 以供关联。

```python
def external_order_source(client_order_id: str | None, owned_ids: set[str]) -> str:
    return "strategy" if client_order_id in owned_ids else "external"
```

该纯函数只判定来源，不推断最近决策；账本关联由 B2 以实际订单 ID 完成。

- [ ] 使用内置平台脱敏夹具和 sample_venue 跑协议契约测试；新增平台同样返回完整标准模型，不增加前端平台判断。SDK 字段不确定时查官方文档再实现，不用未经验证的响应字段凑数。
- [ ] 重跑三个适配器测试及 paper protection/concurrency 现有测试。记录平台明确不支持的读取项；不把本地 fixture 通过称为官方模拟盘验证。

## B2：账户同步、成交收益和详情页面

**Files:** accounts/store／sync／income、runtime、main、runtime_config/models／repository、accounts API、portfolio/aggregator、迁移、账户页面／hooks／schemas。Test：Create `tests/test_account_sync.py`、`test_account_income.py`、`test_accounts_api.py`、`web/src/pages/accounts/connection-detail.test.tsx`。

**Interfaces:** Consumes B1 账户协议、F1 平台目录。Produces `AccountSyncService.sync(connection_id: str) -> AccountSnapshot`、`AccountStore.latest(connection_id)`、`AccountStore.ingest(snapshot, fills, funding)`、`IncomeService.summary(connection_id, start, end) -> IncomeSummary`。

`IncomeSummary` 为 realized_gross、fees、funding、unrealized、net_trading 的按币种 Money 项，以及 completeness 和口径说明。net_trading 只合计同币种完整已实现数据，未实现盈亏单列。

- [ ] 写临时库集成测试：两个连接同一交易对不覆盖；重复同步同成交 ID 只留一行；停用连接重访仍能读；失败保留最近成功快照和失败时间。移动平均 fixture：买2@100，买1@130，卖1@140，已实现毛利30；手续费和资金费另列。

```python
await sync.sync("sim-a")
await sync.sync("sim-a")
assert await store.fill_count("sim-a") == 3
summary = await income.summary("sim-a", start, end)
assert summary.realized_gross[0].amount == Decimal("30")
assert summary.fees[0].currency == "USDT"
assert summary.unrealized[0].unavailable_reason == "缺少当前估值"
```

`AccountStore.fill_count(connection_id)` 为本任务提供的计数查询；测试用固定 UTC start/end 和可读性明确的 fake session，不依赖环境变量内的平台密钥。

- [ ] 红测：`rtk proxy env -u DATABASE_URL -u CONFIG_MASTER_KEY .venv/bin/python -m pytest --no-cov tests/test_account_sync.py tests/test_account_income.py tests/test_accounts_api.py -q`。
- [ ] 建快照、订单、fills、funding、sync_cursor 表。fills 唯一键 `(connection_id, venue_fill_id)`，funding 独立唯一键；游标与对应批次入账同事务提交。订单通过真实 order/client ID 关联决策和人工操作，外部单不强行关联。
- [ ] 保存账户分配有效期 `book_memberships(connection_id, book_id, valid_from, valid_to)`；按成交发生时间归属，已有订单优先使用生成时的归属。配置保存关闭旧区间、开启新区间，不用同步时的当前 book 覆盖过去。无法证明的旧归属为未知。

```python
new_average = (old_quantity * old_average + added_quantity * fill_price) / (old_quantity + added_quantity)
```

该式只用于同向增仓；减仓按减仓前平均成本结转已实现盈亏，反向成交拆平旧仓和建新仓。平台若提供明确结算收益则采用该值，不再累加本地结果；期初成本缺失保持未知。收益计算禁止使用权益差替代。

- [ ] 同步 owner 独立于自动交易 owner，启动和 DB 可配 `accounts.sync_interval_seconds`（默认60秒）定期读取已配置账户，包括停用连接；凭据缺失保留状态不空转。Paper 的余额、成本和订单状态重启可恢复。刷新账户接口只触发同步，不改变 enabled、授权或审批。
- [ ] 接入账户详情四页签：概览、持仓与订单、成交与收益、连接配置。显示 last_success_at／失败原因、未知收益和原币种费用；模拟／真实分开汇总。列表保留停用与未分配账户，资金池读取完整账户结果。无成交显示“运行模拟交易后可在这里核对成交”，不填假收益。
- [ ] 重跑后端新测试、现有 portfolio aggregator 测试及前端详情测试。重新创建服务对象后从同一临时库读取数据，验证样本数、时间、订单来源、账户归属未改变。

## B3：整池风险、持久化峰值与审批复核

**Files:** risk/book_state／models／gate、execution_ownership／cycle_lock、trading_cycle／runtime、portfolio/aggregator、hitl／portfolio_books／runtime_status API、账户 book-detail、审批 hook。Test：Create `tests/test_book_risk_integration.py`、`test_book_execution_ownership.py`；Modify `tests/test_book_risk_gate.py`、`test_multi_book_cycle.py`、`test_portfolio_books_api.py`；Create `web/src/pages/accounts/book-detail.test.tsx`。

**Interfaces:** Consumes B2 `AccountSyncService`、B1 instruments、F5 run scope。Produces `BookRiskStateStore.update(book_id, snapshots) -> BookRiskState`、`ExecutionOwnership.book(book_id: str)` async context manager；`BookRiskRequest` 增加 pair、全池 state，移除临时 peak 参数来源。

`BookRiskState` 保存 book_id、capital_scope、valuation_currency、observed_at、equity、peak_equity、positions_by_instrument、pending_increase_notional、gross_notional、net_notional、used_margin、available_margin、completeness。当前执行模型所需单对 ConnectionPortfolioSnapshot 从该状态投影；总量检查不能用投影替代全池状态。

- [ ] 写穿过 runtime→cycle→risk→journal→API 的集成测试，而非只测 gate：峰值100、本轮80、最大回撤10%；另一例权益100、其他品种70、总上限80、本次目标40。加同池两个并发品种和一个外池测试。

```python
assert saved_state.peak_equity == Decimal("100")
assert saved_state.equity == Decimal("80")
assert drawdown_result.capped_target_exposure == Decimal("0")
assert gross_result.capped_target_exposure <= Decimal("0.10")
assert other_book_was_allowed_to_progress
```

- [ ] 红测：`rtk proxy env -u DATABASE_URL -u CONFIG_MASTER_KEY .venv/bin/python -m pytest --no-cov tests/test_book_risk_integration.py tests/test_book_execution_ownership.py -q`。
- [ ] 按实际账户同币种快照初始化并更新 peak；普通保存不能重置。风险计算替换目标品种当前持仓、保留其他品种，再加未成交增仓占用；减仓单成交前不释放预算。同币种估值缺失或账户不完整时不扩大风险，已知可安全减仓保留。

```python
other_gross = sum(abs(value) for key, value in positions.items() if key != requested_pair)
gross_room = max(Decimal("0"), equity * limits.max_gross_exposure - other_gross - pending_increase)
capped_notional = min(abs(requested_notional), gross_room)
```

该式是增仓的总量上限片段；实现还需保留净敞口、账户集中度、实际保证金和回撤检查，减仓分支优先判定，不能把降低已有风险错误地拒绝。原始目标、调整目标和 cap_source 均写 journal。

- [ ] 新增 ExecutionOwnership，复用已有 Redis 租约／取消安全 helper，把全局 pair 锁改为 book 锁；嵌入原配置应用屏障，禁止在持有 book 锁时反向获取应用写屏障。每池刷新→风险→下单持锁，下一品种看最新订单占用；不同池独立。批准和人工退出同样必须走此边界。
- [ ] 交易范围验证所有成员账户对所选品种的实际能力，任一不支持只阻止该池，不改分配权重。批准时锁内刷新完整账户，用冻结计划对比最新配置、仓位、待成交占用、保证金及保护要求；不再可执行则 invalidated，绝不静默改单量。有效审批仍执行原数量。
- [ ] 页面展示同一风险状态与时间、当前／待成交占用、原始／调整目标和原因；回撤压到0写“风险限制：目标持仓降至零”。移除页面旧的未接线熔断／CVaR 结论。重跑风险、并发、审批和交易主链回归，证明停用／暂停不会取消已提交订单。

## B4：人工退出、停用和安全移除

**Files:** accounts/operations／store、account_operations API、venues／config API、runtime_config/repository、execution/service、exit-dialog、账户详情／book-form。Test：Create `tests/test_account_operations.py`、`test_account_removal.py`、`web/src/pages/accounts/exit-dialog.test.tsx`；Modify 现有资金池配置测试。

**Interfaces:** Consumes B3 book lock、B2 sync、F5配置版本和真实授权。Produces `AccountOperationService.prepare(connection_id, pair, kind, expected_revision, confirm_stop) -> str`；`execute(operation_id: str, plan_version: int) -> str`；`get(operation_id) -> AccountOperationOut`。

本任务在 accounts/store.py 定义 `AccountOperationStore`，提供 create/get/update/invalidated 转移及 `invalidate(operation_id: str, reason: str) -> AccountOperationOut`；它使用同一数据库会话基础，不复制 AccountStore 的快照／成交表。

`ExitPlan` 冻结 operation_id、version、connection_id、book_id、capital_scope、pair、kind、stopped_scope、ordinary_order_ids、position_amount、close_amount、protection_ids、snapshot_time；结果另存实际撤单、成交、剩余持仓和失败原因。操作状态遵循总计划，终态可在重启后查询。

- [ ] 写 fake venue 操作序列测试：先停所属池，其他池保持 enabled；准备后只给用户看计划，不下单。确认执行时普通订单取消在平仓前，保护取消在零仓确认后。模型和策略审批 spy 调用数始终0。

```python
assert calls.index("cancel_ordinary") < calls.index("close_reduce_only")
assert calls.index("confirm_flat") < calls.index("cancel_protection")
assert selected_book.enabled is False
assert unrelated_book.enabled is True
assert model_calls == []
assert approval_calls == []
```

- [ ] 红测：`rtk proxy env -u DATABASE_URL -u CONFIG_MASTER_KEY .venv/bin/python -m pytest --no-cov tests/test_account_operations.py tests/test_account_removal.py -q`。
- [ ] prepare 确认停用范围后通过 CAS 保存 book.enabled=false，无归属则 connection.enabled=false；使用原配置屏障等在途工作结束，取得 book 锁后同步生成 ExitPlan。HTTP 返回 preparing，前端查询进度，不让长请求假装卡死。所有新策略入口读最新 enabled 状态。
- [ ] execute 先核对真实授权、plan_version 和最新状态。先刷新比较冻结数量／订单，变化则 invalidated；确认撤完普通单后再次读取，若数量变化返回新版本 awaiting_confirmation，不自动使用新增数量。衍生品仅 reduce_only，现货不超过 available_amount。沿用 ExecutionService 的订单确认与保护处理，不复制补偿系统。

```python
if fresh.position_amount != plan.position_amount:
    return await self.store.invalidate(operation_id, reason="持仓已变化，请重新确认")
if plan.capital_scope == "live" and not live_order_execution_enabled:
    raise PermissionError("尚未授权真实账户交易")
```

`AccountOperationStore.invalidate(operation_id, reason)` 更新持久化操作，不撤单、不推理；execute 中所有退出路径写实际状态。平仓只有同步证明零仓才 completed，失败保留保护与剩余仓位供查看。资金池不自动恢复。

- [ ] 停用连接先展示所属启用池并停用该池，不能自动分摊权重。配置总 PUT、连接删除及移出资金池均调用同一个 `assert_account_removable(connection_id)`：停用、最新只读同步确认无仓／无挂单；同步失败拒绝移除。历史及已删除连接的归档身份继续保留，不级联删账本。
- [ ] UI 两阶段：确认停用范围→展示实际退出计划→确认执行；说明保护保留与其他池范围，失败给剩余事实／重试读取入口，不误报成功。纯停用不写任何订单。测试 spot、合约、过期计划、尚有挂单的删除、失败保护保留，以及退出后全局手动 run 跳过该池。
- [ ] 运行全部本批测试、旧执行保护回归及前端退出测试；人工查看测试界面中文、焦点、窄屏确认内容。只向用户报告 fake／Paper 测试结果，不自动在已配置的外部账户试平仓。

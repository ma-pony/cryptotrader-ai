# 第三批：研究与告警 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 能用保存的事实复盘信号和回测，并在审批、执行及上游失败时收到可追踪告警。

**Architecture:** 组件评估引用原始信号，回测使用隔离账户及历史时钟，两者均独立保存结果。业务事件写站内告警，再通过现有 Webhook 投递；通知失败不回滚业务。

**Tech Stack:** 现有历史行情 provider、Paper venue、SQLAlchemy async、Notifier、React 图表／表格与查询缓存。

**Spec:** [设计规格](/Users/rccpony/Projects/cryptotrader-ai/docs/superpowers/specs/2026-08-31-trading-workbench-design.md)，依赖 F3–F5、B1–B3；R4 的人工操作关联在 B4 完成后联调。

## Global Constraints

- 信任权重仍由用户调整，不做自动调权、参数寻优或复杂绩效归因。
- 信号产生时冻结起始行情时间、参考价格、方向及评估截止时间；到期后用指定市场数据源的对应已收盘行情评估，禁止重新生成原信号。
- 回测样本与实时样本不混算。
- 告警记录与外部投递结果分别持久化。
- 本批不调用外部付费模型／真实或官方模拟账户；所有行为测试使用可控时钟、行情夹具和假 Webhook。

---

## 文件责任图

| 文件 | 操作及责任 |
| --- | --- |
| `src/cryptotrader/signals/evaluation.py`、`evaluation_store.py` | Create；到期评估和统计持久化 |
| `src/cryptotrader/signals/presentation.py`、`runtime.py`、`src/api/main.py` | Modify；评估数据块及独立后台 owner |
| `src/api/routes/components.py` | Create；组件历史与评估查询 |
| `src/cryptotrader/backtest/engine.py`、`result.py`、`historical_data.py`、`session.py` | Modify；真实模拟成交、历史时钟、删除文件式运行结果存取 |
| `src/cryptotrader/backtest/store.py`、`comparison.py` | Create；数据库运行记录及条件比较 |
| `src/cryptotrader/venues/paper.py`、`models.py` | Modify；可配置成本和历史 bar 的保护触发 |
| `src/api/routes/backtest.py`、`src/cli/main.py`、`scripts/run_backtest.py`、`scripts/kronos_backtest_ab.py` | Modify；全部调用新的结果模型／数据库存取，不留失效调用方式 |
| `src/cryptotrader/alerts/__init__.py`、`models.py`、`store.py`、`service.py` | Create；告警、业务状态关联、投递任务 |
| `src/cryptotrader/notifications.py`、`cycle_events.py`、`scheduler.py`、`runtime_config/models.py` | Modify；扩展事件、复用 Webhook 和配置 |
| `src/api/routes/alerts.py` | Create；事项、已读、投递状态和重试 |
| `src/cryptotrader/migrations/workbench.py` | Modify；评估／回测／告警表和旧回测显式导入 |
| `web/src/hooks/use-component-evaluations.ts`、`use-alerts.ts` | Create；查询、筛选及已读／重试 |
| `web/src/hooks/use-backtest.ts` | Modify；数据库运行 API |
| `web/src/pages/engine/component-evaluation.tsx` | Create；组件效果页签 |
| `web/src/pages/research/index.tsx`、`backtest-detail.tsx`、`backtest-compare.tsx` | Create；迁入原市场／回测有效视图，新增持久化历史与比较 |
| `web/src/pages/settings/notifications.tsx` | Create；规则和投递状态 |
| `web/src/components/alerts/attention-list.tsx` | Create；工作台与账户页复用事项列表 |
| `web/src/types/api.schema.ts`、`api.ts`、`web/src/locales/zh-CN/backtest.json`、`configuration.json` | Modify；严格 DTO、未知／样本和中文口径 |

## R1：冻结周期的组件效果复盘

**Files:** evaluation／evaluation_store／presentation、components API、runtime／main、component-evaluation、hook／schemas。Test：Create `tests/test_signal_evaluation.py`、`tests/test_component_evaluations_api.py`、`web/src/pages/engine/component-evaluation.test.tsx`。

**Interfaces:** Consumes F3 `EvaluationReference`、F4 已保存 ComponentSignal 和 F1 行情源。Produces `EvaluationService.evaluate_due(now: datetime) -> int`（本次完成数量）、`EvaluationStore.summary(filters) -> EvaluationSummary`。

`EvaluationRecord` 键为 `(decision_id, component_id)`，保存 pending/evaluated/missing_market/not_directional/skipped/failed、原 reference、到期实际价格／时间、hit、return_ratio 和预测对照块。`EvaluationSummary` 按 pair/mode/config_revision/interval 分组，含 total、pending、matured_directional、hits、neutral、skipped、failed、missing_market、hit_rate 可空；不能跨周期求一个混合比率。

- [ ] 写有方向上涨命中、下跌未命中、价格持平、未到期、无方向、缺行情、失败／跳过的时间夹具。参考价100、方向long、到期110命中；到期100不命中；原预测 hash 不变。

```python
assert evaluate_direction("long", Decimal("100"), Decimal("110")) is True
assert evaluate_direction("long", Decimal("100"), Decimal("100")) is False
assert direction_hit_rate(hits=0, matured_directional=0) is None
assert original_prediction_after == original_prediction_before
```

本任务实现 `evaluate_direction(direction, reference_price, actual_price) -> bool`（仅接收 long/short）和 `direction_hit_rate(hits: int, matured_directional: int) -> float | None`。

- [ ] 红测：`rtk proxy env -u DATABASE_URL -u CONFIG_MASTER_KEY .venv/bin/python -m pytest --no-cov tests/test_signal_evaluation.py tests/test_component_evaluations_api.py -q`。
- [ ] 仅取已保存 reference 对应行情源的截止已收盘数据；未收盘保持 pending。方向neutral进入 not_directional，缺数据进入 missing_market、后续同步可再评估；不能把其他时间最近价冒充到期价。后台 owner 与交易暂停独立，重复调度按唯一键更新，不重跑组件。

```python
def direction_hit_rate(hits: int, matured_directional: int) -> float | None:
    return hits / matured_directional if matured_directional else None

def evaluate_direction(direction: str, reference_price: Decimal, actual_price: Decimal) -> bool:
    change = actual_price - reference_price
    return change > 0 if direction == "long" else change < 0
```

- [ ] Kronos 曲线按已保存预测时间与后续实际点匹配，另存差值及有完整匹配样本的误差指标；不修改 blocks 内原序列。输入费用只有原用量与明确模型单价齐全才估计，不从当前模型价回填过去“实际费用”。
- [ ] 组件详情增加效果页签及筛选，明确“方向命中率”分母和总样本；0有效样本显示“尚无可评估样本”。pending／neutral／missing／failed 单列，实时与回测分组。加权贡献不能换名成收益。
- [ ] 跑目标后端、前端测试；重建 service 后查询结果一致，GET evaluations 不增加模型调用或改原 journal。用缺行情→补齐行情夹具验证结果可更新且原始预测稳定。

## R2：真实模拟成交、成本与保护触发

**Files:** backtest/engine／result／historical_data、venues/paper／models、accounts账本接口；Modify `tests/test_backtest.py`、`test_live_backtest_decision_parity.py`、`test_paper_exchange_protection.py`；Create `tests/test_backtest_accounting.py`、`tests/test_backtest_asof.py`。

**Interfaces:** Consumes F4 `SignalAnalysisService`、B1 fills／funding、B3整池风险。Produces `BacktestCostModel(fee_rate, slippage_bps, funding_enabled)`、`EquityPoint(time: datetime, equity: Decimal)` 和完整 `BacktestResult`。

BacktestCostModel 定义在 venues/models.py，Paper 直接消费，避免平台层反向依赖 backtest engine；EquityPoint 和 BacktestResult 定义在 backtest/result.py。

结果含 equity_curve、fills、closed_trades、fees、funding、cost_assumptions、unmodeled_costs、data_coverage、decision_ids；win_rate 为已完成平仓回合中的盈利比例，无样本null，fill_count 与 closed_trade_count 分别显示。平仓回合以仓位回到零或反向开新仓为界，按实际成交和分摊费用汇总。

- [ ] 写确定行情和信号序列：买1@100、卖1@110，手续费率0.001、滑点0，初始1000；毛利10、费用0.21、最终1009.79，2笔成交、1次完成平仓。再测不交易、持仓未平和触发保护。

```python
assert result.fill_count == 2
assert result.closed_trade_count == 1
assert result.fees == Decimal("0.21")
assert result.equity_curve[-1].equity == Decimal("1009.79")
assert result.equity_curve[0].time == historical_start
assert result.win_rate == 1.0
```

- [ ] 红测：`rtk proxy env -u DATABASE_URL -u CONFIG_MASTER_KEY .venv/bin/python -m pytest --no-cov tests/test_backtest_accounting.py tests/test_backtest_asof.py -q`。
- [ ] 将 engine 传给 `_compute_result` 的空 trades 替换为 Paper 实际账本，权益在每个历史采样点保存时间。Paper 每次成交按成交价和名义额扣费，买／卖滑点方向相反；资金费仅在已取得的历史结算点计入。来源不可用记入 unmodeled_costs，不伪造0成本。

```python
sign = Decimal("1") if side == "buy" else Decimal("-1")
execution_price = reference_price * (Decimal("1") + sign * slippage_bps / Decimal("10000"))
fee = abs(amount * execution_price) * fee_rate
```

- [ ] 明确本期 bar 模拟规则：先处理上一时点已存在的保护单，再用本根已收盘行情生成信号，市价单按本根收盘价及滑点成交；新保护单从下一根生效。一个 bar 同时触及止盈止损时按止损先触发；跳空越过止损时按开盘更不利价成交。规则写入成本／执行假设，不能据此宣称逐笔成交仿真。
- [ ] 所有组件输入限制在 as_of 之前，历史新闻／资金费缺失明确披露，禁止代用当前新闻。完整复用信号、融合和风险服务；只注册本次临时 Paper 连接，原配置真实授权／真实连接不进入构造。预训练模型可能含历史之后知识写入报告限制，不承诺严格样本外重演。
- [ ] 跑新测试和既有 backtest／Paper保护回归，检查手续费导致权益变化、保护真的减仓、当前时间不泄漏到历史曲线。成本模型使用当前库依赖，不接入参数优化器。

## R3：回测持久化、历史详情和比较

**Files:** backtest/store／comparison／session、API backtest、CLI与两个脚本、迁移、研究页面／hook／schemas。Test：Modify `tests/test_api_backtest_run.py`、`test_api_backtest_status.py`、`test_api_backtest_sessions.py`、`test_cli_backtest.py`；Create `tests/test_backtest_store.py`、`web/src/pages/research/backtest-detail.test.tsx`、`backtest-compare.test.tsx`。

**Interfaces:** Consumes R2 BacktestResult、F4安全快照及任务管理。Produces `BacktestStore.create(params, config_snapshot) -> str`、`update(run_id, status, progress, result=None, error=None)`、`get(run_id)`、`list(limit, offset)`；`compare_runs(left, right) -> BacktestComparison`。

`BacktestComparison` 含 comparable、condition_differences、configuration_differences 和两侧结果。固定比较条件为 pair/start/end/interval/initial_equity/fee_rate/slippage_bps/funding_assumption；模型／提示词身份、数据覆盖及缺失上下文也显示，不能隐藏这些实验差异。

- [ ] 先测试未命名的运行也持久化；保存后销毁 service 和进程态字典，重建 store 仍有全部曲线、成交和安全配置；running 记录在启动恢复时标 interrupted。

```python
restored = await fresh_store.get(run_id)
assert restored.result.equity_curve == result.equity_curve
assert restored.result.fills == result.fills
assert compare_runs(left, different_fee_run).comparable is False
assert "fee_rate" in compare_runs(left, different_fee_run).condition_differences
```

- [ ] 红测：`rtk proxy env -u DATABASE_URL -u CONFIG_MASTER_KEY .venv/bin/python -m pytest --no-cov tests/test_backtest_store.py tests/test_api_backtest_run.py tests/test_api_backtest_status.py -q`。
- [ ] API start 在排入任务前先创建 run；每次进度、终态写数据库。内存仅保留执行中的任务句柄，不作为结果来源。删除 `_RUNS` 结果字典和 session_name 才保存的分支；启动扫描 queued/running 标 interrupted，本期不断点续跑。

```python
run_id = await store.create(params, safe_snapshot)
task_manager.start_backtest(run_id, params)
return BacktestRunResponse(run_id=run_id, status="queued")
```

`BackgroundTaskManager.start_backtest(run_id, params)` 在本任务接到现有 task 调度与 R2 engine；其所有结果交给 BacktestStore，不能重建另一个结果缓存。

- [ ] 新 DTO 直接传 EquityPoint 的历史 time；删除请求时统一 datetime.now 的曲线转换。记录组件参数、信任权重、风险、评估周期、请求／实际模型 ID、提示词内容哈希与可安全保存的版本，以及输入数据覆盖；未知实际模型 ID 单独标明。
- [ ] 旧文件式 session 读取移至一次性迁移，CLI／脚本同步使用 DB store，不接受已经删除的保存目录运行参数。导入的旧记录若当时未保存曲线，显示缺失原因且保留源文件，禁止重新生成历史。
- [ ] 研究页面提供历史列表、详情、复用快照和两次比较；启动前展示数据缺项、成本假设和模型费用提示。比较条件不同先列差异，不排名；无平仓样本显示暂无样本。表单失败保留输入，关闭／刷新不丢任务结果。
- [ ] 跑 API／store／CLI／前端目标测试，加入无名称、无交易、取消、中断和重新进入页面的状态检查；确认查看历史不发模型请求。后台任务与前端均只链接 `/research` 体系，旧页面删除留 U2。

## R4：业务告警和独立 Webhook 投递

**Files:** alerts 目录、notifications／cycle_events／scheduler／runtime_config/models、runtime／main、alerts API、attention-list／notifications 页面／hook。Test：Create `tests/test_alert_lifecycle.py`、`test_alert_delivery.py`、`web/src/pages/settings/notifications.test.tsx`；Modify `tests/test_notifications.py`、`test_runtime_notifications.py`。

**Interfaces:** Consumes F4决策状态、B3审批／风险、B4操作、B2同步错误。Produces `AlertService.record(event: BusinessAlertEvent) -> str`、`mark_read(alert_id)`、`DeliveryService.send(delivery_id)`、`retry(delivery_id)`。

事件类型为 approval_pending、execution_failed、protection_failed、risk_adjusted、component_failed、connection_failed、daily_summary。事件含稳定 event_key、type、occurred_at、capital_scope 可空、decision_id/book_id/connection_id/operation_id 可空及脱敏 message。只分析错误不标交易已发生。

- [ ] 写业务状态与投递分离测试：同 event_key 两次只一个告警，失败 Webhook 留 failed delivery，交易 completed 不变；mark_read 不调用 approve、不改变业务 resolution。

```python
alert_id = await alerts.record(event)
assert await alerts.record(event) == alert_id
await delivery.send(delivery_id)
assert (await store.get_delivery(delivery_id)).status == "failed"
assert (await decisions.get(decision_id)).status == "completed"
await alerts.mark_read(alert_id)
assert approval_store.approve.call_count == 0
```

- [ ] 红测：`rtk proxy env -u DATABASE_URL -u CONFIG_MASTER_KEY .venv/bin/python -m pytest --no-cov tests/test_alert_lifecycle.py tests/test_alert_delivery.py -q`。
- [ ] 业务状态提交后生成稳定键如 `approval:{id}:pending`、`book:{decision_id}:{book_id}:execution_failed`。周期事件 sink 复用实际状态；启动时从持久化待审批／失败状态补建遗漏告警，按同键去重，不从 GET 请求创建事项。失败信息从安全字段选择，不保存原请求体。

```python
await alert_store.record_once(event.event_key, event)
if event.type in notification_config.events:
    await alert_store.enqueue_delivery(event.event_key, channel="webhook")
```

`AlertStore.record_once` 与 `enqueue_delivery` 在本任务实现唯一约束，enqueue 对同告警／通道只创建一次；retry 增加 attempts，不复制业务告警。

- [ ] 将 NotificationConfig.events 扩为上述列表，沿用已有 Webhook 配置和秘密处理。业务保存和投递解耦，独立后台 owner 在浏览器关闭时仍处理 pending；投递错误不抛回交易终态。持久化 attempt、last_error、last_attempt_at 和 delivered_at，用户可重试；每日摘要同样记送达状态。
- [ ] 审批通过／拒绝、退出完成、连接恢复等真实业务状态更新关联事项 resolution；已读仅改变 read_at。风险压低本次目标的事项标为该次已处理限制，不能展示为需要批准。工作台列表链接决策／账户；系统页显示事件选择、最近投递和失败原因。
- [ ] 跑后端与前端目标测试：无浏览器仍发送、重复刷新不增告警、读取不会批准、通知失败交易不回滚、消息无凭据。只用 fake HTTP transport，不向用户已配置的 Webhook 发送测试消息。

# CryptoTrader AI 架构

## 1. 设计中心

系统只有一个顶层业务编排器：`TradingCycle`。模型预测、智能体辩论、规则策略都不能直接决定订单；它们只能作为 `SignalComponent` 提供标准化市场观点。仓位目标、退出策略、人工审批、风控和执行均由组件层之外的共享服务负责。

这条边界解决三个问题：

1. Kronos 与 LLM 委员会可以共同参与，而不是互斥运行。
2. 信任权重和决策参数可以在运行时调整，不需要改编排代码。
3. 实时、模拟、回测使用同一套决策语义，结果可以直接比较。

## 2. 模块边界

```text
src/cryptotrader/
├── signals/
│   ├── component.py          # SignalComponent 协议与组件错误
│   ├── models.py             # ComponentSignal / SignalContext / 数据需求
│   ├── registry.py           # 内置与插件组件注册
│   ├── runner.py             # 并行执行与严格成功策略
│   ├── fusion.py             # 确定性加权融合
│   ├── context.py            # point-in-time 上下文物化
│   └── components/
│       ├── kronos.py
│       └── llm_committee.py
├── profiles/                 # 全局 Profile 领域模型和持久化
├── decision/                 # TargetPosition 与融合后决策
├── execution/                # 差值规划、订单状态与保护单
├── risk/                     # 目标仓位风控
├── hitl/                     # 冻结计划的审批存储
├── journal/                  # trading_cycles 审计记录
├── backtest/                 # 历史上下文和模拟执行
├── bootstrap.py              # 生产依赖装配
└── trading_cycle.py          # 唯一顶层主链
```

Web 端的 `/strategy` 编辑全局 Profile；`/decisions` 读取周期 Journal；`/debate` 展示 `llm_committee.details` 中的内部分析与辩论轮次；`/risk` 和 `/hitl` 分别处理风险状态与人工审批。

## 3. 信号协议

组件必须提供：

- 稳定且唯一的 `id`。
- UI 使用的 `display_name` 和 `description`。
- `requirements()`：声明所需 K 线、链上、新闻、宏观或 Kronos 辅助数据。
- `evaluate(context)`：返回一个 `ComponentSignal`。

组件不能返回仓位比例、订单、杠杆、止损或止盈。这样组件可以专注于市场判断，组合决策不会被某个模型越权控制。

`SignalComponentRegistry` 负责安装组件。生产装配先注册 Kronos 和 LLM 委员会，再加载配置中的 factory。非法 factory、重复 ID 或 Profile 引用未安装组件都会让启动失败。

## 4. Point-in-time 数据

`LiveSignalContextProvider` 先合并所有启用组件和统一退出策略的数据需求，再一次性生成不可变 `SignalContext`。同一周期的所有组件共享 `as_of`，不同周期或回测时间窗不会看到未来 K 线。

上下文包含：交易对、模式、交易所、权益、当前价格、ATR、当前仓位、多周期快照和组合风险事实。`ExchangePortfolioReader` 同时读取余额和衍生品仓位，并补入交易所仓位接口遗漏的现货余额。

## 5. 融合与目标仓位

每个组件方向转换为单位分数：

```text
long = +1
short = -1
neutral = 0
component_score = direction × confidence
fused_score = Σ(weight × component_score)
```

Profile 校验保证所有启用权重合计为 `1.0`。`neutral_threshold` 把低绝对值分数映射为 flat；超过阈值后由 `DecisionEngine` 生成 `TargetPosition(side, size_ratio)`，并受 `max_target_ratio` 限制。

正负融合分数是内部数学表示。下游只接收显式方向和非负比例，避免把符号、仓位大小和用户配置混在同一个数里。

## 6. 退出、审批、风控与执行

`AtrExitPolicy` 是唯一退出价格来源。它根据当前价格、ATR、`atr_stop_multiplier` 和 `reward_ratio` 生成止损止盈；信号组件无权各自携带退出价格。

当冻结 Profile 的 `hitl_required=true` 时，实时和模拟周期在风控前创建 `ApprovalRecord` 并结束为 `awaiting_approval`。审批记录保存请求、SignalContext、Profile revision 和 TradePlan。批准后读取最新仓位，重新计算差值订单，再进入风控和执行；拒绝不会触达风控或交易所。回测永不等待人工输入。

`RiskGate` 接收不可变 `RiskRequest(context, plan)`。所有检查都会运行；异常按拒绝处理；多个仓位 cap 取最严格值。减仓和清仓路径绕过会阻止风险降低的外部依赖检查。

`ExecutionPlanner` 把当前仓位与目标仓位转换成顺序 `OrderIntent`。反手必须先平后开。`ExecutionService` 逐单确认成交，并在最终非空仓位上维护唯一保护单。

## 7. 生命周期与审计

一次周期冻结一个 Profile revision，配置更新不会改变已运行周期。周期可能结束为：完成、无需变更、等待审批、审批拒绝、组件失败、风控拒绝、执行失败或取消。

每次终止都写入 `trading_cycles`，内容包括上下文摘要、组件信号、组件错误、融合贡献、目标仓位、交易计划、HITL、风控和执行结果。API、网页、指标与 CLI 都从这个 Journal 读取事实。

## 8. 关键不变量

- 只有 `TradingCycle` 可以把信号推进到订单。
- 所有启用组件必须全部成功，失败周期不融合、不交易。
- 启用权重之和必须为 `1.0`。
- 同一周期只使用一个冻结 Profile revision 和一个 point-in-time 上下文。
- 组件输出不含订单与退出价格。
- 风控不修改原始计划，只返回拒绝或新的受限计划。
- 实时、模拟、回测共享融合、目标仓位、退出和风控语义。
- 网页配置保存后从下一周期生效。

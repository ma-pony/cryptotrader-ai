# 运行架构与扩展指南

## 请求到执行的数据流

FastAPI lifespan 调用 `build_trading_cycle()` 完成生产装配，并把周期实例、组件 Registry 和 Profile repository 放入 `app.state`。CLI 和 Scheduler 也调用同一装配函数；BacktestEngine 仅替换历史上下文与模拟交易所，决策主链不变。

```text
CycleRequest(pair, mode, exchange_id, as_of?)
  → SignalProfileRepository.get()
  → Registry.enabled(profile)
  → DataRequirements.merge(...)
  → SignalContextProvider.collect()
  → ComponentRunner.run()
  → WeightedSignalFusion.fuse()
  → DecisionEngine.decide()
  → AtrExitPolicy.apply()
  → ApprovalStore / RiskGate / ExecutionPlanner / ExecutionService
  → CycleJournalStore.append()
```

周期事件通过 `CycleEventSink` 发布。网页分析进度直接消费 `cycle_started`、组件状态、委员会智能体与辩论轮次、融合、审批、风控、执行和终止事件。

## 添加自定义组件

实现组件协议：

```python
class MySignal:
    id = "my_signal"
    display_name = "My Signal"
    description = "What evidence this component contributes"

    def requirements(self) -> DataRequirements:
        return DataRequirements(candles=(CandleRequirement("1h", 200),))

    async def evaluate(self, context: SignalContext) -> ComponentSignal:
        return ComponentSignal(
            component_id=self.id,
            direction="neutral",
            confidence=0.6,
            reasoning="...",
        )


def create_component():
    return MySignal()
```

注册 factory：

```toml
[signal_plugins]
factories = ["my_package.signals:create_component"]
```

然后在 Strategy 页面启用组件并重新分配权重。保存前端和 API 都会检查启用权重总和；服务端仍是最终约束来源。

## Profile 更新语义

Profile 使用单行 `global` 记录，完整替换而不是逐字段 patch。每次保存 revision 加一。周期开始时只读取一次 Profile，后续所有 Journal、HITL 和恢复操作都携带该 revision。因此动态设置不会在周期中途改变判断，也不需要进程重启。

## 失败语义

- Factory 加载或 Profile 校验失败：启动失败。
- 组件超时、异常或返回非法信号：周期 `component_failed`。
- HITL 拒绝：周期 `approval_rejected`。
- 风控拒绝：周期 `risk_rejected`，保留拒绝规则和原因。
- 任一订单未成交或保护单失败：周期 `execution_failed`。
- 调用取消：周期写入 `cancelled` 后继续传播取消异常。

系统不使用部分组件结果继续交易，也不在失败时切换到隐藏的替代决策器。

## 数据与执行注意事项

- 多周期 K 线都裁剪到 `as_of`。
- Kronos 辅助数据按组件 requirements 才加载。
- 现货余额由 `ExchangePortfolioReader` 补入，dust 小于 `1e-6` 时忽略。
- 永续权益只计未实现盈亏，不能把合约名义价值重复加入现金。
- 反手执行顺序固定为 reduce-only 平仓，再开相反方向。
- 保护单以最终仓位总量创建，旧保护单先取消。

## 前端页面

- `/strategy`：组件与决策参数；明确提示下一周期生效。
- `/decisions`：按周期展示组件贡献、融合、目标、计划、风险和执行。
- `/debate`：展示 LLM 委员会内部分析、质询和收敛过程。
- `/risk`：组合风险状态和最近拒绝。
- `/hitl`：待审批的冻结目标计划。

## 验证边界

`tests/test_signal_architecture_boundary.py` 确保旧的顶层交易结构不会重新进入运行时代码。新增组件至少应覆盖协议校验、方向/置信度输出、失败传播以及其数据需求；跨模式行为由 TradingCycle 的共享测试覆盖。

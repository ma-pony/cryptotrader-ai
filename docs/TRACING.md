# TradingCycle 可观测性

新运行时不再追踪顶层 LangGraph 节点。Scheduler、CLI、Chat、paper、live 和 backtest 都进入同一个 `TradingCycle`，可观测性也按这条业务链路组织。

## Trace ID

API 中间件、CLI 和 Scheduler 在进入周期前通过 `set_trace_id()` 绑定 trace ID。结构化日志通过 `get_trace_id()` 读取当前值。

`node_logger()` 仅用于需要单独计时的异步边界，不承担业务状态传递。

## CycleEvent

`TradingCycle` 通过 `CycleEventSink` 发布类型化的阶段事件：

- `cycle_started`
- `context_ready`
- `component_started` / `component_completed` / `component_failed`
- `agent_analysis_completed`
- `debate_round_completed`
- `fusion_completed`
- `decision_created`
- `approval_required`
- `risk_checked`
- `execution_completed`
- `cycle_completed` / `cycle_failed` / `cycle_cancelled`

Scheduler 使用空 Sink。Chat 使用 `EventBusCycleSink` 将同一批事件转成 SSE，不会从未完成的组件结果生成部分决策。

## Journal 与 OpenTelemetry

`CycleJournalStore` 是业务审计源，保存 Profile revision、组件信号、融合贡献、目标仓位、HITL、风控和执行结果。

OpenTelemetry 是可选的基础设施级追踪；未配置 collector 时不改变 TradingCycle 的业务语义。

# 可插拔信号融合架构重构设计

日期：2026-08-27
状态：已在讨论中确认

> 后续设计：数据库配置、多交易平台执行以及模拟盘/实盘连接模型，
> 以 `2026-08-28-database-config-multi-venue-execution-design.md` 为准。
> 本文中涉及 TOML、单一 Executor、单一持仓上下文和实盘执行的描述已被后续设计替代。

## 1. 背景

当前交易主链通过 `signal_engine` 在 LLM 四智能体图和 Kronos 图之间二选一。两条路径都直接生成包含方向、仓位、止盈止损和持仓动作的 `TradeVerdict`，导致信号判断、仓位决策、退出策略与执行语义耦合，无法把 Kronos、LLM 委员会和后续自定义信号作为平等组件进行可信度加权。

本次重构直接替换旧决策架构，不保留旧 Graph、`ArenaState`、`TradeVerdict`、`signal_engine`、旧 CLI 图模式或兼容适配层。LangGraph 只保留在真正需要多智能体并行与循环辩论的 LLM 委员会内部。

## 2. 目标

- Kronos、LLM 四智能体委员会和自定义信号实现统一组件协议。
- 组件按全局可配置权重进行确定性融合。
- 最终决策表达目标仓位，而不是命令式买卖动作。
- 仓位、止盈止损、HITL、硬风控和执行保持清晰分层。
- 网页可动态调整已安装组件的启用状态、权重和决策参数。
- 实盘和回测使用同一套组件、融合与决策主链。
- 删除旧代码和旧测试，不维护双轨行为。

## 3. 非目标

- 不通过网页上传、安装或执行任意组件代码。
- 不支持按交易对覆盖 SignalProfile。
- 不支持多个 Profile、草稿、发布流程或历史版本回滚。
- 不支持 Shadow 组件、部分组件降级、覆盖率或运行时权重归一化。
- 不支持动态学习权重或按市场状态自动调权。
- 不支持组件失败后的 mock signal 或 partial verdict。
- 不为被替换的信号、Graph、CLI 图模式和决策模型保留兼容接口。

## 4. 总体架构

所有入口统一调用 `TradingCycle.run(CycleRequest)`：

```text
Scheduler ─┐
CLI ───────┼──→ TradingCycle.run(CycleRequest)
API/Chat ──┤
Backtest ──┘
```

`TradingCycle` 使用普通 Python application service 编排确定性业务主链：

```text
ProfileRepository
      ↓ SignalProfile
ComponentRegistry
      ↓ enabled components + requirements
ContextProvider
      ↓ SignalContext
ComponentRunner
      ↓ list[ComponentSignal]
WeightedSignalFusion
      ↓ FusedSignal
DecisionEngine
      ↓ TargetPosition
ExitPolicy
      ↓ TradePlan
HITL
      ↓ approved TradePlan
RiskGate
      ↓ allowed/clamped TradePlan
ExecutionPlanner
      ↓ OrderIntent
Executor
      ↓ ExecutionResult
DecisionJournal
```

职责如下：

- `ContextProvider`：根据统一 `as_of` 和组件数据需求构建不可变上下文。
- `ProfileRepository`：在周期开始时读取唯一的全局活动配置。
- `ComponentRegistry`：解析 Profile 中已启用的组件并合并数据需求。
- `ComponentRunner`：并发运行所有已启用组件，执行全部成功才进入融合。
- `WeightedSignalFusion`：按活动 Profile 做确定性加权。
- `DecisionEngine`：把融合分数映射为目标仓位。
- `ExitPolicy`：统一生成 ATR 止盈止损。
- `HitlGate`：根据配置直接放行或保存待审批计划。
- `RiskGate`：拥有最终否决和降低目标仓位的权力。
- `ExecutionPlanner`：比较当前仓位与目标仓位，生成订单意图。
- `Executor`：实现 paper/live/backtest 执行差异。
- `DecisionJournal`：记录完整决策和执行链路。

## 5. LLM 委员会

LLM 四智能体保留内部分析和辩论能力，并作为一个外部信号组件：

```text
LLMCommitteeComponent
  → Tech/Chain/News/Macro 四智能体并行分析
  → 辩论门控
  → 多轮交叉辩论
  → 委员会汇总
  → ComponentSignal
```

这部分内部继续使用 LangGraph，并定义自己的 `CommitteeState`。它不再读取顶层 `ArenaState`，也不再生成仓位、止盈止损或交易动作。任意智能体、辩论或汇总失败时，整个 LLM 组件失败。

## 6. 组件协议与数据上下文

组件协议：

```python
class SignalComponent(Protocol):
    id: str

    def requirements(self) -> DataRequirements:
        ...

    async def evaluate(self, context: SignalContext) -> ComponentSignal:
        ...
```

内置组件为：

- `KronosComponent`
- `LLMCommitteeComponent`

自定义组件通过 TOML 中的 Python factory 注册：

```toml
[signal_plugins]
factories = [
  "my_package.signals:CustomSignalComponent",
]
```

组件代码必须已安装在运行环境中，修改 factory 或组件代码后重启服务。Registry 在启动时实例化组件，并拒绝重复 ID。

每个组件声明所需 K 线周期、bar 数量及辅助数据。`ContextProvider` 合并所有启用组件的需求，以相同 `as_of` 构建上下文：

```python
@dataclass(frozen=True)
class SignalContext:
    pair: Pair
    as_of: datetime
    current_price: float
    current_position: Position
    candles: dict[str, CandleSeries]
    onchain: OnchainSnapshot
    news: NewsSnapshot
    macro: MacroSnapshot
```

组件只能读取 `SignalContext`，不能自行获取未绑定 `as_of` 的实时数据。Kronos 的同步模型推理通过工作线程执行，避免阻塞异步 LLM 调用。

组件输出：

```python
@dataclass(frozen=True)
class ComponentSignal:
    component_id: str
    direction: Literal["long", "short", "neutral"]
    confidence: float
    reasoning: str
    details: dict[str, Any]
```

`direction` 表示市场观点，不是下单动作。`confidence` 必须位于 `[0, 1]`。

## 7. 信号融合

融合器把组件方向转换为有符号分数：

```python
signed_score = {
    "long": confidence,
    "short": -confidence,
    "neutral": 0.0,
}[direction]

fused_score = sum(weight * signed_score)
```

输出包含每个组件的原始信号、配置权重和加权贡献：

```python
@dataclass(frozen=True)
class ComponentContribution:
    component_id: str
    weight: float
    signed_score: float
    weighted_score: float

@dataclass(frozen=True)
class FusedSignal:
    score: float
    contributions: tuple[ComponentContribution, ...]
    reasoning: str
```

所有启用组件权重必须在保存 Profile 时显式合计为 `1.0`。运行时不重新归一化。任意启用组件失败，本轮不产生 `FusedSignal`。

## 8. 目标仓位与退出策略

最终决策表达目标状态：

```python
@dataclass(frozen=True)
class TargetPosition:
    side: Literal["long", "short", "flat"]
    size_ratio: float
```

`size_ratio` 表示风险配置允许的最大单一仓位中的占比。内部计算时转换为 `signed_exposure`：多仓为正、空仓为负、空仓状态为零。

`DecisionEngine` 使用全局 Profile 中的单一中性阈值做线性映射：

```python
if abs(fused_score) <= neutral_threshold:
    target = TargetPosition(side="flat", size_ratio=0.0)
else:
    size_ratio = (
        (abs(fused_score) - neutral_threshold)
        / (1 - neutral_threshold)
        * max_target_ratio
    )
    target = TargetPosition(
        side="long" if fused_score > 0 else "short",
        size_ratio=size_ratio,
    )
```

不存在 `long/short/hold/close` 动作和 `position_scale`。`ExecutionPlanner` 把当前仓位和目标仓位转换为有符号名义价值，差值统一表达开仓、加仓、减仓、平仓和反向。

`ExitPolicy` 是唯一止盈止损来源。它使用统一 ATR 和 Profile 参数：

```text
long:
  stop_loss   = entry - ATR × atr_stop_multiplier
  take_profit = entry + risk_distance × reward_ratio

short:
  stop_loss   = entry + ATR × atr_stop_multiplier
  take_profit = entry - risk_distance × reward_ratio
```

组件可以在 `details` 中提供预测信息，但不能控制最终止盈止损。

完整计划：

```python
@dataclass(frozen=True)
class TradePlan:
    target: TargetPosition
    stop_loss: float | None
    take_profit: float | None
    component_signals: tuple[ComponentSignal, ...]
    fused_signal: FusedSignal
```

## 9. 全局动态配置

系统只维护一份全局活动配置：

```python
@dataclass(frozen=True)
class ComponentWeight:
    component_id: str
    enabled: bool
    weight: float

@dataclass(frozen=True)
class SignalProfile:
    revision: int
    components: tuple[ComponentWeight, ...]
    neutral_threshold: float
    max_target_ratio: float
    atr_stop_multiplier: float
    reward_ratio: float
    hitl_required: bool
```

PostgreSQL 表：

```text
signal_profile
├── id = "global"
├── revision BIGINT
├── config JSONB
└── updated_at TIMESTAMPTZ
```

TOML 默认值只用于数据库中尚无 Profile 的首次启动，之后数据库是运行时唯一配置来源。

每个交易周期开始时读取一次 Profile，整个周期使用同一不可变快照；修改从下一周期开始生效。回测启动时同样读取一次，整次回测固定该 revision。Journal 保存 Profile revision 和完整权重。

API：

```text
GET /api/signal-profile
PUT /api/signal-profile
```

`GET` 返回活动配置及已安装组件元数据。`PUT` 完整替换全局配置，并验证：

- 至少启用一个已注册组件。
- 组件 ID 唯一。
- 启用组件权重位于 `[0, 1]` 且总和为 `1.0`。
- `0 <= neutral_threshold < 1`。
- `0 < max_target_ratio <= 1`。
- ATR 倍数和盈亏比为正数。

不提供单字段 patch、草稿或 Profile CRUD。

## 10. Strategy 页面

新增 `/strategy` 页面和侧边栏入口：

```text
策略配置
├── 信号组件
│   ├── Kronos          [启用] [60%]
│   ├── LLM 四智能体    [启用] [40%]
│   └── Custom Factor   [关闭] [0%]
├── 决策参数
│   ├── 中性阈值
│   └── 最大目标仓位
├── 退出策略
│   ├── ATR 止损倍数
│   └── 盈亏比
├── 人工审批
│   └── 启用 HITL
└── [保存并从下一周期生效]
```

页面显示当前权重总计，不等于 100% 时禁止保存。保存成功后显示 revision 和更新时间。网页只能配置 Registry 中已经安装的组件。

## 11. HITL、风控与执行

HITL 关闭时，`TradePlan` 直接进入 `RiskGate`。HITL 开启时，计划由 `ApprovalRepository` 保存并返回 `awaiting_approval`。网页批准后重新读取当前仓位，以原始目标仓位继续执行：

```text
pending TradePlan
      ↓ approve
RiskGate
      ↓
ExecutionPlanner
      ↓
Executor
```

保存的是目标仓位，不是固定订单数量，因此审批时根据最新当前仓位计算差值。拒绝审批后记录结果并结束。

硬风控保持独立并拥有最终决定权：它可以拒绝计划、降低目标仓位或将目标改为 `flat`。实盘保护单由交易所侧维护；paper/backtest Executor 在处理行情时执行已有止盈止损。组件失败不能阻止已有仓位保护。

## 12. 实盘与回测

实盘和回测共用以下对象：

- `ComponentRunner`
- `WeightedSignalFusion`
- `DecisionEngine`
- `ExitPolicy`
- `RiskGate`
- `ExecutionPlanner`

差异仅通过依赖注入提供：

- 实盘使用实时 `ContextProvider` 和交易所 `Executor`。
- Paper 使用实时 `ContextProvider` 和模拟 `Executor`。
- Backtest 使用历史 `ContextProvider` 和回测 `Executor`。
- HITL 在回测中固定关闭。

不保留 lite graph、Kronos backtest graph 或独立回测 Verdict 逻辑。

## 13. 事件与 Chat

`TradingCycle` 接收可选事件出口：

```python
class CycleEventSink(Protocol):
    async def publish(self, event: CycleEvent) -> None:
        ...
```

业务事件包括：

```text
cycle_started
context_ready
component_started
agent_analysis_completed
debate_round_completed
component_completed
fusion_completed
decision_created
approval_required
risk_checked
execution_completed
cycle_failed
cycle_completed
```

Scheduler 使用空事件出口，Chat 使用现有 EventBus 的新适配器。LLM 委员会把内部智能体和辩论事件发布到同一出口。

取消操作直接取消当前周期，不产生部分结果。删除 partial verdict；不允许使用未完成的 Agent 结果进行融合。

## 14. 运行结果与 Journal

周期状态：

```python
CycleStatus = Literal[
    "completed",
    "no_change",
    "awaiting_approval",
    "approval_rejected",
    "component_failed",
    "risk_rejected",
    "execution_failed",
    "cancelled",
]
```

每个周期均写入新 Journal 模型，包括失败周期：

```text
cycle_id
profile_revision
context_summary
component_signals
component_error
fused_signal
target_position
trade_plan
hitl_result
risk_result
execution_result
status
created_at
```

新运行时使用新的 `trading_cycles` 表和结构化 JSONB 字段。旧 `decision_commits` 数据不被新代码读取或写入，但迁移不会主动删除历史生产数据；删除旧表属于独立的数据清理操作，不在本次重构范围内。

Decisions API 和页面改为读取新 Journal，展示组件信号、权重贡献、融合分数、目标仓位、HITL、风控和执行结果。

## 15. 失败规则

- 配置不合法时拒绝保存；启动时 Registry 或默认 Profile 不合法则启动失败。
- 任意启用组件失败，本轮信号交易以 `component_failed` 结束。
- 不生成 mock signal，不剔除失败组件，不修改权重。
- 合法的 `neutral` 是成功信号。
- HITL 拒绝后不进入风控和执行。
- 风控拒绝记录为 `risk_rejected`。
- 执行异常记录为 `execution_failed`，不修改真实风控结果。
- 取消周期不产生融合或部分 Verdict。

## 16. 代码布局

```text
src/cryptotrader/
├── trading_cycle.py
├── signals/
│   ├── models.py
│   ├── component.py
│   ├── registry.py
│   ├── runner.py
│   ├── fusion.py
│   └── components/
│       ├── kronos.py
│       └── llm_committee.py
├── decision/
│   ├── models.py
│   ├── engine.py
│   └── exit_policy.py
├── profiles/
│   ├── models.py
│   └── repository.py
└── cycle_events.py

src/api/routes/signal_profile.py
web/src/pages/strategy/
```

具体基础设施实现继续放在现有 data、risk、execution、journal 和 exchange 包中，但接口改为接收明确领域模型，不接收通用状态字典。

## 17. 删除范围

本次实现完成后删除：

- `src/cryptotrader/graph.py`
- 顶层 `ArenaState`、`merge_dicts` 和 `build_initial_state`
- `src/cryptotrader/nodes/` 旧主链节点
- `signal_engine` 配置和全部分支
- `TradeVerdict`、`verdict_source`、`action`、`position_scale`
- full/lite/debate/kronos/backtest 多套顶层 Graph
- CLI `--graph` 参数
- partial verdict
- 旧 Graph re-export、监督器依赖和兼容测试

四智能体、辩论、Kronos 特征、硬风控、交易所适配器和存储算法会被提取到新边界，不保留旧函数入口 wrapper。

## 18. 测试策略

实现严格采用 TDD，测试顺序与生产代码顺序一致：

1. `ComponentSignal`、`SignalProfile` 验证和 Registry。
2. 加权融合数学与贡献明细。
3. 融合分数到 `TargetPosition` 的线性映射。
4. ATR 多空退出价格。
5. 当前仓位到目标仓位的订单差值，包括减仓和反向。
6. 全组件成功、单组件失败和有效中性结果。
7. HITL 开启、关闭、批准和拒绝。
8. 风控缩小目标仓位与拒绝执行。
9. 相同 Context/Profile 下实盘和回测生成相同 `TradePlan`。
10. Profile GET/PUT API、非法权重拒绝和下一周期生效。
11. Strategy 页面加载、编辑、权重校验和保存。
12. Chat 事件流和取消。
13. Journal 对成功、组件失败、风控拒绝和执行失败的记录。

单元和集成测试使用可控假组件、假 LLM、假 RiskGate 和模拟 Executor，不调用真实 LLM、Kronos 生产模型或交易所。

## 19. 完成标准

- 旧架构符号 `signal_engine`、`ArenaState`、`TradeVerdict`、`build_*_graph` 和 partial verdict 在运行时代码中搜索结果为零。
- Scheduler、CLI、API、Chat 和 Backtest 全部调用 `TradingCycle`。
- Kronos、LLM 委员会和一个测试自定义组件通过同一契约运行。
- 网页保存 Profile 后，从下一周期生效并写入 Journal revision。
- Python 全量测试通过。
- Ruff 通过。
- 前端单元测试通过。
- TypeScript 类型检查通过。
- 前端生产构建通过。
- 不执行真实交易和真实外部模型调用。

# 可插拔信号融合架构重构 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 彻底删除顶层 LangGraph 二选一决策架构，交付可动态配置的 Kronos、LLM 四智能体委员会和自定义组件融合主链，并让实盘、Paper、Chat、HITL、Journal、网页与回测完整可用。

**Architecture:** 顶层改为普通 Python `TradingCycle` application service；LangGraph 仅存在于 `LLMCommitteeComponent` 内部。所有组件输出 `direction + confidence`，确定性加权后由 `DecisionEngine` 生成目标仓位，统一经过 ATR ExitPolicy、可配置 HITL、硬风控、订单差值规划和执行。

**Tech Stack:** Python 3.12、dataclasses、asyncio、LangGraph、SQLAlchemy async、FastAPI、PostgreSQL/SQLite 测试、React 19、TypeScript、TanStack Query、Zod、Vitest、pytest、Ruff。

**Spec:** `docs/superpowers/specs/2026-08-27-pluggable-signal-fusion-refactor-design.md`

## Global Constraints

- 不保留 `signal_engine`、顶层 `ArenaState`、`TradeVerdict`、旧 Graph、旧 CLI 图模式或运行时兼容 wrapper。
- LLM 四智能体分析、辩论门控和多轮交叉辩论必须保留在 `LLMCommitteeComponent` 内部。
- 所有启用组件必须成功后才融合；禁止 mock signal、部分降级、动态归一化和 partial verdict。
- 组件输出统一为 `direction: long|short|neutral` 与 `confidence: [0, 1]`。
- 最终决策统一为 `TargetPosition(side, size_ratio)`；执行层根据当前与目标仓位差值处理开仓、加仓、减仓、平仓和反向。
- 止盈止损只能由统一 ATR ExitPolicy 生成。
- SignalProfile 全局唯一，网页修改后从下一周期生效，当前周期和当前回测固定 revision。
- 自定义组件代码只从启动时 TOML factory 注册，网页不能安装代码。
- 实盘、Paper 和 Backtest 共用组件、融合、决策、退出和执行规划逻辑。
- 不调用真实 LLM、Kronos 生产模型或交易所完成自动化测试。
- 所有项目 Markdown 使用简体中文。

## 设计覆盖矩阵

| 设计章节 | 实施任务 | 验收证据 |
|---|---|---|
| 2–4 目标、非目标、总体架构 | Task 1–3、12、18 | 领域契约、纯 Python TradingCycle、旧 Graph 禁止回归测试 |
| 5 LLM 委员会 | Task 9 | 四智能体并发、内部辩论、严格失败测试 |
| 6 组件协议与上下文 | Task 2、6–9 | Registry、自定义 factory、point-in-time context、组件测试 |
| 7 信号融合 | Task 3、7 | 精确贡献数学、缺失信号失败、全成功 Runner 测试 |
| 8 目标仓位与退出 | Task 3–5 | 线性目标映射、ATR ExitPolicy、仓位差值和风险 cap 测试 |
| 9–10 动态配置与 Strategy 页面 | Task 10、16 | Profile revision API、网页保存/校验、下一周期生效 E2E |
| 11 HITL、风控与执行 | Task 5、12、17 | 暂停/批准/拒绝、11 项风控迁移、顺序执行与 OCO 测试 |
| 12 实盘与回测 | Task 6、12–14 | 相同 context/信号产生相同 TradePlan 的 parity 测试 |
| 13 事件与 Chat | Task 7、13、17 | 进度事件、取消、无 partial verdict 测试 |
| 14 Journal | Task 11、15、17 | 成功/失败周期 round-trip、Decisions 页面契约测试 |
| 15 失败规则 | Task 7、9、12、19 | 组件失败不融合不下单、风险/执行/取消状态 E2E |
| 16–17 布局与删除范围 | Task 1–18 | 文件职责、依赖方向、旧符号零命中门禁 |
| 18–19 测试与完成标准 | Task 19 | Python/前端全量测试、coverage、Ruff、构建、容器配置 |

---

## 文件结构锁定

新建或重写后的核心文件职责如下：

```text
src/cryptotrader/
├── trading_cycle.py                    # 单次交易周期 application service
├── bootstrap.py                        # 组装生产依赖
├── cycle_events.py                     # 业务事件与 EventSink
├── signals/
│   ├── __init__.py
│   ├── models.py                       # 数据需求、上下文、组件信号
│   ├── component.py                    # SignalComponent 协议与组件错误
│   ├── registry.py                     # 内置及 Python factory 注册
│   ├── runner.py                       # 全组件并发、全部成功语义
│   ├── fusion.py                       # 确定性权重融合
│   ├── context.py                      # Live/Historical ContextProvider
│   └── components/
│       ├── __init__.py
│       ├── kronos.py                   # Kronos 纯信号组件
│       └── llm_committee.py            # 四智能体内部 LangGraph
├── decision/
│   ├── __init__.py
│   ├── models.py                       # Position/Target/TradePlan/CycleOutcome
│   ├── engine.py                       # fused score → TargetPosition
│   └── exit_policy.py                  # 统一 ATR SL/TP
├── profiles/
│   ├── __init__.py
│   ├── models.py                       # SignalProfile 及验证
│   └── repository.py                   # 单行全局 Profile 持久化
├── risk/
│   ├── models.py                       # RiskRequest/RiskDecision
│   └── gate.py                         # 对 TradePlan/TargetPosition 风控
├── execution/
│   ├── planner.py                      # 仓位差值 → ExecutionPlan
│   └── service.py                      # paper/live 执行与保护单
└── journal/
    ├── models.py                       # TradingCycleRecord
    └── store.py                        # trading_cycles 新表

src/api/routes/signal_profile.py        # GET/PUT 全局策略配置
web/src/pages/strategy/                 # 动态策略配置页
```

依赖方向固定为：`domain models → pure services → infrastructure adapters → bootstrap/entry points`。领域模型不得导入 FastAPI、SQLAlchemy、ccxt 或 React 相关设施。

---

### Task 1: 建立信号、仓位与 Profile 领域模型

**Files:**
- Create: `src/cryptotrader/signals/__init__.py`
- Create: `src/cryptotrader/signals/models.py`
- Create: `src/cryptotrader/decision/__init__.py`
- Create: `src/cryptotrader/decision/models.py`
- Create: `src/cryptotrader/profiles/__init__.py`
- Create: `src/cryptotrader/profiles/models.py`
- Create: `tests/factories/__init__.py`
- Create: `tests/factories/signal_fusion.py`
- Test: `tests/test_signal_domain.py`
- Test: `tests/test_signal_profile.py`

**Interfaces:**
- Produces: `CandleRequirement`, `DataRequirements`, `PositionSnapshot`, `SignalContext`, `ComponentSignal`。
- Produces: `TargetPosition`, `TradePlan`, `CycleRequest`, `CycleStatus`, `CycleOutcome`。
- Produces: `ComponentWeight`, `SignalProfile`, `validate_signal_profile()`。
- Produces: 后续测试唯一共用的 `position()`、`context()`、`profile()`、`signal()`、`trade_plan()`、`request()` 工厂。

- [ ] **Step 1: 写领域模型失败测试**

```python
def test_component_signal_rejects_confidence_outside_unit_interval():
    with pytest.raises(ValueError, match="confidence"):
        ComponentSignal("kronos", "long", 1.01, "x")


def test_data_requirements_merge_uses_largest_limit_per_timeframe():
    merged = DataRequirements.merge(
        DataRequirements(candles=(CandleRequirement("1h", 100),)),
        DataRequirements(candles=(CandleRequirement("1h", 200), CandleRequirement("4h", 512))),
    )
    assert merged.candles == (
        CandleRequirement("1h", 200),
        CandleRequirement("4h", 512),
    )


def test_target_position_flat_requires_zero_size():
    with pytest.raises(ValueError, match="flat"):
        TargetPosition(side="flat", size_ratio=0.2)
```

- [ ] **Step 2: 写 Profile 失败测试**

```python
def test_profile_requires_enabled_weights_to_sum_to_one():
    profile = SignalProfile(
        revision=1,
        components=(
            ComponentWeight("kronos", True, 0.6),
            ComponentWeight("llm_committee", True, 0.3),
        ),
        neutral_threshold=0.2,
        max_target_ratio=1.0,
        atr_stop_multiplier=2.0,
        reward_ratio=2.0,
        hitl_required=False,
    )
    with pytest.raises(ValueError, match="1.0"):
        validate_signal_profile(profile, {"kronos", "llm_committee"})


def test_profile_rejects_uninstalled_component():
    missing_profile = profile(ComponentWeight("missing", True, 1.0))
    with pytest.raises(ValueError, match="missing"):
        validate_signal_profile(missing_profile, {"kronos"})
```

- [ ] **Step 3: 运行测试确认因模块不存在而失败**

Run: `uv run pytest tests/test_signal_domain.py tests/test_signal_profile.py --no-cov -q`

Expected: FAIL，错误包含 `ModuleNotFoundError: cryptotrader.signals` 或缺少目标类。

- [ ] **Step 4: 实现不可变领域模型和验证**

```python
@dataclass(frozen=True, order=True)
class CandleRequirement:
    timeframe: str
    limit: int

    def __post_init__(self) -> None:
        if not self.timeframe.strip() or self.limit <= 0:
            raise ValueError("candle requirement requires timeframe and positive limit")


@dataclass(frozen=True)
class ComponentSignal:
    component_id: str
    direction: Literal["long", "short", "neutral"]
    confidence: float
    reasoning: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.component_id.strip():
            raise ValueError("component_id must not be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")


@dataclass(frozen=True)
class TargetPosition:
    side: Literal["long", "short", "flat"]
    size_ratio: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.size_ratio <= 1.0:
            raise ValueError("size_ratio must be in [0, 1]")
        if self.side == "flat" and self.size_ratio != 0.0:
            raise ValueError("flat target requires size_ratio=0")
        if self.side != "flat" and self.size_ratio == 0.0:
            raise ValueError("non-flat target requires positive size_ratio")

    @property
    def signed_ratio(self) -> float:
        return {"long": self.size_ratio, "short": -self.size_ratio, "flat": 0.0}[self.side]
```

其余领域模型在本任务一次锁定，不允许后续任务临时加字段：

```python
@dataclass(frozen=True)
class PositionSnapshot:
    side: Literal["long", "short", "flat"]
    amount: float
    size_ratio: float
    avg_price: float | None = None
    unrealized_pnl: float = 0.0

    @property
    def signed_amount(self) -> float:
        return {"long": self.amount, "short": -self.amount, "flat": 0.0}[self.side]

    @property
    def signed_ratio(self) -> float:
        return {"long": self.size_ratio, "short": -self.size_ratio, "flat": 0.0}[self.side]


@dataclass(frozen=True)
class SignalContext:
    pair: Pair
    as_of: datetime
    mode: Literal["paper", "live", "backtest"]
    exchange_id: str
    market_type: MarketType
    equity: float
    current_price: float
    atr: float
    current_position: PositionSnapshot
    snapshots: Mapping[str, DataSnapshot]


@dataclass(frozen=True)
class CycleRequest:
    pair: Pair
    mode: Literal["paper", "live", "backtest"]
    exchange_id: str = ""
    as_of: datetime | None = None


@dataclass(frozen=True)
class CycleOutcome:
    cycle_id: str
    status: CycleStatus
    profile_revision: int
    component_signals: tuple[ComponentSignal, ...] = ()
    fused_signal: FusedSignal | None = None
    trade_plan: TradePlan | None = None
    risk_result: RiskDecision | None = None
    execution_result: ExecutionResult | None = None
    approval_id: str | None = None
    error: str | None = None
```

`SignalContext.snapshots` 每个 key 是 timeframe，每个 `DataSnapshot.timestamp` 必须等于 `as_of`；`market_type` 直接取自 `Pair.market_type`，不再另建第二套交易对语义。`PositionSnapshot.signed_amount` 和 `signed_ratio` 对 long 返回正值、short 返回负值、flat 返回零。为避免 domain package 反向依赖，`CycleOutcome` 中的 `FusedSignal`、`RiskDecision`、`ExecutionResult` 通过 `TYPE_CHECKING` 与 postponed annotations 引用。

`tests/factories/signal_fusion.py` 集中实现以下确定签名，所有后续 Python 测试都从这里导入，不再各自发明同名但形状不同的 fixture：

```python
def position(side="flat", amount=0.0, size_ratio=0.0, **overrides) -> PositionSnapshot:
    values = {"side": side, "amount": amount, "size_ratio": size_ratio}
    return PositionSnapshot(**(values | overrides))


def context(price=100.0, equity=10_000.0, position=None, atr=5.0, market_type="swap", **overrides) -> SignalContext:
    values = {
        "pair": Pair.parse("BTC/USDT:USDT" if market_type == "swap" else "BTC/USDT"),
        "as_of": datetime(2026, 1, 1, tzinfo=UTC),
        "mode": "paper",
        "exchange_id": "okx",
        "market_type": market_type,
        "equity": equity,
        "current_price": price,
        "atr": atr,
        "current_position": position or PositionSnapshot("flat", 0.0, 0.0),
        "snapshots": {},
    }
    return SignalContext(**(values | overrides))


def profile(*components, kronos=0.6, llm=0.4, revision=1, neutral_threshold=0.2, max_target_ratio=1.0, atr_stop_multiplier=2.0, reward_ratio=2.0, hitl=False) -> SignalProfile:
    configured = components or (
        ComponentWeight("kronos", kronos > 0.0, kronos),
        ComponentWeight("llm_committee", llm > 0.0, llm),
    )
    return SignalProfile(
        revision=revision,
        components=configured,
        neutral_threshold=neutral_threshold,
        max_target_ratio=max_target_ratio,
        atr_stop_multiplier=atr_stop_multiplier,
        reward_ratio=reward_ratio,
        hitl_required=hitl,
    )


def signal(component_id="kronos", direction="long", confidence=0.8, **overrides) -> ComponentSignal:
    values = {"component_id": component_id, "direction": direction, "confidence": confidence, "reasoning": "fixture"}
    return ComponentSignal(**(values | overrides))


def trade_plan(target: TargetPosition, **overrides) -> TradePlan:
    fused = FusedSignal(score=0.0, contributions=(), reasoning="fixture")
    values = {"target": target, "stop_loss": None, "take_profit": None, "component_signals": (), "fused_signal": fused}
    return TradePlan(**(values | overrides))


def request(pair="BTC/USDT:USDT", mode="paper", exchange_id="okx", as_of=None) -> CycleRequest:
    return CycleRequest(Pair.parse(pair), mode, exchange_id, as_of)
```

`validate_signal_profile()` 必须验证组件 ID 唯一、至少一个启用组件、启用权重使用 `math.isclose(total, 1.0, abs_tol=1e-9)`、阈值范围和已安装 ID。

- [ ] **Step 5: 运行领域测试**

Run: `uv run pytest tests/test_signal_domain.py tests/test_signal_profile.py --no-cov -q`

Expected: PASS。

- [ ] **Step 6: 提交领域模型**

```bash
git add src/cryptotrader/signals src/cryptotrader/decision src/cryptotrader/profiles tests/factories tests/test_signal_domain.py tests/test_signal_profile.py
git commit -m "feat: add signal fusion domain models"
```

---

### Task 2: 建立组件协议、Registry 和静态默认配置

**Files:**
- Create: `src/cryptotrader/signals/component.py`
- Create: `src/cryptotrader/signals/registry.py`
- Modify: `src/cryptotrader/config.py`
- Modify: `config/default.toml`
- Test: `tests/test_signal_registry.py`
- Modify: `tests/test_config_loader.py`

**Interfaces:**
- Consumes: Task 1 的 `DataRequirements`、`SignalContext`、`ComponentSignal`、`SignalProfile`。
- Produces: `SignalComponent` Protocol、`ComponentExecutionError`、`ComponentMetadata`、`SignalComponentRegistry`。
- Produces: `SignalPluginsConfig.factories` 和 `SignalProfileDefaultsConfig.to_profile()`。

- [ ] **Step 1: 写 Registry 与配置失败测试**

```python
class FakeComponent:
    id = "fake"
    display_name = "Fake"
    description = "test"

    def requirements(self) -> DataRequirements:
        return DataRequirements()

    async def evaluate(self, context: SignalContext) -> ComponentSignal:
        return ComponentSignal(self.id, "neutral", 0.0, "neutral")


def test_registry_rejects_duplicate_component_id():
    registry = SignalComponentRegistry()
    registry.register(FakeComponent())
    with pytest.raises(ValueError, match="duplicate"):
        registry.register(FakeComponent())


def test_config_loads_plugin_factories_and_default_profile(tmp_path):
    cfg = load_config(write_config(tmp_path, factories=["tests.fake:factory"]))
    assert cfg.signal_plugins.factories == ["tests.fake:factory"]
    assert [item.component_id for item in cfg.signal_profile_defaults.components] == [
        "kronos",
        "llm_committee",
    ]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_signal_registry.py tests/test_config_loader.py --no-cov -q`

Expected: FAIL，缺少 Registry 和新配置字段。

- [ ] **Step 3: 实现协议和动态 factory 加载**

```python
@runtime_checkable
class SignalComponent(Protocol):
    id: str
    display_name: str
    description: str

    def requirements(self) -> DataRequirements:
        raise NotImplementedError

    async def evaluate(self, context: SignalContext) -> ComponentSignal:
        raise NotImplementedError


class SignalComponentRegistry:
    def register(self, component: SignalComponent) -> None:
        if component.id in self._components:
            raise ValueError(f"duplicate signal component id: {component.id}")
        self._components[component.id] = component

    def load_factory(self, path: str) -> None:
        module_name, attr_name = path.split(":", 1)
        factory = getattr(importlib.import_module(module_name), attr_name)
        component = factory()
        if not isinstance(component, SignalComponent):
            raise TypeError(f"factory {path} did not return SignalComponent")
        self.register(component)
```

- [ ] **Step 4: 增加 TOML 默认配置**

```toml
[signal_plugins]
factories = []

[signal_profile]
neutral_threshold = 0.20
max_target_ratio = 1.0
atr_stop_multiplier = 2.0
reward_ratio = 2.0
hitl_required = false

[[signal_profile.components]]
component_id = "kronos"
enabled = true
weight = 0.60

[[signal_profile.components]]
component_id = "llm_committee"
enabled = true
weight = 0.40
```

在 `config.py` 中增加 `SignalPluginsConfig`、`SignalProfileDefaultsConfig` 及构建函数。此任务暂不删除旧 `signal_engine`，因为入口尚未切换；Task 18 在所有入口完成替换后一次性删除。

- [ ] **Step 5: 运行配置与 Registry 测试**

Run: `uv run pytest tests/test_signal_registry.py tests/test_config_loader.py --no-cov -q`

Expected: PASS。

- [ ] **Step 6: 提交 Registry 和默认配置**

```bash
git add src/cryptotrader/signals/component.py src/cryptotrader/signals/registry.py src/cryptotrader/config.py config/default.toml tests/test_signal_registry.py tests/test_config_loader.py
git commit -m "feat: add configurable signal component registry"
```

---

### Task 3: 实现融合器、目标仓位映射和统一 ATR ExitPolicy

**Files:**
- Create: `src/cryptotrader/signals/fusion.py`
- Create: `src/cryptotrader/decision/engine.py`
- Create: `src/cryptotrader/decision/exit_policy.py`
- Test: `tests/test_signal_fusion.py`
- Test: `tests/test_decision_engine.py`
- Test: `tests/test_exit_policy.py`

**Interfaces:**
- Consumes: `ComponentSignal`、`ComponentWeight`、`SignalProfile`、`SignalContext`。
- Produces: `ComponentContribution`、`FusedSignal`、`WeightedSignalFusion.fuse()`。
- Produces: `DecisionEngine.target_for()`、`AtrExitPolicy.build_plan()`。

- [ ] **Step 1: 写融合数学失败测试**

```python
def test_weighted_fusion_preserves_configured_contributions():
    signals = (
        ComponentSignal("kronos", "long", 0.8, "k"),
        ComponentSignal("llm_committee", "short", 0.3, "l"),
    )
    weights = (
        ComponentWeight("kronos", True, 0.6),
        ComponentWeight("llm_committee", True, 0.4),
    )
    fused = WeightedSignalFusion().fuse(signals, weights)
    assert fused.score == pytest.approx(0.36)
    assert [c.weighted_score for c in fused.contributions] == pytest.approx([0.48, -0.12])


def test_fusion_rejects_missing_enabled_signal():
    with pytest.raises(ValueError, match="llm_committee"):
        WeightedSignalFusion().fuse(
            (ComponentSignal("kronos", "long", 0.8, "k"),),
            (ComponentWeight("kronos", True, 0.6), ComponentWeight("llm_committee", True, 0.4)),
        )
```

- [ ] **Step 2: 写目标仓位与 ExitPolicy 失败测试**

```python
def test_neutral_band_maps_to_flat():
    target = DecisionEngine().target_for(FusedSignal(score=0.2, contributions=(), reasoning=""), profile())
    assert target == TargetPosition("flat", 0.0)


def test_linear_long_mapping_after_neutral_band():
    target = DecisionEngine().target_for(FusedSignal(score=0.6, contributions=(), reasoning=""), profile())
    assert target.side == "long"
    assert target.size_ratio == pytest.approx(0.5)


def test_atr_exit_policy_builds_short_prices():
    plan = AtrExitPolicy().build_plan(
        context=context(price=100.0, atr=5.0),
        target=TargetPosition("short", 0.5),
        signals=(),
        fused=FusedSignal(-0.6, (), ""),
        profile=profile(atr_stop_multiplier=2.0, reward_ratio=3.0),
    )
    assert plan.stop_loss == pytest.approx(110.0)
    assert plan.take_profit == pytest.approx(70.0)
```

- [ ] **Step 3: 运行测试确认失败**

Run: `uv run pytest tests/test_signal_fusion.py tests/test_decision_engine.py tests/test_exit_policy.py --no-cov -q`

Expected: FAIL，缺少三个服务。

- [ ] **Step 4: 实现纯函数服务**

```python
class WeightedSignalFusion:
    def fuse(
        self,
        signals: tuple[ComponentSignal, ...],
        weights: tuple[ComponentWeight, ...],
    ) -> FusedSignal:
        signal_by_id = {item.component_id: item for item in signals}
        contributions = []
        for configured in (item for item in weights if item.enabled):
            signal = signal_by_id.get(configured.component_id)
            if signal is None:
                raise ValueError(f"missing signal for enabled component {configured.component_id}")
            signed = {"long": signal.confidence, "short": -signal.confidence, "neutral": 0.0}[signal.direction]
            contributions.append(
                ComponentContribution(configured.component_id, configured.weight, signed, configured.weight * signed)
            )
        score = sum(item.weighted_score for item in contributions)
        reasoning = " | ".join(
            f"{item.component_id}: {item.weighted_score:+.4f}" for item in contributions
        )
        return FusedSignal(score=score, contributions=tuple(contributions), reasoning=reasoning)


class DecisionEngine:
    def target_for(self, fused: FusedSignal, profile: SignalProfile) -> TargetPosition:
        magnitude = abs(fused.score)
        if magnitude <= profile.neutral_threshold:
            return TargetPosition("flat", 0.0)
        ratio = (magnitude - profile.neutral_threshold) / (1.0 - profile.neutral_threshold)
        ratio *= profile.max_target_ratio
        return TargetPosition("long" if fused.score > 0 else "short", ratio)
```

ATR 使用 `SignalContext.atr`，flat 计划的 SL/TP 为 `None`。

- [ ] **Step 5: 运行纯服务测试**

Run: `uv run pytest tests/test_signal_fusion.py tests/test_decision_engine.py tests/test_exit_policy.py --no-cov -q`

Expected: PASS。

- [ ] **Step 6: 提交融合与决策服务**

```bash
git add src/cryptotrader/signals/fusion.py src/cryptotrader/decision tests/test_signal_fusion.py tests/test_decision_engine.py tests/test_exit_policy.py
git commit -m "feat: add deterministic signal fusion and target decisions"
```

---

### Task 4: 实现目标仓位差值 ExecutionPlanner

**Files:**
- Create: `src/cryptotrader/execution/planner.py`
- Modify: `src/cryptotrader/decision/models.py`
- Test: `tests/test_execution_planner.py`

**Interfaces:**
- Consumes: `SignalContext`、`TradePlan`、`risk.position.max_single_pct`。
- Produces: `OrderIntent`、`ExecutionPlan`、`ExecutionPlanner.plan()`。

- [ ] **Step 1: 写开仓、减仓、平仓和反向失败测试**

```python
@pytest.mark.parametrize(
    ("current", "target", "expected"),
    [
        (position("flat", 0.0, 0.0), TargetPosition("long", 0.5), [("buy", 5.0, False)]),
        (position("long", 8.0, 0.8), TargetPosition("long", 0.3), [("sell", 5.0, True)]),
        (position("long", 8.0, 0.8), TargetPosition("flat", 0.0), [("sell", 8.0, True)]),
        (
            position("short", 4.0, 0.4),
            TargetPosition("long", 0.3),
            [("buy", 4.0, True), ("buy", 3.0, False)],
        ),
    ],
)
def test_planner_creates_position_delta(current, target, expected):
    result = ExecutionPlanner(max_single_pct=0.1).plan(
        context=context(price=100.0, equity=10_000.0, position=current, market_type="swap"),
        trade_plan=trade_plan(target),
    )
    assert [(x.side, x.amount, x.reduce_only) for x in result.intents] == expected


def test_spot_short_is_rejected():
    with pytest.raises(ExecutionPlanningError, match="spot"):
        ExecutionPlanner(max_single_pct=0.1).plan(
            context=context(price=100.0, equity=10_000.0, position=position("flat", 0, 0), market_type="spot"),
            trade_plan=trade_plan(TargetPosition("short", 0.5)),
        )
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_execution_planner.py --no-cov -q`

Expected: FAIL，缺少 planner。

- [ ] **Step 3: 实现差值算法**

```python
target_amount = (
    context.equity
    * self.max_single_pct
    * trade_plan.target.size_ratio
    / context.current_price
)
current_signed = context.current_position.signed_amount
target_signed = {
    "long": target_amount,
    "short": -target_amount,
    "flat": 0.0,
}[trade_plan.target.side]

if current_signed * target_signed < 0:
    close = OrderIntent(
        pair=context.pair.canonical(),
        side="sell" if current_signed > 0 else "buy",
        amount=abs(current_signed),
        reduce_only=True,
    )
    enter = OrderIntent(
        pair=context.pair.canonical(),
        side="buy" if target_signed > 0 else "sell",
        amount=abs(target_signed),
        reduce_only=False,
    )
    return ExecutionPlan((close, enter), trade_plan.stop_loss, trade_plan.take_profit)

delta = target_signed - current_signed
```

同方向 delta 的符号决定 buy/sell，`abs(target_signed) < abs(current_signed)` 时 `reduce_only=True`。`abs(delta) < 1e-12` 返回空 intents。

- [ ] **Step 4: 运行 planner 测试**

Run: `uv run pytest tests/test_execution_planner.py --no-cov -q`

Expected: PASS。

- [ ] **Step 5: 提交 ExecutionPlanner**

```bash
git add src/cryptotrader/execution/planner.py src/cryptotrader/decision/models.py tests/test_execution_planner.py
git commit -m "feat: plan orders from target position deltas"
```

---

### Task 5: 把硬风控改为 TargetPosition 模型

**Files:**
- Create: `src/cryptotrader/risk/models.py`
- Rewrite: `src/cryptotrader/risk/gate.py`
- Modify: `src/cryptotrader/risk/checks/available_margin.py`
- Modify: `src/cryptotrader/risk/checks/concentration.py`
- Modify: `src/cryptotrader/risk/checks/cooldown.py`
- Modify: `src/cryptotrader/risk/checks/correlation.py`
- Modify: `src/cryptotrader/risk/checks/cvar.py`
- Modify: `src/cryptotrader/risk/checks/exchange.py`
- Modify: `src/cryptotrader/risk/checks/loss.py`
- Modify: `src/cryptotrader/risk/checks/position.py`
- Modify: `src/cryptotrader/risk/checks/rate_limit.py`
- Modify: `src/cryptotrader/risk/checks/token_security.py`
- Modify: `src/cryptotrader/risk/checks/volatility.py`
- Modify: `src/cryptotrader/models.py`
- Modify: `tests/test_risk_checks.py`
- Modify: `tests/test_risk_helpers.py`
- Modify: `tests/test_risk_gate_isolation.py`
- Test: `tests/test_target_risk_gate.py`

**Interfaces:**
- Consumes: `TradePlan` 和 `SignalContext.current_position`。
- Produces: `RiskRequest`、`RiskCheckResult(size_ratio_cap)`、`RiskDecision`。
- Produces: `RiskGate.check(request, portfolio) -> RiskDecision`。

- [ ] **Step 1: 写风险降低和仓位 clamp 失败测试**

```python
async def test_flat_target_bypasses_unavailable_redis(redis_down, risk_config):
    gate = RiskGate(risk_config, redis_down)
    request = risk_request(current=position("long", 1.0, 0.5), target=TargetPosition("flat", 0.0))
    result = await gate.check(request, portfolio())
    assert result.passed is True
    assert result.plan.target.side == "flat"


async def test_gate_applies_strictest_size_ratio_cap(fake_check, risk_config, redis_ok):
    gate = RiskGate(risk_config, redis_ok, checks=[cap_check(0.4), cap_check(0.25)])
    result = await gate.check(
        risk_request(current=position("flat", 0, 0), target=TargetPosition("long", 0.8)),
        portfolio(),
    )
    assert result.plan.target == TargetPosition("long", 0.25)
```

- [ ] **Step 2: 运行风险测试确认失败**

Run: `uv run pytest tests/test_target_risk_gate.py tests/test_risk_checks.py tests/test_risk_helpers.py tests/test_risk_gate_isolation.py --no-cov -q`

Expected: FAIL，风险层仍依赖 `TradeVerdict`。

- [ ] **Step 3: 定义新风险模型**

```python
@dataclass(frozen=True)
class RiskRequest:
    context: SignalContext
    plan: TradePlan

    @property
    def target(self) -> TargetPosition:
        return self.plan.target

    @property
    def reduces_exposure(self) -> bool:
        current = self.context.current_position.signed_ratio
        target = self.target.signed_ratio
        return target == 0.0 or (current * target > 0 and abs(target) <= abs(current))


@dataclass(frozen=True)
class RiskCheckResult:
    passed: bool
    reason: str = ""
    size_ratio_cap: float | None = None


@dataclass(frozen=True)
class RiskDecision:
    passed: bool
    plan: TradePlan
    rejected_by: str = ""
    reason: str = ""
```

- [ ] **Step 4: 重写全部 RiskCheck 输入语义**

每个检查的签名统一为：

```python
async def evaluate(self, request: RiskRequest, portfolio: dict) -> RiskCheckResult:
```

方向判断使用 `request.target.side`，仓位比例使用 `request.target.size_ratio`，flat 使用 `request.target.side == "flat"`。删除所有 `TradeVerdict` import、`verdict.action` 和 `verdict.position_scale`。`RiskGate` 聚合 `size_ratio_cap`，用 `dataclasses.replace()` 生成新的 `TradePlan`，不修改输入对象。

- [ ] **Step 5: 运行完整风险测试**

Run: `uv run pytest tests/test_target_risk_gate.py tests/test_risk_checks.py tests/test_risk_helpers.py tests/test_risk_gate_isolation.py --no-cov -q`

Expected: PASS。

- [ ] **Step 6: 提交风控重构**

```bash
git add src/cryptotrader/risk src/cryptotrader/models.py tests/test_target_risk_gate.py tests/test_risk_checks.py tests/test_risk_helpers.py tests/test_risk_gate_isolation.py
git commit -m "refactor: evaluate risk against target positions"
```

---

### Task 6: 构建 point-in-time 多周期 SignalContext

**Files:**
- Create: `src/cryptotrader/signals/context.py`
- Modify: `src/cryptotrader/data/snapshot.py`
- Modify: `src/cryptotrader/data/market.py`
- Test: `tests/test_signal_context_provider.py`

**Interfaces:**
- Consumes: `CycleRequest`、`DataRequirements`、现有 `SnapshotAggregator`、Portfolio reader。
- Produces: `SignalContextProvider.collect()`、`SignalContextProvider.refresh_execution_state()`、`LiveSignalContextProvider`、`HistoricalSignalContextProvider`。

- [ ] **Step 1: 写多周期与统一 as_of 失败测试**

```python
async def test_live_provider_materializes_each_required_timeframe_once():
    aggregator = FakeSnapshotAggregator()
    market = FakeMarketCollector()
    provider = LiveSignalContextProvider(aggregator, market, FakePortfolioReader())
    context = await provider.collect(
        CycleRequest(pair=Pair.parse("BTC/USDT:USDT"), mode="paper"),
        DataRequirements(
            candles=(CandleRequirement("1h", 100), CandleRequirement("4h", 512)),
            onchain=True,
            news=True,
            macro=True,
        ),
    )
    assert set(context.snapshots) == {"1h", "4h"}
    assert {snapshot.timestamp for snapshot in context.snapshots.values()} == {context.as_of}
    assert market.calls == [("BTC/USDT:USDT", "4h", 512)]


async def test_historical_provider_never_reads_live_market():
    provider = HistoricalSignalContextProvider(history=history_fixture())
    context = await provider.collect(request_at("2025-01-02T00:00:00Z"), requirements_4h())
    assert context.as_of.isoformat() == "2025-01-02T00:00:00+00:00"
    assert context.snapshots["4h"].market.ohlcv.index.max() <= context.as_of


async def test_refresh_execution_state_only_updates_price_equity_and_position():
    provider = LiveSignalContextProvider(FakeSnapshotAggregator(), FakeMarketCollector(), FakePortfolioReader())
    original = await provider.collect(request(), requirements_4h())
    provider.portfolio.current_position = position("long", 0.2, 0.2)
    refreshed = await provider.refresh_execution_state(original)
    assert refreshed.snapshots is original.snapshots
    assert refreshed.as_of == original.as_of
    assert refreshed.current_position == position("long", 0.2, 0.2)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_signal_context_provider.py --no-cov -q`

Expected: FAIL，ContextProvider 不存在。

- [ ] **Step 3: 实现 Live provider**

```python
primary = requirements.candles[0]
base = await self.aggregator.collect(
    pair=request.pair.canonical(),
    exchange_id=request.exchange_id,
    timeframe=primary.timeframe,
    limit=primary.limit,
    backtest_mode=False,
)
as_of = base.timestamp
snapshots = {primary.timeframe: replace(base, timestamp=as_of)}
for requirement in requirements.candles[1:]:
    market = await self.market.collect(
        request.pair.canonical(),
        request.exchange_id,
        requirement.timeframe,
        requirement.limit,
    )
    snapshots[requirement.timeframe] = replace(base, timestamp=as_of, market=market)
```

从 Portfolio reader 得到 equity 和当前仓位，计算 `current_position.size_ratio`。ATR 从 `config.data.default_timeframe` 的最近 14 根已闭合 K 线计算；TradingCycle 在合并数据需求时保证该时间周期存在。

`HistoricalSignalContextProvider` 只使用传入历史窗口，拒绝超过 `request.as_of` 的 bar。

`refresh_execution_state(context)` 只刷新 `current_price`、`equity` 和 `current_position`，必须保留原始 `as_of`、ATR 和 snapshots 对象。它只用于 HITL 等待后按最新仓位重新计算订单差值，不重跑组件、不重新融合，也不切换 Profile revision；Historical provider 返回原 context。

- [ ] **Step 4: 运行 ContextProvider 测试**

Run: `uv run pytest tests/test_signal_context_provider.py tests/test_data_collectors.py --no-cov -q`

Expected: PASS。

- [ ] **Step 5: 提交上下文物化**

```bash
git add src/cryptotrader/signals/context.py src/cryptotrader/data/snapshot.py src/cryptotrader/data/market.py tests/test_signal_context_provider.py tests/test_data_collectors.py
git commit -m "feat: materialize point-in-time multi-timeframe contexts"
```

---

### Task 7: 建立业务事件和全成功 ComponentRunner

**Files:**
- Create: `src/cryptotrader/cycle_events.py`
- Create: `src/cryptotrader/signals/runner.py`
- Test: `tests/test_component_runner.py`
- Test: `tests/test_cycle_events.py`

**Interfaces:**
- Consumes: Registry 中已启用 `SignalComponent` 和统一 `SignalContext`。
- Produces: `CycleEvent`、`CycleEventSink`、`NullCycleEventSink`、`ComponentRunError`、`ComponentRunner.run()`。

- [ ] **Step 1: 写并发成功和任一失败测试**

```python
async def test_runner_returns_signals_in_configured_order():
    sink = RecordingSink()
    result = await ComponentRunner(sink).run(
        (component("kronos", delay=0.02), component("llm_committee", delay=0.0)),
        context(),
    )
    assert [signal.component_id for signal in result] == ["kronos", "llm_committee"]
    assert [event.name for event in sink.events].count("component_completed") == 2


async def test_runner_raises_one_error_containing_all_component_failures():
    with pytest.raises(ComponentRunError) as caught:
        await ComponentRunner(RecordingSink()).run(
            (failing_component("kronos"), failing_component("llm_committee")),
            context(),
        )
    assert set(caught.value.errors) == {"kronos", "llm_committee"}
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_component_runner.py tests/test_cycle_events.py --no-cov -q`

Expected: FAIL，缺少 runner/event types。

- [ ] **Step 3: 实现事件和严格 runner**

```python
results = await asyncio.gather(
    *(self._run_one(component, context) for component in components),
    return_exceptions=True,
)
errors = {
    component.id: result
    for component, result in zip(components, results, strict=True)
    if isinstance(result, BaseException)
}
if errors:
    raise ComponentRunError(errors)
return tuple(cast(ComponentSignal, item) for item in results)
```

`_run_one()` 发布 `component_started` 和 `component_completed`；异常发布 `component_failed` 后重新抛出。`CancelledError` 原样传播，不包装为组件失败。

- [ ] **Step 4: 运行 runner/event 测试**

Run: `uv run pytest tests/test_component_runner.py tests/test_cycle_events.py --no-cov -q`

Expected: PASS。

- [ ] **Step 5: 提交 runner 和事件协议**

```bash
git add src/cryptotrader/cycle_events.py src/cryptotrader/signals/runner.py tests/test_component_runner.py tests/test_cycle_events.py
git commit -m "feat: run all signal components with typed events"
```

---

### Task 8: 把 Kronos 重写为纯信号组件

**Files:**
- Create: `src/cryptotrader/signals/components/__init__.py`
- Create: `src/cryptotrader/signals/components/kronos.py`
- Modify: `src/cryptotrader/agents/_kronos_features.py`
- Replace tests: `tests/test_kronos_node.py` → `tests/test_kronos_component.py`
- Modify: `tests/test_kronos_aux.py`

**Interfaces:**
- Consumes: `KronosConfig`、`SignalContext.snapshots[timeframe]`、现有 gate/predictor/feature 算法。
- Produces: `KronosComponent.requirements()` 和 `KronosComponent.evaluate()`。

- [ ] **Step 1: 写 Kronos 组件失败测试**

```python
async def test_gate_rejection_is_valid_neutral_signal():
    component = kronos_component(gate_probability=0.49)
    signal = await component.evaluate(kronos_context())
    assert signal.direction == "neutral"
    assert signal.confidence == 0.0
    assert signal.details["gate_proba"] == pytest.approx(0.49)


async def test_predictor_failure_raises_component_error():
    component = kronos_component(predictor_error=RuntimeError("model down"))
    with pytest.raises(ComponentExecutionError, match="prediction"):
        await component.evaluate(kronos_context())


async def test_predictor_runs_through_to_thread(monkeypatch):
    called = False

    async def fake_to_thread(func, *args, **kwargs):
        nonlocal called
        called = True
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)
    await kronos_component().evaluate(kronos_context())
    assert called is True
```

- [ ] **Step 2: 运行测试确认旧 node 语义失败**

Run: `uv run pytest tests/test_kronos_component.py tests/test_kronos_aux.py --no-cov -q`

Expected: FAIL，`KronosComponent` 不存在。

- [ ] **Step 3: 提取 Kronos 算法并删除 Verdict 生成**

`KronosComponent.requirements()` 返回：

```python
return DataRequirements(
    candles=(CandleRequirement(self.config.timeframe, self.config.ohlcv_limit),),
    onchain=True,
    macro=True,
    kronos_aux=True,
)
```

gate 低于 0.5 或 Step2 弱空信号返回合法 neutral。缺失数据、gate 加载、特征计算、分类和预测异常统一抛出 `ComponentExecutionError`。保留当前五维 confidence 算法，输出：

```python
return ComponentSignal(
    component_id=self.id,
    direction="long" if raw_signal > 0 else "short",
    confidence=confidence,
    reasoning=reasoning,
    details={
        "raw_signal": raw_signal,
        "gate_proba": gate_proba,
        "h10_20": h10_20,
        "h30_50": h30_50,
    },
)
```

删除 Kronos 组件内的 position scale、risk multiple、SL 和 TP 计算。`predictor.predict` 必须通过 `await asyncio.to_thread(...)` 调用。

- [ ] **Step 4: 运行 Kronos 测试**

Run: `uv run pytest tests/test_kronos_component.py tests/test_kronos_aux.py --no-cov -q`

Expected: PASS。

- [ ] **Step 5: 提交 Kronos 组件**

```bash
git add src/cryptotrader/signals/components src/cryptotrader/agents/_kronos_features.py tests/test_kronos_component.py tests/test_kronos_aux.py
git rm tests/test_kronos_node.py
git commit -m "refactor: make Kronos a pure signal component"
```

---

### Task 9: 把四智能体辩论重写为 LLMCommitteeComponent

**Files:**
- Create: `src/cryptotrader/signals/components/llm_committee.py`
- Modify: `src/cryptotrader/debate/challenge.py`
- Modify: `src/cryptotrader/debate/convergence.py`
- Test: `tests/test_llm_committee_component.py`
- Replace: `tests/test_debate_parallel.py`
- Modify: `tests/test_debate_anti_ratchet.py`
- Modify: `tests/test_debate_turn_capture.py`

**Interfaces:**
- Consumes: 现有 Agent classes、PromptBuilder、debate challenge/convergence、CycleEventSink。
- Produces: `CommitteeState`、内部 LangGraph、`LLMCommitteeComponent.evaluate()`。

- [ ] **Step 1: 写委员会成功、辩论和严格失败测试**

```python
async def test_committee_runs_four_agents_and_debate_before_summary():
    agents = fake_agents(
        tech=("bullish", 0.8),
        chain=("bearish", 0.7),
        news=("neutral", 0.2),
        macro=("bullish", 0.6),
    )
    sink = RecordingSink()
    component = LLMCommitteeComponent(config(), agents=agents, summary=summary("long", 0.65), sink=sink)
    signal = await component.evaluate(llm_context())
    assert signal == ComponentSignal("llm_committee", "long", 0.65, "committee summary", signal.details)
    assert any(event.name == "debate_round_completed" for event in sink.events)


async def test_one_agent_failure_fails_whole_component():
    agents = fake_agents(tech=RuntimeError("timeout"))
    component = LLMCommitteeComponent(config(), agents=agents, summary=summary("neutral", 0.0))
    with pytest.raises(ComponentExecutionError, match="tech_agent"):
        await component.evaluate(llm_context())


async def test_summary_returns_only_component_signal_fields():
    payload = await run_summary_payload()
    assert set(payload) == {"direction", "confidence", "reasoning"}
```

- [ ] **Step 2: 运行委员会测试确认失败**

Run: `uv run pytest tests/test_llm_committee_component.py tests/test_debate_parallel.py tests/test_debate_anti_ratchet.py tests/test_debate_turn_capture.py --no-cov -q`

Expected: FAIL，委员会仍依赖 `ArenaState` 和 mock 降级。

- [ ] **Step 3: 定义内部 CommitteeState 和 LangGraph**

```python
class CommitteeState(TypedDict):
    context: SignalContext
    analyses: dict[str, AgentAnalysis]
    debate_round: int
    debate_turns: list[dict[str, Any]]
    divergence_scores: list[float]
    debate_skipped: bool
    debate_skip_reason: str
    final_signal: ComponentSignal | None
```

内部图固定为：

```text
START → analyze_all → debate_gate
                    ├─ skip → summarize → END
                    └─ debate_round → convergence
                                      ├─ continue → debate_round
                                      └─ converged → summarize → END
```

`analyze_all` 用 `asyncio.gather()` 运行四个 Agent；任何异常抛出 `ComponentExecutionError`，不创建 `is_mock`。辩论单个回合异常同样向上抛出。每个 Agent 完成和每轮辩论通过 `CycleEventSink` 发布业务事件。

- [ ] **Step 4: 实现纯信号委员会汇总 prompt**

最终 LLM schema 固定为：

```json
{
  "direction": "long|short|neutral",
  "confidence": 0.0,
  "reasoning": "基于四个领域观点和辩论证据的市场判断"
}
```

prompt 明确禁止输出仓位、动作、止盈止损和订单建议。解析后构造 `ComponentSignal`，并把 `analyses`、`debate_turns`、`consensus_metrics` 放入 `details` 供 Journal 和 Debate 页面使用。

- [ ] **Step 5: 运行委员会与辩论测试**

Run: `uv run pytest tests/test_llm_committee_component.py tests/test_debate_parallel.py tests/test_debate_anti_ratchet.py tests/test_debate_turn_capture.py --no-cov -q`

Expected: PASS。

- [ ] **Step 6: 提交 LLM 委员会**

```bash
git add src/cryptotrader/signals/components/llm_committee.py src/cryptotrader/debate tests/test_llm_committee_component.py tests/test_debate_parallel.py tests/test_debate_anti_ratchet.py tests/test_debate_turn_capture.py
git commit -m "refactor: isolate the four-agent debate component"
```

---

### Task 10: 持久化全局 SignalProfile 并提供 GET/PUT API

**Files:**
- Create: `src/cryptotrader/profiles/repository.py`
- Create: `src/api/routes/signal_profile.py`
- Modify: `src/api/routes/__init__.py`
- Modify: `src/api/main.py`
- Test: `tests/test_signal_profile_repository.py`
- Test: `tests/test_signal_profile_api.py`

**Interfaces:**
- Consumes: Task 1/2 的 `SignalProfile`、defaults、Registry metadata。
- Produces: `SignalProfileRepository.ensure_table()`、`get()`、`get_or_create()`、`replace()`。
- Produces: `GET /api/signal-profile`、`PUT /api/signal-profile`。

- [ ] **Step 1: 写 repository revision 失败测试**

```python
async def test_repository_creates_default_and_increments_revision(sqlite_url):
    repo = SignalProfileRepository(sqlite_url)
    first = await repo.get_or_create(default_profile())
    assert first.revision == 1
    second = await repo.replace(replace(first, neutral_threshold=0.3))
    assert second.revision == 2
    assert (await repo.get()).neutral_threshold == 0.3
```

- [ ] **Step 2: 写 API 失败测试**

```python
async def test_put_profile_rejects_weights_not_equal_to_one(api_client):
    response = await api_client.put(
        "/api/signal-profile",
        json=profile_payload(kronos=0.6, llm=0.3),
    )
    assert response.status_code == 422
    assert "1.0" in response.json()["detail"]


async def test_get_profile_returns_registry_metadata(api_client):
    response = await api_client.get("/api/signal-profile")
    body = response.json()
    assert body["revision"] == 1
    assert {item["component_id"] for item in body["installed_components"]} == {
        "kronos",
        "llm_committee",
    }
```

- [ ] **Step 3: 运行 repository/API 测试确认失败**

Run: `uv run pytest tests/test_signal_profile_repository.py tests/test_signal_profile_api.py --no-cov -q`

Expected: FAIL，表和路由不存在。

- [ ] **Step 4: 实现单行表和完整替换**

```python
class SignalProfileRow(Base):
    __tablename__ = "signal_profile"
    id = mapped_column(String(20), primary_key=True)
    revision = mapped_column(BigInteger, nullable=False)
    config = mapped_column(JSON().with_variant(JSONB, "postgresql"), nullable=False)
    updated_at = mapped_column(DateTime(timezone=True), nullable=False)
```

唯一 ID 固定为 `global`。`replace()` 在事务内读取当前 revision，加一后覆盖 JSON。Pydantic request/response model 与领域模型显式转换；API 不直接返回 SQLAlchemy row。

- [ ] **Step 5: 在 FastAPI lifespan 初始化 Registry/Profile**

启动流程必须：注册 Kronos、LLM 内置组件 → 加载 TOML factories → 校验默认 Profile → 建表并 seed → 保存 Registry 和 repository 到 `app.state`。非法 factory 或默认权重让应用启动失败。

- [ ] **Step 6: 运行 repository/API 测试**

Run: `uv run pytest tests/test_signal_profile_repository.py tests/test_signal_profile_api.py tests/test_api_security_hardening.py --no-cov -q`

Expected: PASS。

- [ ] **Step 7: 提交动态 Profile 后端**

```bash
git add src/cryptotrader/profiles/repository.py src/api/routes/signal_profile.py src/api/routes/__init__.py src/api/main.py tests/test_signal_profile_repository.py tests/test_signal_profile_api.py
git commit -m "feat: persist and expose the global signal profile"
```

---

### Task 11: 用 trading_cycles Journal 替换 DecisionCommit

**Files:**
- Create: `src/cryptotrader/journal/models.py`
- Rewrite: `src/cryptotrader/journal/store.py`
- Delete: `src/cryptotrader/journal/commit.py`
- Modify: `src/cryptotrader/journal/__init__.py`
- Test: `tests/test_cycle_journal_store.py`
- Delete/Replace: `tests/test_journal_db.py`
- Delete/Replace: `tests/test_journal_sl_tp_fields.py`
- Delete/Replace: `tests/test_observability_models.py`
- Delete/Replace: `tests/test_observability_migration.py`

**Interfaces:**
- Consumes: `CycleOutcome`、Profile revision、SignalContext summary 和各阶段结果。
- Produces: `TradingCycleRecord`、`CycleJournalStore.append/list/get`。

- [ ] **Step 1: 写成功和失败周期 round-trip 测试**

```python
async def test_cycle_record_round_trips_component_contributions(sqlite_url):
    store = CycleJournalStore(sqlite_url)
    record = cycle_record(
        status="completed",
        profile_revision=3,
        component_signals=[{"component_id": "kronos", "direction": "long", "confidence": 0.8}],
        fused_signal={"score": 0.48, "contributions": [{"component_id": "kronos", "weighted_score": 0.48}]},
    )
    await store.append(record)
    loaded = await store.get(record.cycle_id)
    assert loaded == record


async def test_component_failure_is_journaled_without_trade_plan(sqlite_url):
    store = CycleJournalStore(sqlite_url)
    record = cycle_record(status="component_failed", component_error={"llm_committee": "timeout"})
    await store.append(record)
    assert (await store.get(record.cycle_id)).trade_plan is None
```

- [ ] **Step 2: 运行 Journal 测试确认失败**

Run: `uv run pytest tests/test_cycle_journal_store.py --no-cov -q`

Expected: FAIL，CycleJournalStore 不存在。

- [ ] **Step 3: 实现新记录和新表**

```python
@dataclass(frozen=True)
class TradingCycleRecord:
    cycle_id: str
    created_at: datetime
    pair: str
    status: CycleStatus
    profile_revision: int
    context_summary: Mapping[str, Any]
    component_signals: tuple[Mapping[str, Any], ...]
    component_error: Mapping[str, str] | None
    fused_signal: Mapping[str, Any] | None
    target_position: Mapping[str, Any] | None
    trade_plan: Mapping[str, Any] | None
    hitl_result: Mapping[str, Any] | None
    risk_result: Mapping[str, Any] | None
    execution_result: Mapping[str, Any] | None


class TradingCycleRow(Base):
    __tablename__ = "trading_cycles"
    cycle_id = mapped_column(String(36), primary_key=True)
    created_at = mapped_column(DateTime(timezone=True), index=True)
    pair = mapped_column(String(50), index=True)
    status = mapped_column(String(32), index=True)
    profile_revision = mapped_column(BigInteger, nullable=False)
    payload = mapped_column(JSON().with_variant(JSONB, "postgresql"), nullable=False)
```

旧 `decision_commits` 表不删除、不读取、不写入。新 store 可以在 `database_url=None` 时使用实例级内存列表，供纯单元测试使用；这不是旧 schema 兼容。

- [ ] **Step 4: 删除旧 Journal 测试并完成新 store 测试**

Run: `uv run pytest tests/test_cycle_journal_store.py --no-cov -q`

Expected: PASS。

- [ ] **Step 5: 提交新 Journal**

```bash
git add src/cryptotrader/journal tests/test_cycle_journal_store.py
git rm tests/test_journal_db.py tests/test_journal_sl_tp_fields.py tests/test_observability_models.py tests/test_observability_migration.py
git commit -m "refactor: replace decision commits with cycle journal"
```

---

### Task 12: 实现 TradingCycle、HITL、风险和执行全主链

**Files:**
- Create: `src/cryptotrader/trading_cycle.py`
- Create: `src/cryptotrader/execution/service.py`
- Rewrite: `src/cryptotrader/hitl/gate.py`
- Rewrite: `src/cryptotrader/hitl/store.py`
- Rewrite: `src/api/routes/hitl.py`
- Modify: `src/cryptotrader/execution/order.py`
- Modify: `src/cryptotrader/execution/exchange.py`
- Test: `tests/test_trading_cycle.py`
- Replace: `tests/test_hitl_gate.py`
- Replace: `tests/test_hitl_api.py`
- Modify: `tests/test_exchange_algo_oco.py`

**Interfaces:**
- Consumes: ProfileRepository、Registry、ContextProvider、Runner、Fusion、Decision、ExitPolicy、RiskGate、ExecutionPlanner、Executor、Journal、EventSink。
- Produces: `TradingCycle.run(request)`、`TradingCycle.resume_approved(approval_id)`。

- [ ] **Step 1: 写主链成功和组件失败测试**

```python
async def test_cycle_runs_components_fusion_risk_execution_and_journal():
    cycle = build_test_cycle(
        signals=(signal("kronos", "long", 0.8), signal("llm_committee", "long", 0.6)),
        profile=profile(kronos=0.6, llm=0.4, hitl=False),
    )
    outcome = await cycle.run(CycleRequest(Pair.parse("BTC/USDT:USDT"), "paper"))
    assert outcome.status == "completed"
    assert outcome.trade_plan.target.side == "long"
    assert cycle.executor.executed[0].intents
    assert cycle.journal.records[0].profile_revision == outcome.profile_revision


async def test_component_failure_skips_fusion_and_execution_but_journals_cycle():
    cycle = build_test_cycle(component_error={"llm_committee": RuntimeError("timeout")})
    outcome = await cycle.run(request())
    assert outcome.status == "component_failed"
    assert cycle.executor.executed == []
    assert cycle.journal.records[0].component_error == {"llm_committee": "RuntimeError: timeout"}
```

- [ ] **Step 2: 写 HITL 暂停、批准和拒绝测试**

```python
async def test_hitl_stores_target_plan_and_approval_replans_from_current_position():
    cycle = build_test_cycle(profile=profile(hitl=True))
    pending = await cycle.run(request())
    assert pending.status == "awaiting_approval"
    cycle.context_provider.current_position = position("long", 0.2, 0.2)
    approved = await cycle.resume_approved(pending.approval_id)
    assert approved.status == "completed"
    assert cycle.execution_planner.last_context.current_position.size_ratio == 0.2


async def test_rejected_approval_never_reaches_risk_or_execution():
    cycle = build_test_cycle(profile=profile(hitl=True))
    pending = await cycle.run(request())
    rejected = await cycle.reject_approval(pending.approval_id, decision_by="web")
    assert rejected.status == "approval_rejected"
    assert cycle.risk.calls == []
    assert cycle.executor.executed == []
```

- [ ] **Step 3: 运行主链/HITL 测试确认失败**

Run: `uv run pytest tests/test_trading_cycle.py tests/test_hitl_gate.py tests/test_hitl_api.py --no-cov -q`

Expected: FAIL，TradingCycle 与新审批模型不存在。

- [ ] **Step 4: 实现 TradingCycle 严格阶段流**

```python
async def run(self, request: CycleRequest) -> CycleOutcome:
    cycle_id = str(uuid4())
    profile = await self.profiles.get()
    validate_signal_profile(profile, self.registry.ids())
    components = self.registry.enabled(profile)
    requirements = DataRequirements.merge(*(item.requirements() for item in components), self.exit_requirement)
    context = await self.contexts.collect(request, requirements)
    try:
        signals = await self.runner.run(components, context)
    except ComponentRunError as exc:
        return await self._finish_component_failure(cycle_id, request, profile, context, exc)
    fused = self.fusion.fuse(signals, profile.components)
    target = self.decisions.target_for(fused, profile)
    plan = self.exits.build_plan(context, target, signals, fused, profile)
    if self._target_equals_current(plan.target, context.current_position):
        return await self._finish_no_change(cycle_id, profile, context, signals, fused, plan)
    if profile.hitl_required:
        approval_id = await self.approvals.create(cycle_id, request, profile, context, plan)
        return await self._finish_pending(cycle_id, approval_id, profile, context, signals, fused, plan)
    return await self._risk_plan_execute(cycle_id, profile, context, signals, fused, plan)
```

每个 return 路径在返回前写 `TradingCycleRecord` 并发布完成/失败事件。`CancelledError` 写 cancelled record 后重新抛出。

- [ ] **Step 5: 重写 ApprovalStore 和 HITL API**

新表 `trade_plan_approvals` 保存 `approval_id`、`cycle_id`、`pair`、`profile_revision`、`request JSON`、`trade_plan JSON`、`status`、`decision_by`、时间字段。删除 verdict/agent snapshot 字段。批准 API 调用 `TradingCycle.resume_approved()`；拒绝 API 调用 `reject_approval()`。

`resume_approved()` 原子地把审批从 pending 改为 approved，读取审批中冻结的 `TradePlan` 和 revision，再调用 `contexts.refresh_execution_state(stored_context)` 只刷新价格、权益和当前仓位，然后执行 `_risk_plan_execute()`。禁止重新读取活动 Profile、重跑组件、重新融合或重新生成退出价；这样网页配置只影响下一轮新周期，同时批准时的订单差值基于最新实际仓位。

- [ ] **Step 6: 实现 ExecutionService**

`ExecutionService.execute(ExecutionPlan, context)` 顺序执行 intents；反向计划第一笔未成交时不得发送第二笔。全部成交后为最终非 flat 仓位创建统一 OCO，flat 则取消已有保护单。返回结构化 `ExecutionResult`，订单失败不得修改 RiskDecision。

- [ ] **Step 7: 运行主链、HITL、执行保护测试**

Run: `uv run pytest tests/test_trading_cycle.py tests/test_hitl_gate.py tests/test_hitl_api.py tests/test_exchange_algo_oco.py tests/test_order_manager.py --no-cov -q`

Expected: PASS。

- [ ] **Step 8: 提交完整主链**

```bash
git add src/cryptotrader/trading_cycle.py src/cryptotrader/execution src/cryptotrader/hitl src/api/routes/hitl.py tests/test_trading_cycle.py tests/test_hitl_gate.py tests/test_hitl_api.py tests/test_exchange_algo_oco.py tests/test_order_manager.py
git commit -m "feat: orchestrate target-position trading cycles"
```

---

### Task 13: 组装生产依赖并切换 Scheduler、CLI 和 Chat

**Files:**
- Create: `src/cryptotrader/bootstrap.py`
- Rewrite: `src/cryptotrader/chat/analysis_runner.py`
- Modify: `src/cryptotrader/chat/event_bus.py`
- Modify: `src/cryptotrader/chat/task_manager.py`
- Delete: `src/cryptotrader/chat/partial_verdict.py`
- Rewrite: `src/cryptotrader/scheduler.py`
- Rewrite: `src/cli/main.py`
- Modify: `src/api/routes/chat.py`
- Modify: `src/api/routes/chat_control.py`
- Modify: `src/cryptotrader/tracing.py`
- Test: `tests/test_bootstrap.py`
- Modify: `tests/test_scheduler.py`
- Modify: `tests/test_cli_agent_list.py`
- Modify: `tests/test_chat_task_manager.py`
- Delete: `tests/test_chat_partial_verdict.py`
- Create: `tests/test_chat_cycle_cancellation.py`

**Interfaces:**
- Consumes: Task 2/8/9/10/11/12 的生产组件。
- Produces: `build_trading_cycle(config, mode, event_sink)`。
- Produces: Scheduler/CLI/Chat 对 `TradingCycle` 的唯一入口。

- [ ] **Step 1: 写 bootstrap 和 CLI 无 Graph 失败测试**

```python
def test_bootstrap_registers_builtins_and_custom_factories(monkeypatch):
    cycle = build_trading_cycle(config_with_factory("tests.fake_component:create"), mode="paper")
    assert cycle.registry.ids() == {"kronos", "llm_committee", "fake"}


def test_run_command_has_no_graph_option(runner):
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "--graph" not in result.stdout
```

- [ ] **Step 2: 写 Chat 事件和取消测试**

```python
async def test_chat_runner_forwards_cycle_events():
    bus = RecordingEventBus()
    await run_analysis_and_buffer(
        pair="BTC/USDT:USDT",
        session_id="s1",
        event_bus=bus,
        interrupt_event=asyncio.Event(),
        state_mgr=FakeStateManager(),
        cycle=FakeTradingCycle(events=[CycleEvent("component_completed", {"component_id": "kronos"})]),
    )
    assert any(name == "component_completed" for name, _ in bus.events)


async def test_chat_cancel_does_not_publish_partial_verdict():
    bus = RecordingEventBus()
    interrupt = asyncio.Event()
    interrupt.set()
    await run_cancelled_analysis(bus, interrupt)
    assert "verdict_partial" not in [name for name, _ in bus.events]
    assert "cycle_cancelled" in [name for name, _ in bus.events]
```

- [ ] **Step 3: 运行入口测试确认失败**

Run: `uv run pytest tests/test_bootstrap.py tests/test_scheduler.py tests/test_cli_agent_list.py tests/test_chat_task_manager.py tests/test_chat_cycle_cancellation.py --no-cov -q`

Expected: FAIL，入口仍构建顶层 Graph。

- [ ] **Step 4: 实现 bootstrap**

`build_trading_cycle()` 创建 Registry，注册 `KronosComponent`、`LLMCommitteeComponent` 和 TOML factory，创建 Profile/CycleJournal/Approval repositories、ContextProvider、Runner、Fusion、DecisionEngine、ExitPolicy、RiskGate、ExecutionPlanner 和对应 Executor。

```python
def build_trading_cycle(config: AppConfig, mode: str, event_sink: CycleEventSink | None = None) -> TradingCycle:
    registry = SignalComponentRegistry()
    registry.register(KronosComponent(config.kronos))
    registry.register(LLMCommitteeComponent(config, sink=event_sink or NullCycleEventSink()))
    for factory in config.signal_plugins.factories:
        registry.load_factory(factory)
    return TradingCycle(
        profiles=SignalProfileRepository(config.infrastructure.database_url),
        registry=registry,
        contexts=build_context_provider(config, mode),
        runner=ComponentRunner(event_sink or NullCycleEventSink()),
        fusion=WeightedSignalFusion(),
        decisions=DecisionEngine(),
        exits=AtrExitPolicy(),
        approvals=ApprovalStore(config.infrastructure.database_url),
        risk=build_risk_gate(config),
        execution_planner=ExecutionPlanner(config.risk.position.max_single_pct),
        executor=build_executor(config, mode),
        journal=CycleJournalStore(config.infrastructure.database_url),
        events=event_sink or NullCycleEventSink(),
    )
```

- [ ] **Step 5: 切换 Scheduler、CLI 和 Chat**

Scheduler 每个 pair 调用 `cycle.run()`；CLI `arena run` 删除 graph 参数；Chat 使用 `EventBusCycleSink`。`tracing.py` 改为记录业务阶段事件，不再提供顶层 `run_graph_traced()`。删除 partial verdict 代码与测试，原测试文件改为断言取消后无 partial 事件。

- [ ] **Step 6: 运行入口测试**

Run: `uv run pytest tests/test_bootstrap.py tests/test_scheduler.py tests/test_scheduler_misfire.py tests/test_cli_agent_list.py tests/test_chat_task_manager.py tests/test_chat_cycle_cancellation.py --no-cov -q`

Expected: PASS。

- [ ] **Step 7: 提交入口切换**

```bash
git add src/cryptotrader/bootstrap.py src/cryptotrader/scheduler.py src/cryptotrader/chat src/cryptotrader/tracing.py src/cli/main.py src/api/routes/chat.py src/api/routes/chat_control.py tests/test_bootstrap.py tests/test_scheduler.py tests/test_scheduler_misfire.py tests/test_cli_agent_list.py tests/test_chat_task_manager.py tests/test_chat_cycle_cancellation.py
git rm tests/test_chat_partial_verdict.py
git rm src/cryptotrader/chat/partial_verdict.py
git commit -m "refactor: route every live entry through TradingCycle"
```

---

### Task 14: 把 BacktestEngine 切换到相同 TradingCycle

**Files:**
- Rewrite: `src/cryptotrader/backtest/engine.py`
- Modify: `src/cryptotrader/backtest/session.py`
- Modify: `src/cryptotrader/backtest/result.py`
- Modify: `src/api/routes/backtest.py`
- Modify: `scripts/run_backtest.py`
- Modify: `scripts/kronos_backtest_ab.py`
- Modify: `tests/test_backtest.py`
- Modify: `tests/test_backtest_session_cache.py`
- Test: `tests/test_live_backtest_decision_parity.py`

**Interfaces:**
- Consumes: `TradingCycle` 核心服务和 `HistoricalSignalContextProvider`。
- Produces: BacktestExecutor、按启动 Profile revision 固定的回测运行。

- [ ] **Step 1: 写实盘/回测同决策失败测试**

```python
async def test_live_and_backtest_build_identical_trade_plan_from_same_context():
    active_profile = profile(kronos=0.6, llm=0.4)
    signals = (signal("kronos", "long", 0.8), signal("llm_committee", "short", 0.3))
    live = build_test_cycle(context_provider=FixedContextProvider(context()), signals=signals, profile=active_profile)
    backtest = build_test_cycle(context_provider=FixedContextProvider(context()), signals=signals, profile=active_profile)
    live_outcome = await live.run(request(mode="paper"))
    backtest_outcome = await backtest.run(request(mode="backtest"))
    assert live_outcome.trade_plan == backtest_outcome.trade_plan


async def test_backtest_profile_revision_is_fixed_for_entire_run(profile_repo):
    engine = BacktestEngine(profile_repository=profile_repo, cycle_factory=fake_cycle_factory())
    task = asyncio.create_task(engine.run())
    await engine.first_bar_processed.wait()
    await profile_repo.replace(profile(revision=2, kronos=1.0, llm=0.0))
    result = await task
    assert set(result.profile_revisions) == {1}
```

- [ ] **Step 2: 运行回测测试确认失败**

Run: `uv run pytest tests/test_live_backtest_decision_parity.py tests/test_backtest.py tests/test_backtest_session_cache.py --no-cov -q`

Expected: FAIL，BacktestEngine 仍构建 backtest graph 和旧 Verdict。

- [ ] **Step 3: 重写 BacktestEngine 主循环**

回测启动时读取一次 `SignalProfile`，为全部 bar 注入固定 Profile repository。每个 bar 用 `HistoricalSignalContextProvider` 构建截至当前时间的数据，用同一 Runner/Fusion/Decision/Exit/Risk/ExecutionPlanner 运行。`BacktestExecutor` 维持 equity、position、手续费、滑点和保护单状态；当前 bar 生成目标，下一 bar 开盘执行。

```python
for index in range(self.warmup_bars, len(self.ohlcv) - 1):
    as_of = self.ohlcv.index[index]
    request = CycleRequest(self.pair, "backtest", as_of=as_of)
    outcome = await cycle.run(request)
    executor.execute_pending_at(self.ohlcv.iloc[index + 1])
    self._record_equity(as_of, outcome)
```

删除 `_build_graph()`、`build_initial_state()`、`run_graph_traced()`、verdict/action/position_scale 分支。回测 Session 保存 `TradingCycleRecord` 或 cycle IDs，不保存 `DecisionCommit`。

- [ ] **Step 4: 运行回测与防前视测试**

Run: `uv run pytest tests/test_live_backtest_decision_parity.py tests/test_backtest.py tests/test_backtest_session_cache.py tests/test_anti_overfitting_equivalence.py --no-cov -q`

Expected: PASS。

- [ ] **Step 5: 提交统一回测主链**

```bash
git add src/cryptotrader/backtest src/api/routes/backtest.py scripts/run_backtest.py scripts/kronos_backtest_ab.py tests/test_live_backtest_decision_parity.py tests/test_backtest.py tests/test_backtest_session_cache.py tests/test_anti_overfitting_equivalence.py
git commit -m "refactor: run backtests through the shared trading cycle"
```

---

### Task 15: 重写 Decisions API 及依赖 Journal 的后端页面数据

**Files:**
- Rewrite: `src/api/routes/decisions.py`
- Modify: `src/api/routes/metrics.py`
- Modify: `src/api/routes/risk.py`
- Modify: `src/api/routes/portfolio_v2.py`
- Modify: `src/cryptotrader/hitl/store.py`
- Replace: `tests/test_api_decisions_list.py`
- Replace: `tests/test_api_decisions_detail.py`
- Modify: `tests/test_api_metrics_summary.py`
- Modify: `tests/test_api_portfolio_snapshot.py`
- Modify: `tests/test_api_risk.py`

**Interfaces:**
- Consumes: `CycleJournalStore`。
- Produces: 新 Decisions list/detail schema 和组件贡献展示数据。

- [ ] **Step 1: 写新 Decisions API 失败测试**

```python
async def test_decision_detail_exposes_components_fusion_and_target(api_client, cycle_store):
    record = await cycle_store.append(completed_cycle_record())
    response = await api_client.get(f"/api/decisions/{record.cycle_id}")
    body = response.json()
    assert body["cycle_id"] == record.cycle_id
    assert body["profile_revision"] == 3
    assert body["components"][0]["component_id"] == "kronos"
    assert body["fusion"]["score"] == pytest.approx(0.36)
    assert body["target_position"] == {"side": "long", "size_ratio": 0.2}


async def test_decision_list_includes_failed_cycles(api_client, cycle_store):
    await cycle_store.append(component_failed_record())
    body = (await api_client.get("/api/decisions")).json()
    assert body["items"][0]["status"] == "component_failed"
```

- [ ] **Step 2: 运行 API 测试确认失败**

Run: `uv run pytest tests/test_api_decisions_list.py tests/test_api_decisions_detail.py tests/test_api_metrics_summary.py tests/test_api_portfolio_snapshot.py tests/test_api_risk.py --no-cov -q`

Expected: FAIL，路由仍读取 `DecisionCommit`。

- [ ] **Step 3: 定义新 API schema 并切换 store**

list item 返回：`cycle_id`、`ts`、`pair`、`status`、`profile_revision`、`fused_score`、`target_position`、`risk_result`、`execution_result`。detail 额外返回 `components`、`fusion.contributions`、`trade_plan`、LLM component details 中的 agent analyses/debate turns。

Metrics、Risk、Portfolio 中需要历史交易的查询统一使用 `CycleJournalStore.list(status="completed")` 和 execution result；HITL cold-start 数量改为统计 `trading_cycles.status='completed'`。

- [ ] **Step 4: 运行后端 API 测试**

Run: `uv run pytest tests/test_api_decisions_list.py tests/test_api_decisions_detail.py tests/test_api_metrics_summary.py tests/test_api_portfolio_snapshot.py tests/test_api_risk.py --no-cov -q`

Expected: PASS。

- [ ] **Step 5: 提交 Journal API 切换**

```bash
git add src/api/routes/decisions.py src/api/routes/metrics.py src/api/routes/risk.py src/api/routes/portfolio_v2.py src/cryptotrader/hitl/store.py tests/test_api_decisions_list.py tests/test_api_decisions_detail.py tests/test_api_metrics_summary.py tests/test_api_portfolio_snapshot.py tests/test_api_risk.py
git commit -m "refactor: serve decisions from cycle journal records"
```

---

### Task 16: 实现网页 Strategy 动态配置页面

**Files:**
- Create: `web/src/pages/strategy/index.tsx`
- Create: `web/src/pages/strategy/components/component-weight-card.tsx`
- Create: `web/src/pages/strategy/components/decision-settings-card.tsx`
- Create: `web/src/pages/strategy/strategy-page.test.tsx`
- Create: `web/src/hooks/use-signal-profile.ts`
- Modify: `web/src/types/api.schema.ts`
- Modify: `web/src/types/api.ts`
- Modify: `web/src/App.tsx`
- Modify: `web/src/components/layout/sidebar.tsx`
- Modify: `web/src/components/layout/top-bar.tsx`
- Create: `web/src/locales/zh-CN/strategy.json`
- Create: `web/src/locales/en-US/strategy.json`
- Modify: `web/src/locales/zh-CN/common.json`
- Modify: `web/src/locales/en-US/common.json`
- Modify: `web/src/lib/i18n.ts`

**Interfaces:**
- Consumes: `GET/PUT /api/signal-profile`。
- Produces: `/strategy` 页面、查询/保存 hook、权重总计客户端验证。

- [ ] **Step 1: 写 Strategy 页面失败测试**

```tsx
it('disables save until enabled weights total 100%', async () => {
  renderStrategyPage(profileResponse({ kronos: 0.6, llm_committee: 0.4 }));
  await user.clear(screen.getByLabelText('Kronos 权重'));
  await user.type(screen.getByLabelText('Kronos 权重'), '50');
  expect(screen.getByText('权重总计 90%')).toBeInTheDocument();
  expect(screen.getByRole('button', { name: '保存并从下一周期生效' })).toBeDisabled();
});


it('saves the complete profile and renders the new revision', async () => {
  const put = mockProfilePut({ revision: 4 });
  renderStrategyPage(profileResponse({ revision: 3 }));
  await user.click(screen.getByRole('button', { name: '保存并从下一周期生效' }));
  expect(put).toHaveBeenCalledWith(expect.objectContaining({ revision: 3 }));
  expect(await screen.findByText('Revision 4')).toBeInTheDocument();
});
```

- [ ] **Step 2: 运行前端测试确认失败**

Run: `pnpm --dir web test -- src/pages/strategy/strategy-page.test.tsx`

Expected: FAIL，页面与 schema 不存在。

- [ ] **Step 3: 增加 Zod schema 和 TanStack Query hook**

```tsx
export const SignalProfileSchema = z.object({
  revision: z.number().int().positive(),
  components: z.array(z.object({
    component_id: z.string(),
    enabled: z.boolean(),
    weight: z.number().min(0).max(1),
  })),
  neutral_threshold: z.number().min(0).lt(1),
  max_target_ratio: z.number().gt(0).max(1),
  atr_stop_multiplier: z.number().positive(),
  reward_ratio: z.number().positive(),
  hitl_required: z.boolean(),
  installed_components: z.array(z.object({
    component_id: z.string(),
    display_name: z.string(),
    description: z.string(),
  })),
  updated_at: z.string(),
});
```

hook 使用 `queryKey: ['signal-profile']`，PUT 成功后用 response 更新 query cache。

- [ ] **Step 4: 实现页面和导航**

组件列表使用 Switch 和百分比 number input；禁用组件提交 weight=0。页面包含中性阈值、最大目标仓位、ATR 倍数、盈亏比和 HITL Switch。总权重用整数百分比显示，`Math.abs(total - 1) > 1e-9` 时禁用保存。

- [ ] **Step 5: 运行前端页面测试、类型和 lint**

Run: `pnpm --dir web test -- src/pages/strategy/strategy-page.test.tsx`

Expected: PASS。

Run: `pnpm --dir web typecheck`

Expected: PASS。

Run: `pnpm --dir web lint`

Expected: PASS。

- [ ] **Step 6: 提交 Strategy 页面**

```bash
git add web/src/pages/strategy web/src/hooks/use-signal-profile.ts web/src/types web/src/App.tsx web/src/components/layout web/src/locales web/src/lib/i18n.ts
git commit -m "feat(web): add dynamic signal strategy settings"
```

---

### Task 17: 更新 Decisions、Debate、HITL 和 Chat 前端契约

**Files:**
- Modify: `web/src/types/api.schema.ts`
- Modify: `web/src/pages/decisions/index.tsx`
- Modify: `web/src/pages/decisions/components/decisions-table.tsx`
- Modify: `web/src/components/decision-detail/decision-detail-panel.tsx`
- Modify: `web/src/pages/debate/index.tsx`
- Modify: `web/src/pages/risk/components/approval-item.tsx`
- Modify: `web/src/pages/risk/components/approval-queue-card.tsx`
- Modify: `web/src/hooks/use-analysis-progress.ts`
- Modify: `web/src/hooks/use-chat-messages.ts`
- Modify: `web/src/pages/chat/index.tsx`
- Test: `web/src/pages/decisions/decisions-cycle-record.test.tsx`
- Test: `web/src/pages/risk/components/approval-target-plan.test.tsx`
- Modify: `web/src/pages/chat/chat-page.test.tsx`

**Interfaces:**
- Consumes: 新 Decisions/HITL API 和 CycleEvent names。
- Produces: 组件贡献、融合分数、目标仓位、内部辩论和计划审批 UI。

- [ ] **Step 1: 写新决策详情和审批卡测试**

```tsx
it('renders component weighted contributions and target position', async () => {
  renderDecisionDetail(cycleDecisionFixture());
  expect(await screen.findByText('Kronos')).toBeInTheDocument();
  expect(screen.getByText('+0.48')).toBeInTheDocument();
  expect(screen.getByText('目标多仓 30%')).toBeInTheDocument();
});


it('approval card renders target plan instead of legacy verdict', () => {
  render(<ApprovalItem approval={targetPlanApprovalFixture()} />);
  expect(screen.getByText('目标空仓 40%')).toBeInTheDocument();
  expect(screen.queryByText('position_scale')).not.toBeInTheDocument();
});
```

- [ ] **Step 2: 写 Chat 无 partial verdict 测试**

```tsx
it('shows cancellation without constructing a partial verdict', async () => {
  renderChatWithEvents([{ type: 'cycle_cancelled', data: { reason: 'user' } }]);
  expect(await screen.findByText('本轮分析已取消')).toBeInTheDocument();
  expect(screen.queryByText('部分裁决')).not.toBeInTheDocument();
});
```

- [ ] **Step 3: 运行前端契约测试确认失败**

Run: `pnpm --dir web test -- src/pages/decisions/decisions-cycle-record.test.tsx src/pages/risk/components/approval-target-plan.test.tsx src/pages/chat/chat-page.test.tsx`

Expected: FAIL，前端仍依赖 legacy Verdict。

- [ ] **Step 4: 替换 Zod schema 与页面字段**

Decisions schema 使用 `cycle_id/status/profile_revision/components/fusion/target_position/trade_plan/risk_result/execution_result`。Debate 从 `components[id='llm_committee'].details` 读取 analyses、turns 和 consensus。HITL schema 使用 `trade_plan`。Chat progress reducer只处理 CycleEvent，不生成 partial verdict state。

- [ ] **Step 5: 运行前端相关测试与类型检查**

Run: `pnpm --dir web test -- src/pages/decisions/decisions-cycle-record.test.tsx src/pages/risk/components/approval-target-plan.test.tsx src/pages/chat/chat-page.test.tsx`

Expected: PASS。

Run: `pnpm --dir web typecheck`

Expected: PASS。

- [ ] **Step 6: 提交前端契约切换**

```bash
git add web/src/types/api.schema.ts web/src/pages/decisions web/src/components/decision-detail web/src/pages/debate web/src/pages/risk/components web/src/hooks/use-analysis-progress.ts web/src/hooks/use-chat-messages.ts web/src/pages/chat
git commit -m "refactor(web): render cycle fusion decisions"
```

---

### Task 18: 删除旧顶层架构并建立禁止回归门禁

**Files:**
- Delete: `src/cryptotrader/graph.py`
- Delete: `src/cryptotrader/state.py`
- Delete: `src/cryptotrader/nodes/`
- Delete: `src/cryptotrader/debate/verdict.py`
- Delete: `src/cryptotrader/debate/researchers.py`
- Modify: `src/cryptotrader/models.py`
- Modify: `src/cryptotrader/config.py`
- Modify: `config/default.toml`
- Modify: `pyproject.toml`
- Modify: `README.md`
- Modify: `README_EN.md`
- Modify: `ARCHITECTURE.md`
- Modify: `docs/ARCHITECTURE.md`
- Delete/Replace: graph、state、verdict、node 专属测试
- Create: `tests/test_signal_architecture_boundary.py`

**Interfaces:**
- Consumes: 前 17 个任务完成的所有新入口。
- Produces: 旧架构符号搜索为零的硬门禁。

- [ ] **Step 1: 写禁止旧架构的失败测试**

```python
def test_legacy_trading_architecture_is_absent():
    root = Path(__file__).parents[1]
    assert not (root / "src/cryptotrader/graph.py").exists()
    assert not (root / "src/cryptotrader/state.py").exists()
    assert not (root / "src/cryptotrader/nodes").exists()
    forbidden = (
        "signal_engine",
        "TradeVerdict",
        "verdict_source",
        "build_trading_graph",
        "build_kronos_graph",
        "build_backtest_graph",
        "build_initial_state",
        "verdict_partial",
    )
    runtime = "\n".join(path.read_text(errors="ignore") for path in (root / "src").rglob("*.py"))
    for symbol in forbidden:
        assert symbol not in runtime
```

- [ ] **Step 2: 运行门禁确认旧代码仍存在**

Run: `uv run pytest tests/test_signal_architecture_boundary.py --no-cov -q`

Expected: FAIL，列出旧文件或符号。

- [ ] **Step 3: 删除旧文件、字段、配置和测试**

从 `models.py` 删除 `TradeVerdict`、`DecisionCommit`、`CommitObservability` 及只为旧 Journal 存在的类型。从 `config.py` 和 `config/default.toml` 删除 `signal_engine` 和 `kronos.enabled`。从 Ruff banned imports 和 per-file ignores 删除 graph/nodes 条目。

删除以下旧测试组并保留前面任务创建的新测试：

```text
tests/test_graph.py
tests/test_graph_topology.py
tests/test_state_coverage.py
tests/test_us1_state_schema_bump.py
tests/test_verdict.py
tests/test_verdict_helpers.py
tests/test_verdict_nodes_coverage.py
tests/test_post_process_verdict.py
tests/test_nodes.py
tests/test_nodes_execution_sync.py
tests/test_observability_nodes.py
```

使用 `rg` 找出其余旧符号测试；如果测试只验证被删除接口则删除，如果验证仍存在的业务规则则改为新领域模型测试。

从 `tests/test_data_collectors.py` 删除只覆盖 `run_debate()`、`judge_debate()` 和 `_format_reports()` 的旧 researcher 测试，保留 Market/Onchain/Macro/Snapshot collector 测试。四智能体内部辩论由 Task 9 的 `tests/test_llm_committee_component.py`、`tests/test_debate_anti_ratchet.py` 和 `tests/test_debate_turn_capture.py` 接替覆盖。

- [ ] **Step 4: 更新项目文档为单一 TradingCycle 架构**

README 和 Architecture 必须描述组件注册、全局 Profile、加权融合、TargetPosition、统一 ExitPolicy 和 Strategy 页面；删除三种图模式、Kronos/LLM 二选一、旧 Verdict 和 lite graph 命令示例。

- [ ] **Step 5: 运行删除门禁和 import collection**

Run: `uv run pytest tests/test_signal_architecture_boundary.py --no-cov -q`

Expected: PASS。

Run: `uv run pytest --collect-only -q`

Expected: PASS，无 import error。

- [ ] **Step 6: 提交旧架构删除**

```bash
git add -A src config pyproject.toml README.md README_EN.md ARCHITECTURE.md docs/ARCHITECTURE.md tests
git commit -m "refactor: remove the legacy graph verdict architecture"
```

---

### Task 19: 全流程 E2E、质量门禁和完成审计

**Files:**
- Create: `tests/test_signal_fusion_e2e.py`
- Modify: `tests/test_integration.py`
- Modify: `tests/test_contract_shape.py`
- Modify: `tests/test_arch_boundary_8_1.py`
- Modify: `tests/test_arch_boundary_8_2.py`
- Modify: `tests/test_docker_compose.py`
- Modify: `docker-compose.yml`
- Modify: `Dockerfile`

**Interfaces:**
- Consumes: 完整新系统。
- Produces: 离线全流程验收证据和最终可运行构建。

- [ ] **Step 1: 写端到端成功、动态配置和失败周期测试**

```python
async def test_full_cycle_uses_web_saved_profile_on_next_cycle(sqlite_url):
    app, cycle = await build_e2e_system(sqlite_url)
    await put_profile(app, kronos=0.6, llm=0.4, neutral_threshold=0.2)
    first = await cycle.run(request())
    assert first.status == "completed"
    assert first.fused_signal.score == pytest.approx(0.36)
    await put_profile(app, kronos=1.0, llm=0.0, neutral_threshold=0.2)
    second = await cycle.run(request())
    assert second.profile_revision == first.profile_revision + 1
    assert second.fused_signal.score == pytest.approx(0.8)


async def test_failed_component_records_cycle_and_places_no_order(sqlite_url):
    app, cycle = await build_e2e_system(sqlite_url, llm_error=RuntimeError("offline"))
    outcome = await cycle.run(request())
    assert outcome.status == "component_failed"
    assert cycle.executor.orders == []
    record = await cycle.journal.get(outcome.cycle_id)
    assert record.component_error == {"llm_committee": "RuntimeError: offline"}
```

- [ ] **Step 2: 运行新 E2E 测试**

Run: `uv run pytest tests/test_signal_fusion_e2e.py --no-cov -q`

Expected: PASS。

- [ ] **Step 3: 运行 Python 全量测试与 Ruff**

Run: `uv run pytest tests/ -q`

Expected: PASS，coverage ≥ 70%。

Run: `uv run ruff check src tests`

Expected: PASS。

Run: `uv run ruff format --check src tests`

Expected: PASS。

- [ ] **Step 4: 运行前端全部质量门禁**

Run: `pnpm --dir web test`

Expected: PASS。

Run: `pnpm --dir web typecheck`

Expected: PASS。

Run: `pnpm --dir web lint`

Expected: PASS。

Run: `pnpm --dir web build`

Expected: PASS，生成 `web/dist`。

- [ ] **Step 5: 验证容器配置和旧符号为零**

Run: `docker compose config --quiet`

Expected: exit 0。

Run: `rg -n "signal_engine|TradeVerdict|verdict_source|build_(trading|kronos|backtest|lite)_graph|build_initial_state|verdict_partial" src config web/src`

Expected: 无输出，exit 1。

- [ ] **Step 6: 对照设计文档逐项完成审计**

逐节核对设计文档 2–19：组件协议、严格失败、融合数学、目标仓位、ATR、动态 Profile、Strategy 页面、HITL、RiskGate、实盘/回测同路、事件、Journal、删除范围和测试。每项必须有当前代码与测试输出证据；缺失项继续实现，不以“多数测试通过”替代完整验收。

- [ ] **Step 7: 提交最终 E2E 与门禁修复**

```bash
git add -A
git commit -m "test: verify the pluggable signal fusion workflow"
```

- [ ] **Step 8: 最终工作区和提交审计**

Run: `git status --short`

Expected: 无输出。

Run: `git log --oneline --max-count=20`

Expected: 包含本计划各任务提交，且没有无关文件。

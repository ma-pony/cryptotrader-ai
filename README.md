# CryptoTrader AI

CryptoTrader AI 是一个可插拔的加密资产信号融合与交易系统。Kronos、LLM 四智能体委员会以及自定义策略都只是信号组件；系统把所有启用组件的输出按信任权重确定性融合，再统一生成目标仓位、退出价格、风控结论和执行计划。

## 核心模型

每个组件只回答三个问题：方向、置信度和理由。

```text
ComponentSignal
  component_id
  direction: long | short | neutral
  confidence: 0..1
  reasoning
  details
```

融合器先把方向映射为 `long=+1`、`short=-1`、`neutral=0`，计算启用组件的加权分数。这个正负分数只用于内部融合；最终业务对象是更直观的：

```text
TargetPosition
  side: long | short | flat
  size_ratio: 0..1
```

`size_ratio` 表示在风控允许的单标的上限内希望达到的比例，不是杠杆倍数，也不是直接下单量。执行层比较当前仓位和目标仓位，只交易差值。

## 内置组件

- `kronos`：Kronos 时序模型与门控逻辑，输出纯市场方向信号。
- `llm_committee`：技术、链上、新闻、宏观四个智能体。委员会内部保留多轮交叉质询、收敛判断和最终摘要，外部仍只输出一个标准信号。

自定义组件实现统一协议后，通过 `[signal_plugins].factories` 注册。Factory 使用 `package.module:function` 格式并返回组件实例。启动时会校验组件 ID 唯一、Profile 引用的组件均已安装。

## 全局 Signal Profile

默认配置位于 `config/default.toml`：

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

所有启用组件的权重必须精确合计为 `1.0`。网页 `/strategy` 可以动态启停组件、调整信任占比、融合阈值、目标仓位上限、ATR 止损倍数、盈亏比和 HITL。保存会生成新的 Profile revision；已经开始的周期继续使用冻结快照，新配置从下一周期生效。

## 唯一交易主链

实时、模拟和回测共用同一个 `TradingCycle`：

```text
冻结 Profile
  → 汇总所有组件的数据需求
  → 生成同一时点的 SignalContext
  → 并行运行全部启用组件
  → 严格成功检查
  → 加权融合
  → TargetPosition
  → 统一 ATR ExitPolicy
  → 可选 HITL
  → RiskGate
  → 目标仓位差值下单
  → trading_cycles Journal
```

任何启用组件失败，整个周期以 `component_failed` 结束，不拿残缺结果交易。HITL 开启后保存完整的 Profile revision、上下文和交易计划；网页审批通过时会重新读取当前仓位并重新规划订单，但不会重新生成信号。

## 启动

要求 Python 3.12+、Node.js 20+、uv 和 pnpm。

```bash
uv sync --all-extras
cp config/default.toml config/local.toml

uv run arena serve --port 8003
cd web && pnpm install && pnpm dev
```

常用命令：

```bash
uv run arena run --pair BTC/USDT --mode paper
uv run arena backtest --pair BTC/USDT --start 2025-01-01 --end 2025-03-01
uv run arena journal log
uv run arena journal show <cycle-id>
uv run arena scheduler start
uv run arena migrate
```

API 的主要资源包括 `/api/signal-profile`、`/api/decisions`、`/api/hitl`、`/api/portfolio`、`/api/risk` 和 `/api/backtest`。

## 验证

```bash
uv run pytest --no-cov -q
uv run ruff check src tests scripts
cd web && pnpm test && pnpm typecheck && pnpm lint
```

更完整的模块边界和数据流见 [ARCHITECTURE.md](ARCHITECTURE.md)。

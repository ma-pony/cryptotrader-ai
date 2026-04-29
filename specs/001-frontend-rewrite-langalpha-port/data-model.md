# Phase 1 — Data Model

**Feature**: `001-frontend-rewrite-langalpha-port`
**Date**: 2026-04-16
**Source**: spec §Key Entities + 后端既有 Pydantic models

本文档定义本特性涉及的数据实体，按"前后端共享契约"维度组织。每个实体含字段、类型、关系、约束。前端 zod schema、后端 pydantic model 均以此为单一事实来源。

> **约定**：
> - 时间戳统一 ISO 8601 字符串（带时区，UTC `Z` 后缀）
> - 金额统一 `number`（浮点，币种由上下文给出）
> - 枚举值统一小写下划线（如 `long` / `short` / `hold`）
> - 可选字段标 `?:`；后端 pydantic 用 `Optional[X] = None`

---

## 1. 投资组合域（Dashboard 用）

### Portfolio

```ts
type Portfolio = {
  equity: number;            // 总权益（USDT 计价）
  cash: number;              // 可用现金
  positions: Position[];     // 当前持仓
  pnl_24h: number;           // 24h 盈亏（绝对值；负数表示亏损）
  pnl_24h_pct: number;       // 24h 盈亏百分比
  drawdown: number;          // 当前回撤（百分比，0~1）
  updated_at: string;        // ISO 8601
};
```

**关系**：`Position[]` 1:N
**约束**：
- `equity ≥ 0`；`cash ≥ 0`
- `drawdown ∈ [0, 1]`
- `updated_at` 在响应中必填，前端用于显示"X 秒前更新"

### Position

```ts
type Position = {
  pair: string;              // 例: "BTC/USDT"
  side: "long" | "short";
  size: number;              // 持仓数量（基础币种）
  avg_price: number;         // 成本均价（USDT）
  unrealized_pnl: number;    // 未实现盈亏（USDT）
  unrealized_pnl_pct: number;
  opened_at: string;
};
```

**约束**：
- `size > 0`
- `pair` 格式 `<base>/<quote>`，正则 `^[A-Z0-9]{2,10}\/[A-Z]{3,6}$`

### EquityPoint

```ts
type EquityPoint = {
  ts: string;                // ISO 8601
  equity: number;            // 该时点权益
};

type EquityCurve = {
  range: "24h" | "7d" | "30d" | "all";
  points: EquityPoint[];     // 按 ts 升序
};
```

**约束**：
- 1000 点上限（前端渲染性能预算 NFR-P-004）；超出由后端降采样
- `range=all` 至多返回 1000 点（ASOF 降采样）

---

## 2. 决策域（Decisions 页 + Backtest 详情用）

### DecisionCommit

```ts
type DecisionCommit = {
  commit_hash: string;       // 主键，git-style sha
  ts: string;                // 决策生成时间
  pair: string;
  price: number;             // 决策时的市价
  trace_id?: string;         // OTel trace（可选）

  agent_analyses: AgentAnalysis[];     // 4 个（NewsAgent/MacroAgent/SentimentAgent/TechnicalAgent）
  experience_memory_ref?: ExperienceMemoryRef;
  debate_rounds: DebateRound[];         // 0 或 2 轮（debate_skipped 时为 []）
  debate_skipped?: boolean;
  debate_skip_reason?: "consensus" | "confusion" | "";
  verdict: Verdict;
  risk_gate: RiskGate;
  execution?: Execution;                // 拒绝时无
  node_timeline: NodeTimelineEntry[];   // pipeline 各节点耗时
  is_filled: boolean;
};
```

**关系**：
- `AgentAnalysis[]` 1:N（固定 4）
- `DebateRound[]` 0:N
- `Verdict` 1:1
- `RiskGate` 1:1
- `Execution` 0:1

**状态转换**：
```
created → analyzed → (debated?) → verdicted → risk_checked → (executed | rejected)
```

### AgentAnalysis

```ts
type AgentAnalysis = {
  name: "NewsAgent" | "MacroAgent" | "SentimentAgent" | "TechnicalAgent";
  score: number;             // -1.0 ~ 1.0（看跌 ~ 看涨）
  confidence: number;        // 0.0 ~ 1.0
  reasoning: string;         // Markdown
  is_mock: boolean;          // 数据缺失时为 true
};
```

### DebateRound

```ts
type DebateRound = {
  round: 1 | 2;
  bull_message: string;      // Markdown
  bear_message: string;      // Markdown
};
```

### Verdict

```ts
type Verdict = {
  action: "long" | "short" | "hold";
  size: number;              // 仓位比例 0.0~1.0
  confidence: number;        // 0.0~1.0
  reasoning: string;         // Markdown
  source: "ai" | "weighted_downgrade";
};
```

### RiskGate

```ts
type RiskGate = {
  passed: boolean;
  rejected_by?: string;      // 首个失败检查名（passed=false 时）
  reason?: string;           // 拒绝/调整原因
  scale_adjustment?: number; // 聚合后的 position_scale 上限（min(所有检查的提议)）；null = 无 check 提议调整
  checks: RiskCheck[];       // 序列，按执行顺序
};

type RiskCheck = {
  name: string;              // 例: "max_position", "circuit_breaker", "post_loss_cooldown"
  passed: boolean;
  reason?: string;           // 拒绝时必填
  threshold?: number | string;
  scale_adjustment?: number; // 该 check 提议的 position_scale 上限（∈ [0, 1]）；null = 无意见
};
```

**契约 (PROD-I3)**：风控 check **MUST NOT** 在 `evaluate()` 中原地修改 `verdict.position_scale`，必须通过 `CheckResult.scale_adjustment` 返回提议。`risk_check` 节点聚合所有 passing check 的提议（取 min），然后通过 LangGraph 的 return delta 写回 `state["data"]["verdict"]["position_scale"]`，保证 graph 状态变更可追踪。

### Execution

```ts
type Execution = {
  order_id: string;
  status: "filled" | "partial" | "open" | "canceled" | "rejected";
  fill_price: number;
  fill_size: number;
  fee: number;
  slippage_bps: number;      // 滑点基点
  exchange: "binance" | "okx" | "paper";
};
```

### NodeTimelineEntry

```ts
type NodeTimelineEntry = {
  node: string;              // 例: "verbal_reinforcement", "agents_parallel", "debate_gate"
  start_ms: number;          // 相对决策开始的毫秒数
  duration_ms: number;
};
```

### ExperienceMemoryRef

```ts
type ExperienceMemoryRef = {
  memory_id: string;
  success_patterns: ExperienceRule[];
  forbidden_zones: ExperienceRule[];
  strategic_insights: ExperienceRule[];
};

type ExperienceRule = {
  pattern: string;           // 自然语言描述
  conditions: { regime_tags: string[]; [k: string]: unknown };
  rate: number;              // 0~1，胜率或规则强度
  maturity: "draft" | "validated" | "verified";
  source: "reflection" | "manual";
};
```

---

## 3. 回测域（Backtest 页用）

### BacktestParams

```ts
type BacktestParams = {
  start: string;             // YYYY-MM-DD
  end: string;               // YYYY-MM-DD
  pair: string;
  initial_capital: number;   // ≥ 100
  mode: "rules" | "llm";
  session_name?: string;     // 可选保存名
};
```

**约束**：
- `start < end`
- `end ≤ today`
- `initial_capital ≥ 100`（FR-301）

### BacktestRun

```ts
type BacktestRun = {
  run_id: string;
  params: BacktestParams;
  status: "queued" | "running" | "completed" | "failed" | "canceled";
  progress: number;          // 0~1
  started_at: string;
  finished_at?: string;
  error?: string;
  result?: BacktestResult;
};
```

**状态转换**：
```
queued → running → (completed | failed | canceled)
```

### BacktestResult

```ts
type BacktestResult = {
  metrics: {
    total_return_pct: number;
    sharpe: number;
    max_drawdown_pct: number;
    win_rate: number;        // 0~1
    trades_count: number;
  };
  equity_curve: EquityPoint[];
  decisions: DecisionCommit[];   // mode=llm 才有；rules 模式为 []
};
```

### BacktestSession

```ts
type BacktestSession = {
  name: string;              // 主键
  params: BacktestParams;
  result: BacktestResult;
  saved_at: string;
};
```

---

## 4. 风控域（Risk 页用）

### RiskStatus

```ts
type RiskStatus = {
  trade_count_hour: number;
  trade_count_day: number;
  circuit_breaker: CircuitBreakerStatus;
  thresholds: RiskThresholds;
  redis_available: boolean;  // false 时 trade_count_* 显示 N/A
};

type CircuitBreakerStatus = {
  state: "active" | "inactive";
  triggered_at?: string;
  expires_at?: string;       // active 时必填
  reason?: string;           // active 时必填
};

type RiskThresholds = {
  max_position_pct: number;       // 例: 0.3
  max_daily_loss_pct: number;     // 例: 0.05
  max_stop_loss_pct: number;      // 例: 0.02
  max_trades_per_hour: number;
  max_trades_per_day: number;
  post_loss_cooldown_seconds: number;
};
```

### CircuitBreakerResetRequest / Response

```ts
type CircuitBreakerResetRequest = {
  // 无 body；仅 X-API-Key 鉴权 + 二次确认（前端 dialog）
};

type CircuitBreakerResetResponse = {
  success: boolean;
  message: string;           // i18n 文案 key 或英文消息
};
```

---

## 5. 调度器域（Dashboard 用）

### SchedulerStatus

```ts
type SchedulerStatus = {
  enabled: boolean;
  next_pair?: string;        // 下一次触发的币对
  next_run_at?: string;      // ISO 8601
  redis_available: boolean;
};
```

---

## 6. 指标域（Metrics 页用）

### MetricsSummary

```ts
type MetricsSummary = {
  counters: {
    trades_total: number;
    orders_placed: number;
    orders_failed: number;
    risk_rejections: number;
    debate_skipped_total: number;
    [k: string]: number;
  };
  percentiles: {
    pipeline_p50_ms: number;
    pipeline_p95_ms: number;
    execution_p50_ms: number;
    execution_p95_ms: number;
    [k: string]: number;
  };
  collected_at: string;
};
```

### MetricsTrendPoint（前端 IndexedDB 缓存）

```ts
type MetricsTrendPoint = {
  ts: string;
  trades_total: number;
  pipeline_p95_ms: number;
  execution_p95_ms: number;
  risk_rejections: number;
};
// FIFO 60 样本上限（FR-503）
```

---

## 7. ChatAgent 域（P2）

### ChatMessage

```ts
type ChatMessage = {
  id: string;
  role: "user" | "assistant" | "tool";
  content_chunks: string[];          // 流式累积
  tool_calls?: ToolCall[];
  inline_widgets?: InlineWidget[];
  created_at: string;
};

type ToolCall = {
  call_id: string;
  tool_name: string;
  args: Record<string, unknown>;
  result?: unknown;
};

type InlineWidget = {
  widget_id: string;
  type: "chart" | "table" | "verdict" | "markdown";
  payload: Record<string, unknown>;  // 在 iframe sandbox 内渲染
};
```

### ChatSession

```ts
type ChatSession = {
  session_id: string;
  title: string;
  messages: ChatMessage[];
  model: string;             // 选用的 LLM 模型 id
  created_at: string;
  updated_at: string;
};
// IndexedDB 持久化
```

---

## 8. 市场域（MarketView P2）

### MarketDataPoint

```ts
type FundingRatePoint = {
  ts: string;
  rate: number;              // 例: 0.0001（万分之一）
  predicted_rate?: number;
};

type OpenInterestPoint = {
  ts: string;
  oi_usd: number;
};

type LiquidationEvent = {
  ts: string;
  side: "long" | "short";
  size_usd: number;
  price: number;
};

type MarketSnapshot = {
  pair: string;
  exchange: "binance" | "okx";
  funding_rate_24h: FundingRatePoint[];
  open_interest_24h: OpenInterestPoint[];
  liquidations_24h: LiquidationEvent[];
  perp_spot_basis_pct: number;  // 永续-现货价差
};
```

---

## 9. 共享元类型

### ApiError

```ts
type ApiError = {
  detail: string;            // 人类可读消息（中文）
  code?: string;             // 业务错误码（可选）
  trace_id?: string;
};
```

### Pagination

```ts
type PaginatedResponse<T> = {
  items: T[];
  total: number;
  page: number;              // 1-based
  size: number;
  has_next: boolean;
};
```

---

## 10. 验证规则汇总

| 实体 | 关键校验 |
|------|---------|
| Portfolio.drawdown | 0 ≤ x ≤ 1 |
| Position.size | > 0 |
| Position.pair | 正则匹配 `^[A-Z0-9]{2,10}\/[A-Z]{3,6}$` |
| EquityCurve.points | 长度 ≤ 1000；按 ts 升序 |
| AgentAnalysis.score | -1 ≤ x ≤ 1 |
| AgentAnalysis.confidence | 0 ≤ x ≤ 1 |
| Verdict.size | 0 ≤ x ≤ 1 |
| BacktestParams.initial_capital | ≥ 100 |
| BacktestParams 日期 | start < end ≤ today |
| RiskThresholds.max_*_pct | 0 < x < 1 |
| MetricsTrendPoint 数组 | FIFO 60 上限 |
| ChatMessage.content_chunks | 流式追加，永不重写 |
| InlineWidget.payload | 必须可 JSON.stringify（含 NaN/Infinity 由 streamFetch patch） |

---

## 11. 实体关系图（简化）

```
Portfolio ─┬─ Position[]
           └─ EquityCurve

DecisionCommit ─┬─ AgentAnalysis[]
                ├─ DebateRound[]
                ├─ Verdict
                ├─ RiskGate ── RiskCheck[]
                ├─ Execution
                ├─ NodeTimelineEntry[]
                └─ ExperienceMemoryRef ── ExperienceRule[]

BacktestRun ── BacktestResult ── DecisionCommit[]
BacktestSession ── BacktestResult

RiskStatus ─┬─ CircuitBreakerStatus
            └─ RiskThresholds

ChatSession ── ChatMessage ─┬─ ToolCall[]
                            └─ InlineWidget[]

MarketSnapshot ─┬─ FundingRatePoint[]
                ├─ OpenInterestPoint[]
                └─ LiquidationEvent[]
```

---

**Phase 1 数据模型出口**：实体清晰、约束明确、关系闭合，可基于此生成 zod schema 与 pydantic model。下一步进入接口合约（contracts/）。

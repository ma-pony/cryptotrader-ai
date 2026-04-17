import { z } from 'zod';

// ── Common ──

export const ApiErrorSchema = z.object({
  code: z.string(),
  message: z.string(),
  trace_id: z.string().optional(),
  details: z.record(z.unknown()).optional(),
});
export type ApiError = z.infer<typeof ApiErrorSchema>;

// ── §1 Portfolio (matches PortfolioSnapshotOut / EquityCurveOut) ──

export const PositionSchema = z.object({
  pair: z.string(),
  side: z.enum(['long', 'short']),
  size: z.number(),
  avg_price: z.number(),
  unrealized_pnl: z.number(),
  unrealized_pnl_pct: z.number(),
  opened_at: z.string().nullable().optional(),
});

export const PortfolioSchema = z.object({
  equity: z.number(),
  cash: z.number(),
  positions: z.array(PositionSchema),
  pnl_24h: z.number(),
  pnl_24h_pct: z.number(),
  drawdown: z.number(),
  updated_at: z.string(),
});

export const EquityPointSchema = z.object({
  ts: z.string(),
  equity: z.number(),
});

export const RangeWindowSchema = z.enum(['24h', '7d', '30d', 'all']);

export const EquityCurveSchema = z.object({
  range: RangeWindowSchema,
  points: z.array(EquityPointSchema),
});

// ── §2 Scheduler (matches SchedulerContractStatus) ──

export const SchedulerStatusSchema = z.object({
  enabled: z.boolean(),
  next_pair: z.string().nullable(),
  next_run_at: z.string().nullable(),
  redis_available: z.boolean(),
});

// ── §3 Decisions (matches DecisionListItem / DecisionDetailOut) ──

export const VerdictSlimSchema = z.object({
  action: z.string(),
  size: z.number().default(0),
  confidence: z.number().default(0),
  reasoning: z.string().default(''),
  source: z.string().default('ai'),
});

export const DecisionListItemSchema = z.object({
  commit_hash: z.string(),
  ts: z.string(),
  pair: z.string(),
  price: z.number().default(0),
  verdict: VerdictSlimSchema,
  is_filled: z.boolean().default(false),
  trace_id: z.string().nullable().optional(),
});

export const PaginatedDecisionsSchema = z.object({
  items: z.array(DecisionListItemSchema),
  total: z.number(),
  page: z.number(),
  size: z.number(),
  has_next: z.boolean(),
});

export const AgentAnalysisSchema = z.object({
  name: z.string(),
  score: z.number(),
  confidence: z.number(),
  reasoning: z.string().default(''),
  is_mock: z.boolean().default(false),
});

export const DebateRoundSchema = z.object({
  round: z.number(),
  bull_message: z.string().default(''),
  bear_message: z.string().default(''),
});

export const RiskCheckSchema = z.object({
  name: z.string(),
  passed: z.boolean(),
  reason: z.string().nullable().optional(),
  threshold: z.union([z.number(), z.string()]).nullable().optional(),
});

export const RiskGateSchema = z.object({
  passed: z.boolean(),
  checks: z.array(RiskCheckSchema).default([]),
});

export const ExecutionSchema = z.object({
  order_id: z.string(),
  status: z.string(),
  fill_price: z.number().default(0),
  fill_size: z.number().default(0),
  fee: z.number().default(0),
  slippage_bps: z.number().default(0),
  exchange: z.string().default('paper'),
});

export const NodeTimelineEntrySchema = z.object({
  node: z.string(),
  start_ms: z.number(),
  duration_ms: z.number(),
});

export const ExperienceMemoryRefSchema = z.object({
  memory_id: z.string().default(''),
  success_patterns: z.array(z.record(z.unknown())).default([]),
  forbidden_zones: z.array(z.record(z.unknown())).default([]),
  strategic_insights: z.array(z.unknown()).default([]),
});

export const DecisionDetailSchema = z.object({
  commit_hash: z.string(),
  ts: z.string(),
  pair: z.string(),
  price: z.number(),
  agent_analyses: z.array(AgentAnalysisSchema),
  debate_rounds: z.array(DebateRoundSchema),
  verdict: VerdictSlimSchema,
  risk_gate: RiskGateSchema,
  execution: ExecutionSchema.nullable(),
  node_timeline: z.array(NodeTimelineEntrySchema),
  experience_memory_ref: ExperienceMemoryRefSchema,
  trace_id: z.string().nullable().optional(),
});

// ── §4 Backtest (matches BacktestParams / BacktestRunStatus / sessions) ──

export const BacktestParamsSchema = z.object({
  start: z.string(),
  end: z.string(),
  pair: z.string(),
  initial_capital: z.number(),
  mode: z.enum(['rules', 'llm']),
  session_name: z.string().nullable().optional(),
});

export const BacktestMetricsSchema = z.object({
  total_return_pct: z.number(),
  sharpe: z.number(),
  max_drawdown_pct: z.number(),
  win_rate: z.number(),
  trades_count: z.number(),
});

export const BacktestResultSchema = z.object({
  metrics: BacktestMetricsSchema,
  equity_curve: z.array(z.object({ ts: z.string(), equity: z.number() })),
  decisions: z.array(z.unknown()),
});

export const BacktestRunStatusSchema = z.object({
  run_id: z.string(),
  params: BacktestParamsSchema,
  status: z.enum(['queued', 'running', 'completed', 'failed', 'canceled']),
  progress: z.number(),
  started_at: z.string(),
  finished_at: z.string().nullable().optional(),
  error: z.string().nullable().optional(),
  result: BacktestResultSchema.nullable().optional(),
});

export const BacktestRunResponseSchema = z.object({
  run_id: z.string(),
});

export const BacktestCancelResponseSchema = z.object({
  canceled: z.boolean(),
});

export const BacktestSessionsListSchema = z.object({
  sessions: z.array(z.string()),
});

export const BacktestSessionDetailSchema = z.object({
  name: z.string(),
  params: z.record(z.unknown()),
  result: z.record(z.unknown()),
  saved_at: z.string(),
});

// ── §5 Risk (matches RiskStatusOut / CircuitBreakerResetOut) ──

export const CircuitBreakerStatusSchema = z.object({
  state: z.enum(['active', 'inactive']),
  triggered_at: z.string().nullable().optional(),
  expires_at: z.string().nullable().optional(),
  reason: z.string().nullable().optional(),
});

export const RiskThresholdsSchema = z.object({
  max_position_pct: z.number(),
  max_daily_loss_pct: z.number(),
  max_stop_loss_pct: z.number(),
  max_trades_per_hour: z.number(),
  max_trades_per_day: z.number(),
  post_loss_cooldown_seconds: z.number(),
});

export const RiskStatusSchema = z.object({
  trade_count_hour: z.number().nullable(),
  trade_count_day: z.number().nullable(),
  circuit_breaker: CircuitBreakerStatusSchema,
  thresholds: RiskThresholdsSchema,
  redis_available: z.boolean(),
});

export const CircuitBreakerResetSchema = z.object({
  success: z.boolean(),
  message: z.string(),
});

// ── §6 Metrics (matches MetricsSummaryV2Response) ──

export const MetricsCountersSchema = z.object({
  trades_total: z.number(),
  orders_placed: z.number(),
  orders_failed: z.number(),
  risk_rejections: z.number(),
  debate_skipped_total: z.number(),
});

export const MetricsPercentilesSchema = z.object({
  pipeline_p50_ms: z.number(),
  pipeline_p95_ms: z.number(),
  execution_p50_ms: z.number(),
  execution_p95_ms: z.number(),
});

export const MetricsSummarySchema = z.object({
  counters: MetricsCountersSchema,
  percentiles: MetricsPercentilesSchema,
  collected_at: z.string(),
});

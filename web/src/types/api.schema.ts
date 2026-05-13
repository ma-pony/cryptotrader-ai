import { z } from 'zod';

// ── Common ──

export const ApiErrorSchema = z.object({
  code: z.string(),
  message: z.string(),
  trace_id: z.string().optional(),
  details: z.record(z.unknown()).optional(),
});
export type ApiError = z.infer<typeof ApiErrorSchema>;

// ── Spec 013: market_type for Pair semantics ──
export const MarketTypeSchema = z.enum(['spot', 'swap', 'future', 'option']);
export type MarketType = z.infer<typeof MarketTypeSchema>;

// ── §1 Portfolio (matches PortfolioSnapshotOut / EquityCurveOut) ──

export const PositionSchema = z.object({
  pair: z.string(), // ccxt canonical: "BTC/USDT" (spot) or "BTC/USDT:USDT" (perp)
  pair_display: z.string(), // spec 013: "BTC/USDT (perp)"
  market_type: MarketTypeSchema.default('spot'),
  side: z.enum(['long', 'short']),
  size: z.number(),
  avg_price: z.number(),
  unrealized_pnl: z.number(),
  unrealized_pnl_pct: z.number(),
  opened_at: z.string().nullable().optional(),
});

export const PnlBreakdownSchema = z.object({
  window: z.string(), // "24h" | "7d" | "30d"
  delta: z.number(),
  realized: z.number(),
  funding: z.number().default(0),
  fees: z.number().default(0),
  unrealized_delta: z.number().default(0),
  exchange_data_available: z.boolean().default(false),
});
export type PnlBreakdown = z.infer<typeof PnlBreakdownSchema>;

export const PortfolioSchema = z.object({
  equity: z.number(),
  cash: z.number(),
  positions: z.array(PositionSchema),
  pnl_24h: z.number(),
  pnl_24h_pct: z.number(),
  drawdown: z.number(),
  updated_at: z.string(),
  // Alignment with frontend prototype (2026-04-24):
  sharpe_90d: z.number().nullable().optional(),
  win_rate: z.number().nullable().optional(),
  total_trades: z.number().default(0),
  realized_pnl_30d: z.number().default(0),
  // Inception-to-date total return (current equity − first snapshot).
  total_return: z.number().default(0),
  total_return_pct: z.number().default(0),
  // Mean realized PnL per filled trade. Null until at least one trade has settled.
  avg_trade_pnl: z.number().nullable().optional(),
  // spec 021: PnL attribution breakdown per window (24h / 7d / 30d).
  pnl_breakdowns: z.array(PnlBreakdownSchema).default([]),
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
  pair: z.string(), // ccxt canonical
  pair_display: z.string().optional(), // spec 013 — fallback to `pair` if absent
  market_type: MarketTypeSchema.default('spot'),
  price: z.number().default(0),
  verdict: VerdictSlimSchema,
  is_filled: z.boolean().default(false),
  trace_id: z.string().nullable().optional(),
  // Alignment with prototype (2026-04-24):
  pnl: z.number().nullable().optional(),
  debate_status: z.string().default(''),
  reject_reason: z.string().nullable().optional(),
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

export const DebateTurnSchema = z.object({
  round: z.number(),
  from: z.string(),
  to: z.string().nullable().optional(),
  before_direction: z.string(),
  before_confidence: z.number(),
  after_direction: z.string(),
  after_confidence: z.number(),
  move: z.string(),
  reasoning: z.string().default(''),
  new_findings: z.string().default(''),
  errored: z.boolean().default(false),
});

export const DebateGateSchema = z.object({
  decision: z.string(),
  reason: z.string().default(''),
  strength: z.number().default(0),
  mean_score: z.number().default(0),
  dispersion: z.number().default(0),
});

export const ConsensusMetricsSchema = z.object({
  strength: z.number().default(0),
  mean_score: z.number().default(0),
  dispersion: z.number().default(0),
  skip_threshold: z.number().default(0.5),
  confusion_threshold: z.number().default(0.05),
});

export const LatencyBreakdownSchema = z.object({
  data_ms: z.number().default(0),
  agents_ms: z.number().default(0),
  debate_ms: z.number().default(0),
  verdict_ms: z.number().default(0),
  risk_ms: z.number().default(0),
  execute_ms: z.number().default(0),
  other_ms: z.number().default(0),
  total_ms: z.number().default(0),
});

export const TokenUsageSchema = z.object({
  input_tokens: z.number().default(0),
  output_tokens: z.number().default(0),
  cache_hits: z.number().default(0),
  calls: z.number().default(0),
  cost_usd: z.number().default(0),
  by_model: z.record(z.record(z.number())).default({}),
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

export const DecisionDetailSchema = z.object({
  commit_hash: z.string(),
  ts: z.string(),
  pair: z.string(), // ccxt canonical
  pair_display: z.string().optional(), // spec 013
  market_type: MarketTypeSchema.default('spot'),
  price: z.number(),
  agent_analyses: z.array(AgentAnalysisSchema),
  debate_rounds: z.array(DebateRoundSchema),
  verdict: VerdictSlimSchema,
  risk_gate: RiskGateSchema,
  execution: ExecutionSchema.nullable(),
  node_timeline: z.array(NodeTimelineEntrySchema),
  trace_id: z.string().nullable().optional(),
  // Alignment with prototype (2026-04-24):
  debate_turns: z.array(DebateTurnSchema).default([]),
  debate_gate: DebateGateSchema.nullable().optional(),
  consensus_metrics: ConsensusMetricsSchema.nullable().optional(),
  latency_breakdown: LatencyBreakdownSchema.default({
    data_ms: 0, agents_ms: 0, debate_ms: 0, verdict_ms: 0,
    risk_ms: 0, execute_ms: 0, other_ms: 0, total_ms: 0,
  }),
  token_usage: TokenUsageSchema.default({
    input_tokens: 0, output_tokens: 0, cache_hits: 0, calls: 0, cost_usd: 0, by_model: {},
  }),
  pnl: z.number().nullable().optional(),
  retrospective: z.string().nullable().optional(),
  debate_skip_reason: z.string().default(''),
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

export const CorrelationGroupSchema = z.object({
  name: z.string(),
  open: z.number(),
  max: z.number(),
  pairs: z.array(z.string()),
});

export const CooldownSchema = z.object({
  pair: z.string(),
  until_seconds: z.number(),
  kind: z.string(),
});

export const RecentBlockSchema = z.object({
  ts: z.string(),
  commit_hash: z.string(),
  rule: z.string(),
  detail: z.string(),
});

export const RiskStatusSchema = z.object({
  trade_count_hour: z.number().nullable(),
  trade_count_day: z.number().nullable(),
  circuit_breaker: CircuitBreakerStatusSchema,
  thresholds: RiskThresholdsSchema,
  redis_available: z.boolean(),
  // Alignment with prototype (2026-04-24):
  daily_loss_pct: z.number().nullable().optional(),
  drawdown_pct: z.number().nullable().optional(),
  total_exposure_pct: z.number().nullable().optional(),
  cvar_95: z.number().nullable().optional(),
  correlation_groups: z.array(CorrelationGroupSchema).default([]),
  cooldowns: z.array(CooldownSchema).default([]),
  recent_blocks: z.array(RecentBlockSchema).default([]),
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

export const LatencyHistogramBucketSchema = z.object({
  upper_bound_s: z.number(),
  count: z.number(),
});

export const DailyCostPointSchema = z.object({
  ts: z.string(),
  cost_usd: z.number(),
});

export const MetricsSummarySchema = z.object({
  counters: MetricsCountersSchema,
  percentiles: MetricsPercentilesSchema,
  collected_at: z.string(),
  // Alignment with prototype (2026-04-24):
  llm_calls_24h: z.number().default(0),
  llm_cost_24h: z.number().default(0),
  cache_hit_rate: z.number().default(0),
  decisions_per_day: z.number().default(0),
  latency_histogram: z.array(LatencyHistogramBucketSchema).default([]),
  cost_14d: z.array(DailyCostPointSchema).default([]),
});

// ── §7 Triggers ──

export const TriggerTypeSchema = z.enum(['price_threshold', 'pct_change', 'candle_pattern', 'funding_rate']);
export const ScheduleRuleSchema = z.object({
  id: z.string(),
  name: z.string(),
  trigger_type: TriggerTypeSchema,
  pair: z.string(),
  parameters: z.record(z.unknown()),
  cooldown_minutes: z.number(),
  enabled: z.boolean(),
  ttl_expires_at: z.string().nullable(),
  created_by: z.string(),
  schedule_depth: z.number(),
  created_at: z.string(),
  updated_at: z.string(),
  in_cooldown: z.boolean(),
  last_triggered_at: z.string().nullable(),
});
export const ScheduleRuleListSchema = z.array(ScheduleRuleSchema);
export const TriggerEventSchema = z.object({
  id: z.string(),
  rule_id: z.string(),
  triggered_at: z.string(),
  trigger_reason: z.string(),
  price_snapshot: z.record(z.unknown()),
  analysis_commit_id: z.string().nullable(),
  schedule_depth: z.number(),
  cooldown_skipped: z.boolean(),
});
export const PaginatedTriggerEventsSchema = z.object({
  items: z.array(TriggerEventSchema),
  total: z.number(),
  page: z.number(),
  size: z.number(),
});

// ── §8 HITL Approvals ──

export const AgentAnalysisSummarySchema = z.object({
  agent: z.string(),
  direction: z.string(),
  confidence: z.number(),
});

export const ApprovalRequestSchema = z.object({
  approval_id: z.string(),
  pair: z.string(),
  created_at: z.string().nullable(),
  expires_at: z.string().nullable(),
  trigger_reason: z.string(),
  verdict_snapshot: z.object({
    action: z.string(),
    position_scale: z.number().optional(),
    confidence: z.number().optional(),
    reasoning: z.string().optional(),
  }),
  agent_analyses_snapshot: z.array(AgentAnalysisSummarySchema),
  status: z.enum(['pending', 'approved', 'rejected', 'expired']),
  decision_by: z.string().nullable(),
  decided_at: z.string().nullable(),
});

export const HitlPendingListSchema = z.array(ApprovalRequestSchema);

export const HitlRespondSchema = z.object({
  approval_id: z.string(),
  status: z.string(),
  message: z.string(),
});

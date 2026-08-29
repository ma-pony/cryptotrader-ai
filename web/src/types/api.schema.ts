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

// ── §3 Trading cycle decisions ──

export const CycleStatusSchema = z.enum([
  'completed',
  'no_change',
  'awaiting_approval',
  'approval_rejected',
  'component_failed',
  'cycle_failed',
  'risk_rejected',
  'execution_failed',
  'cancelled',
]);

export const SignalDirectionSchema = z.enum(['long', 'short', 'neutral']);

export const TargetPositionSchema = z.object({
  side: z.enum(['long', 'short', 'flat']),
  size_ratio: z.number(),
});

export const CommitteeAgentAnalysisSchema = z
  .object({
    agent_id: z.string(),
    pair: z.string().optional(),
    direction: z.enum(['bullish', 'bearish', 'neutral']),
    confidence: z.number(),
    reasoning: z.string(),
    key_factors: z.array(z.string()).default([]),
    risk_flags: z.array(z.string()).default([]),
    data_points: z.record(z.unknown()).default({}),
    data_sufficiency: z.enum(['high', 'medium', 'low']).optional(),
    timestamp: z.string().optional(),
    new_findings: z.string().optional(),
  })
  .strict();

export const CommitteeDebateTurnSchema = z.object({
  round: z.number(),
  from: z.string(),
  to: z.string().nullable(),
  before: z.object({
    direction: z.string(),
    confidence: z.number(),
  }).strict(),
  after: z.object({
    direction: z.string(),
    confidence: z.number(),
  }).strict(),
  move: z.string(),
  reasoning: z.string(),
  new_findings: z.string().default(''),
  errored: z.boolean().default(false),
}).strict();

export const ConsensusMetricsSchema = z
  .object({
    strength: z.number().default(0),
    mean_score: z.number().default(0),
    dispersion: z.number().default(0),
  })
  .strict();

export const CommitteeDetailsSchema = z
  .object({
    analyses: z.record(CommitteeAgentAnalysisSchema).optional(),
    debate_turns: z.array(CommitteeDebateTurnSchema).optional(),
    consensus_metrics: ConsensusMetricsSchema.optional(),
    debate_skipped: z.boolean().optional(),
    debate_skip_reason: z.string().optional(),
  })
  .strict();

export const ComponentDetailsSchema = CommitteeDetailsSchema;

export const ComponentSignalSchema = z.object({
  component_id: z.string(),
  direction: SignalDirectionSchema,
  confidence: z.number(),
  reasoning: z.string(),
  details: ComponentDetailsSchema.default({}),
});

export const ComponentContributionSchema = z.object({
  component_id: z.string(),
  weight: z.number(),
  signed_score: z.number(),
  weighted_score: z.number(),
});

export const FusedSignalSchema = z.object({
  score: z.number(),
  reasoning: z.string(),
  contributions: z.array(ComponentContributionSchema),
});

export const CycleRiskResultSchema = z.object({
  passed: z.boolean(),
  rejected_by: z.string(),
  reason: z.string(),
  cap_source: z.string(),
  target: TargetPositionSchema,
});

// ── §4 Backtest (matches BacktestParams / BacktestRunStatus / sessions) ──

export const BacktestParamsSchema = z.object({
  start: z.string(),
  end: z.string(),
  pair: z.string(),
  initial_capital: z.number(),
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

// ── Signal strategy profile ──

export const ComponentWeightSchema = z.object({
  component_id: z.string(),
  enabled: z.boolean(),
  weight: z.number().min(0).max(1),
});

export const InstalledSignalComponentSchema = z.object({
  component_id: z.string(),
  display_name: z.string(),
  description: z.string(),
});

export const SignalProfileSchema = z.object({
  revision: z.number().int().positive(),
  updated_at: z.string().nullable(),
  components: z.array(ComponentWeightSchema),
  neutral_threshold: z.number().min(0).lt(1),
  max_target_ratio: z.number().gt(0).max(1),
  atr_stop_multiplier: z.number().positive(),
  reward_ratio: z.number().positive(),
  hitl_required: z.boolean(),
  installed_components: z.array(InstalledSignalComponentSchema),
});

// Runtime configuration response DTOs deliberately model JsonEntry values separately
// from the writable document shape.  Secrets are represented only by a configured bit.
export type JsonValueOut = {
  kind: 'null' | 'boolean' | 'number' | 'string' | 'datetime' | 'pair' | 'array' | 'object';
  boolean_value: boolean | null;
  number_value: string | null;
  string_value: string | null;
  datetime_value: string | null;
  pair_value: string | null;
  items: JsonValueOut[];
  entries: Array<{ key: string; value: JsonValueOut }>;
};
export const JsonValueSchema: z.ZodType<JsonValueOut> = z.lazy(() =>
  z
    .object({
      kind: z.enum(['null', 'boolean', 'number', 'string', 'datetime', 'pair', 'array', 'object']),
      boolean_value: z.boolean().nullable(),
      number_value: z.string().nullable(),
      string_value: z.string().nullable(),
      datetime_value: z.string().nullable(),
      pair_value: z.string().nullable(),
      items: z.array(JsonValueSchema),
      entries: z.array(JsonEntrySchema),
    })
    .strict()
    .superRefine((value, ctx) => {
      const scalars = ['boolean_value', 'number_value', 'string_value', 'datetime_value', 'pair_value'] as const;
      const owner = value.kind === 'boolean' ? 'boolean_value' : value.kind === 'number' ? 'number_value' : value.kind === 'string' ? 'string_value' : value.kind === 'datetime' ? 'datetime_value' : value.kind === 'pair' ? 'pair_value' : null;
      for (const field of scalars) {
        if (field === owner ? value[field] === null : value[field] !== null) ctx.addIssue({ code: z.ZodIssueCode.custom, path: [field], message: 'JsonValue envelope field does not match kind' });
      }
      if (value.kind === 'array' ? value.entries.length !== 0 : value.items.length !== 0) ctx.addIssue({ code: z.ZodIssueCode.custom, path: [value.kind === 'array' ? 'entries' : 'items'], message: 'JsonValue envelope container does not match kind' });
      if (value.kind !== 'object' && value.entries.length !== 0) ctx.addIssue({ code: z.ZodIssueCode.custom, path: ['entries'], message: 'JsonValue envelope entries do not match kind' });
    }),
);
export const JsonEntrySchema: z.ZodType<{ key: string; value: JsonValueOut }> = z.lazy(() =>
  z.object({ key: z.string(), value: JsonValueSchema }).strict(),
);
export const RuntimeConnectionSchema = z
  .object({
    id: z.string(),
    label: z.string(),
    adapter_id: z.string(),
    environment: z.enum(['paper', 'demo', 'testnet', 'live']),
    enabled: z.boolean(),
    credential_configured: z.boolean(),
    credential_updated_at: z.string().nullable(),
    leverage: z.number().int(),
    margin_mode: z.enum(['cross', 'isolated']),
    canary_only: z.boolean(),
    parameters: z.array(JsonEntrySchema),
  })
  .strict();
export const RuntimeBookSchema = z
  .object({
    id: z.string(),
    label: z.string(),
    capital_scope: z.enum(['simulated', 'real']),
    enabled: z.boolean(),
    hitl_required: z.boolean(),
    allocations: z.array(z.object({ connection_id: z.string(), enabled: z.boolean(), weight: z.number() }).strict()),
  })
  .strict();
const strictRecord = <T extends z.ZodRawShape>(shape: T) => z.object(shape).strict();
const LlmModelsSchema = strictRecord({
  analysis: z.string(),
  debate: z.string(),
  committee_summary: z.string(),
  tech_agent: z.string(),
  chain_agent: z.string(),
  news_agent: z.string(),
  macro_agent: z.string(),
  fallback: z.string(),
  timeout_seconds: z.number().int(),
});
const RuntimeDocumentSchema = strictRecord({
  system: strictRecord({ active: z.boolean() }),
  security: strictRecord({
    enabled: z.boolean(),
    access_credential_configured: z.boolean(),
    access_credential_updated_at: z.string().nullable(),
  }),
  market_data: strictRecord({ source_id: z.string(), parameters: z.array(JsonEntrySchema) }),
  llm: strictRecord({
    base_url: z.string(),
    streaming_models: z.array(z.string()),
    default_temperature: z.number(),
    timeout: z.number().int(),
    prompt_caching: z.boolean(),
    retry: strictRecord({
      max_attempts: z.number().int(),
      retry_base_delay_s: z.number(),
      retry_backoff_factor: z.number(),
      retry_jitter: z.boolean(),
    }),
    model_costs: z.array(
      strictRecord({ name: z.string(), input_usd_per_mtok: z.number(), output_usd_per_mtok: z.number() }),
    ),
    models: LlmModelsSchema,
    gateway_credential_configured: z.boolean(),
    gateway_credential_updated_at: z.string().nullable(),
  }),
  signals: strictRecord({
    components: z.array(
      strictRecord({
        component_id: z.string(),
        enabled: z.boolean(),
        weight: z.number(),
        parameters: z.array(JsonEntrySchema),
      }),
    ),
    neutral_threshold: z.number(),
    max_target_ratio: z.number(),
    atr_stop_multiplier: z.number(),
    reward_ratio: z.number(),
    hitl_required: z.boolean(),
  }),
  risk: strictRecord({
    max_stop_loss_pct: z.number(),
    position: strictRecord({
      max_single_pct: z.number(),
      max_total_exposure_pct: z.number(),
      max_margin_used_pct: z.number(),
      max_correlated_positions: z.number().int(),
      max_same_direction_positions: z.number().int(),
    }),
    loss: strictRecord({
      max_daily_loss_pct: z.number(),
      max_drawdown_pct: z.number(),
      max_cvar_95: z.number(),
      cvar_min_returns: z.number().int(),
    }),
    cooldown: strictRecord({ same_pair_minutes: z.number().int(), post_loss_minutes: z.number().int() }),
    volatility: strictRecord({
      flash_crash_threshold: z.number(),
      funding_rate_threshold: z.number(),
      flash_crash_lookback: z.number().int(),
    }),
    exchange: strictRecord({ max_api_latency_ms: z.number().int(), health_check_interval_s: z.number().int() }),
    rate_limit: strictRecord({ max_trades_per_hour: z.number().int(), max_trades_per_day: z.number().int() }),
  }),
  execution: strictRecord({
    connections: z.array(RuntimeConnectionSchema),
    books: z.array(RuntimeBookSchema),
    allocation_policy: z.string(),
    live_order_execution_enabled: z.boolean(),
  }),
  hitl: strictRecord({ approval_ttl_minutes: z.number().int() }),
  scheduler: strictRecord({
    enabled: z.boolean(),
    pairs: z.array(z.string()),
    interval_minutes: z.number().int(),
    daily_summary_hour: z.number().int(),
  }),
  triggers: strictRecord({
    enabled: z.boolean(),
    max_rules: z.number().int(),
    ws_reconnect_max_s: z.number().int(),
    funding_rate_poll_interval_minutes: z.number().int(),
  }),
  notifications: strictRecord({
    webhook_url: z.string(),
    enabled: z.boolean(),
    webhook_timeout: z.number().int(),
    events: z.array(z.string()),
    telegram: strictRecord({ enabled: z.boolean(), chat_id: z.string() }),
  }),
  infrastructure: strictRecord({ redis_url: z.string() }),
  observability: strictRecord({ otlp_endpoint: z.string() }),
});
export const RuntimeConfigSchema = strictRecord({
  revision: z.number().int(),
  updated_at: z.string(),
  setup_required: z.boolean(),
  apply_status: z.enum(['pending', 'applied', 'failed']),
  applied_revision: z.number().int().nullable(),
  apply_error: z.string().nullable(),
  document: RuntimeDocumentSchema,
});
export const VenueMutationSchema = strictRecord({ revision: z.number().int(), connection: RuntimeConnectionSchema });
export const CredentialMutationSchema = strictRecord({
  revision: z.number().int(),
  credential: strictRecord({ configured: z.boolean(), updated_at: z.string().nullable() }),
});
export const RuntimeTokenMutationSchema = strictRecord({
  revision: z.number().int(),
  configured: z.boolean(),
  updated_at: z.string(),
});
export const ConnectionHealthSchema = strictRecord({
  connection_id: z.string(),
  healthy: z.boolean(),
  environment: z.enum(['paper', 'demo', 'testnet', 'live']),
  credential_configured: z.boolean(),
  capabilities: strictRecord({
    market_types: z.array(z.string()),
    native_protection: z.boolean(),
    hedge_mode: z.boolean(),
    reduce_only: z.boolean(),
    supported_order_types: z.array(z.string()),
  }),
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
  cycle_id: z.string(),
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

const PairSymbolSchema = z.object({ symbol: z.string() }).strict();
const ConnectionPlanSchema = z.object({
  book_id: z.string(), connection_id: z.string(), pair: PairSymbolSchema,
  current_signed_notional: z.string(), target_signed_notional: z.string(), delta_signed_notional: z.string(),
  current_signed_amount: z.string(), target_signed_amount: z.string(), delta_signed_amount: z.string(),
  post_fill_signed_amount: z.string(), quote: z.object({ pair: PairSymbolSchema, bid: z.string(), ask: z.string(), last: z.string() }).strict(),
  execution_price: z.string(), amount: z.string(), side: z.string(), reduce_only: z.boolean(), market_type: z.string(),
  stop_loss: z.string().nullable(), take_profit: z.string().nullable(), old_protection_ids: z.array(z.string()),
  capabilities: z.object({ market_types: z.array(z.string()), native_protection: z.boolean(), hedge_mode: z.boolean(), reduce_only: z.boolean(), supported_order_types: z.array(z.string()) }).strict(),
}).strict();
const BookProposalSchema = z.object({
  version: z.literal(1), book_id: z.string(), capital_scope: z.enum(['simulated', 'real']), config_revision: z.number().int(), pair: PairSymbolSchema,
  requested_target_exposure: z.string(), target_exposure: z.string(),
  risk: z.object({ passed: z.boolean(), requested_target_exposure: z.string(), capped_target_exposure: z.string(), connection_weights: z.array(z.string()), connection_targets: z.array(z.object({ book_id: z.string(), connection_id: z.string(), weight: z.string(), book_equity: z.string(), target_exposure: z.string(), target_signed_notional: z.string() }).strict()), rejected_by: z.string(), reason: z.string(), cap_source: z.string() }).strict(),
  connection_risks: z.array(z.object({ connection_id: z.string(), passed: z.boolean(), risk_increase: z.boolean(), reason: z.string(), operation: z.string() }).strict()),
  connection_plans: z.array(ConnectionPlanSchema), unavailable_connections: z.array(z.string()), errors: z.array(z.string()), ready: z.boolean(),
}).strict();
export const ApprovalRequestSchema = z.object({ approval_id: z.string(), cycle_id: z.string(), book_id: z.string(), pair: z.string(), config_revision: z.number().int(), proposal: BookProposalSchema, status: z.enum(['pending', 'approved', 'rejected', 'invalidated', 'executed']), created_at: z.string(), decided_at: z.string().nullable() }).strict();

export const HitlPendingListSchema = z.array(ApprovalRequestSchema);

export const HitlRespondSchema = z.object({
  approval_id: z.string(),
  cycle_id: z.string(),
  approval_status: z.string(),
  cycle_status: z.string(),
  execution_status: z.string(),
  requires_attention: z.boolean(),
}).strict();

const ConnectionPortfolioSchema = z.object({ connection_id: z.string(), equity: z.string(), balances: z.array(z.object({ asset: z.string(), amount: z.string() }).strict()), position: z.object({ pair: z.string(), signed_amount: z.string(), signed_notional: z.string(), entry_price: z.string().nullable() }).strict() }).strict();
const PortfolioBookSchema = z.object({ book_id: z.string(), capital_scope: z.enum(['simulated', 'real']), pair: z.string(), total_equity: z.string(), total_signed_notional: z.string(), connections: z.array(ConnectionPortfolioSchema) }).strict();
export const PortfolioBooksSchema = z.object({ pair: z.string(), simulated: z.object({ books: z.array(PortfolioBookSchema), totals: z.object({ equity: z.string(), signed_notional: z.string() }).strict() }).strict(), real: z.object({ books: z.array(PortfolioBookSchema), totals: z.object({ equity: z.string(), signed_notional: z.string() }).strict() }).strict() }).strict();
const ConnectionRiskSchema = z.object({ connection_id: z.string(), passed: z.boolean(), risk_increase: z.boolean(), reason: z.string(), operation: z.string() }).strict();
const NormalizedOrderSchema = z.object({ id: z.string(), pair: PairSymbolSchema, side: z.string(), order_type: z.string(), amount: z.string(), filled_amount: z.string(), average_price: z.string().nullable(), status: z.string(), reduce_only: z.boolean() }).strict();
const ProtectionStateSchema = z.object({ protection_ids: z.array(z.string()), pair: PairSymbolSchema, position_side: z.string(), amount: z.string(), stop_loss: z.string().nullable(), take_profit: z.string().nullable(), active: z.boolean(), triggered: z.boolean() }).strict();
const FinalPositionSchema = z.object({ position: z.object({ pair: PairSymbolSchema, signed_amount: z.string(), signed_notional: z.string(), entry_price: z.string().nullable() }).strict(), protected: z.boolean(), protection_ids: z.array(z.string()), protections: z.array(ProtectionStateSchema) }).strict();
const CompensationSchema = z.object({ attempted: z.boolean(), succeeded: z.boolean(), order: NormalizedOrderSchema.nullable(), operation: z.string(), safe_signed_amount: z.string().nullable(), required_protection: ProtectionStateSchema.nullable() }).strict();
const ConnectionExecutionSchema = z.object({ book_id: z.string(), connection_id: z.string(), pair: PairSymbolSchema, target_signed_notional: z.string(), target_signed_amount: z.string(), status: z.string(), orders: z.array(NormalizedOrderSchema), protection: ProtectionStateSchema.nullable(), compensation: CompensationSchema, final_position: FinalPositionSchema.nullable(), error_operation: z.string(), requires_attention: z.boolean(), trace: z.array(z.string()), execution_quote: z.object({ pair: PairSymbolSchema, bid: z.string(), ask: z.string(), last: z.string() }).strict().nullable() }).strict();
const CycleConnectionSchema = z.object({ connection_id: z.string(), portfolio_before: ConnectionPortfolioSchema.nullable(), portfolio_after: ConnectionPortfolioSchema.nullable(), risk: ConnectionRiskSchema.nullable(), plan: ConnectionPlanSchema.nullable(), execution: ConnectionExecutionSchema.nullable(), unavailable: z.boolean() }).strict();
const BookRiskSchema = z.object({ passed: z.boolean(), requested_target_exposure: z.string(), capped_target_exposure: z.string(), connection_weights: z.array(z.string()), connection_targets: z.array(z.object({ book_id: z.string(), connection_id: z.string(), weight: z.string(), book_equity: z.string(), target_exposure: z.string(), target_signed_notional: z.string() }).strict()), rejected_by: z.string(), reason: z.string(), cap_source: z.string() }).strict();
const CycleBookSchema = z.object({ book_id: z.string(), capital_scope: z.enum(['simulated', 'real']), config_revision: z.number().int(), pair: z.string(), market_type: z.string(), status: z.string(), hitl: z.object({ approval_id: z.string().nullable(), status: z.string(), config_revision: z.number().int() }).strict(), failure: z.object({ stage: z.string() }).strict().nullable(), requested_target_exposure: z.string().nullable(), target_exposure: z.string().nullable(), risk: BookRiskSchema.nullable(), ready: z.boolean().nullable(), errors: z.array(z.string()), execution: z.object({ status: z.string(), requires_attention: z.boolean(), reallocated: z.boolean() }).strict().nullable(), portfolio_before: PortfolioBookSchema.nullable(), portfolio_after: PortfolioBookSchema.nullable(), portfolio_after_available: z.boolean().nullable(), connections: z.array(CycleConnectionSchema) }).strict();
export const CycleSchema = z.object({ cycle_id: z.string(), config_revision: z.number().int(), market_data_source_id: z.string(), shared_signals: z.object({ components: z.array(z.object({ component_id: z.string(), direction: z.string(), confidence: z.number(), reasoning: z.string(), details: z.array(JsonEntrySchema) }).strict()), fused: z.object({ score: z.number(), reasoning: z.string(), contributions: z.array(z.object({ component_id: z.string(), weight: z.number(), signed_score: z.number(), weighted_score: z.number() }).strict()) }).strict().nullable(), target_position: z.object({ side: z.string(), size_ratio: z.number() }).strict().nullable() }).strict(), books: z.array(CycleBookSchema), cycle_status: z.string(), execution_status: z.string(), requires_attention: z.boolean(), created_at: z.string() }).strict();
export const PaginatedCyclesSchema = z.object({ items: z.array(CycleSchema), total: z.number().int(), page: z.number().int(), size: z.number().int(), has_next: z.boolean() }).strict();

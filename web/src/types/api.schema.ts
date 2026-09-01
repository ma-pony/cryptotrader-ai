import { z } from 'zod';

export const AccountOperationAcceptedSchema = z
  .object({
    operation_id: z.string(),
    status: z.enum(['preparing', 'executing']),
  })
  .strict();
export const AccountOperationSchema = z
  .object({
    operation_id: z.string(),
    connection_id: z.string(),
    pair: z.string(),
    kind: z.enum(['cancel_orders', 'flatten']),
    status: z.enum(['preparing', 'awaiting_confirmation', 'executing', 'completed', 'failed', 'invalidated']),
    created_at: z.string(),
    updated_at: z.string(),
    plan: z
      .object({
        operation_id: z.string(),
        version: z.number(),
        connection_id: z.string(),
        book_id: z.string().nullable(),
        capital_scope: z.enum(['simulated', 'real']),
        pair: z.string(),
        kind: z.enum(['cancel_orders', 'flatten']),
        stopped_scope: z.array(z.string()),
        ordinary_order_ids: z.array(z.string()),
        position_amount: z.string(),
        close_amount: z.string(),
        protection_ids: z.array(z.string()),
        snapshot_time: z.string(),
      })
      .strict()
      .nullable(),
    result: z
      .object({
        canceled_order_ids: z.array(z.string()),
        canceled_protection_ids: z.array(z.string()),
        orders: z.array(
          z
            .object({
              id: z.string(),
              pair: z.string(),
              side: z.string(),
              order_type: z.string(),
              amount: z.string(),
              filled_amount: z.string(),
              average_price: z.string().nullable(),
              status: z.string(),
              reduce_only: z.boolean(),
              client_order_id: z.string().nullable(),
            })
            .strict(),
        ),
        remaining_position: z.string().nullable(),
        remaining_order_ids: z.array(z.string()),
        remaining_protection_ids: z.array(z.string()),
        observed_at: z.string().nullable(),
        failure_reason: z.string().nullable(),
        reconciliation_required: z.boolean(),
      })
      .strict(),
  })
  .strict();

// ── Common ──

export const ApiErrorSchema = z
  .object({
    code: z.string(),
    message: z.string(),
    trace_id: z.string().optional(),
    details: z
      .object({
        fieldErrors: z.record(z.string(), z.string()),
      })
      .strict()
      .optional(),
  })
  .strict();
export type ApiError = z.infer<typeof ApiErrorSchema>;

// ── Spec 013: market_type for Pair semantics ──
export const MarketTypeSchema = z.enum(['spot', 'swap', 'future', 'option']);
export type MarketType = z.infer<typeof MarketTypeSchema>;

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

export const CommitteeDebateTurnSchema = z
  .object({
    round: z.number(),
    from: z.string(),
    to: z.string().nullable(),
    before: z
      .object({
        direction: z.string(),
        confidence: z.number(),
      })
      .strict(),
    after: z
      .object({
        direction: z.string(),
        confidence: z.number(),
      })
      .strict(),
    move: z.string(),
    reasoning: z.string(),
    new_findings: z.string().default(''),
    errored: z.boolean().default(false),
  })
  .strict();

export const ConsensusMetricsSchema = z
  .object({
    strength: z.number().default(0),
    mean_score: z.number().default(0),
    dispersion: z.number().default(0),
  })
  .strict();

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

export type RawJsonValue = null | boolean | number | string | RawJsonValue[] | { [key: string]: RawJsonValue };
export const RawJsonValueSchema: z.ZodType<RawJsonValue> = z.lazy(() =>
  z.union([
    z.null(),
    z.boolean(),
    z.number().finite(),
    z.string(),
    z.array(RawJsonValueSchema),
    z.record(z.string(), RawJsonValueSchema),
  ]),
);
const RawJsonObjectSchema = z.record(z.string(), RawJsonValueSchema);

export const BacktestParamsSchema = z
  .object({
    start: z.string(),
    end: z.string(),
    pair: z.string(),
    interval: z.string().nullable(),
    initial_equity: z.string().nullable(),
    fee_rate: z.string().nullable(),
    slippage_bps: z.string().nullable(),
    funding_assumption: z.enum(['available_only', 'disabled']).nullable(),
    name: z.string().nullable().optional(),
    snapshot_run_id: z.string().nullable().optional(),
  })
  .strict();

export const BacktestSnapshotSchema = z
    .object({
      version: z.literal(1),
      revision: z.number(),
      updated_at: z.string(),
      market_data: z
        .object({ source_id: z.string(), timeframe: z.string(), parameters: RawJsonObjectSchema })
        .strict(),
      signals: z
        .object({
          components: z.array(
            z
              .object({
                component_id: z.string(),
                enabled: z.boolean(),
                weight: z.number(),
                parameters: RawJsonObjectSchema,
                model_identity: RawJsonObjectSchema,
              })
              .strict(),
          ),
          neutral_threshold: z.number(),
          max_target_ratio: z.number(),
          evaluation_interval: z.string().nullable(),
          atr_stop_multiplier: z.number(),
          reward_ratio: z.number(),
        })
        .strict(),
      risk: z
        .object({
          position: z
            .object({ max_single_pct: z.number(), max_total_exposure_pct: z.number(), max_margin_used_pct: z.number() })
            .strict(),
          loss: z.object({ max_drawdown_pct: z.number() }).strict(),
        })
        .strict(),
      llm: z
        .object({
          streaming_models: z.array(z.string()),
          default_temperature: z.number(),
          timeout: z.number(),
          prompt_caching: z.boolean(),
          retry: z
            .object({
              max_attempts: z.number(),
              retry_base_delay_s: z.number(),
              retry_backoff_factor: z.number(),
              retry_jitter: z.boolean(),
            })
            .strict(),
          model_costs: z.array(
            z.object({ name: z.string(), input_usd_per_mtok: z.number(), output_usd_per_mtok: z.number() }).strict(),
          ),
          models: z.record(z.string(), z.union([z.string(), z.number()])),
        })
        .strict(),
    })
    .strict()
    .nullable();

export const BacktestModelEvidenceSchema = z
  .object({
    requested_model: z.string().nullable(),
    actual_model: z.string().nullable(),
    actual_model_reason: z.string().nullable(),
    prompt_hash: z.string(),
    prompt_version: z.string(),
    status: z.enum(['started', 'completed', 'failed']),
  })
  .strict();

export const BacktestMetricsSchema = z
  .object({
    total_return_pct: z.number(),
    sharpe: z.number(),
    max_drawdown_pct: z.number(),
    win_rate: z.number().nullable(),
    fill_count: z.number().int().nonnegative(),
    closed_trade_count: z.number().int().nonnegative(),
  })
  .strict();

export const BacktestEquityPointSchema = z.object({ ts: z.string(), equity: z.number() }).strict();

const BacktestDecisionSummarySchema = z
  .object({ cycle_id: z.string(), status: z.string(), config_revision: z.number().int().nonnegative() })
  .strict();

export const BacktestResultSchema = z
  .object({
    metrics: BacktestMetricsSchema,
    equity_curve: z.array(BacktestEquityPointSchema),
    decisions: z.array(z.union([z.lazy(() => DecisionSchema), BacktestDecisionSummarySchema])),
    decision_ids: z.array(z.string()),
    fills: z.array(z.lazy(() => AccountFillsSchema.shape.items.element.omit({ attribution: true }))),
    closed_trades: z.array(
      z
        .object({
          pair: z.string(),
          opened_at: z.string(),
          closed_at: z.string(),
          side: z.enum(['long', 'short']),
          gross_pnl: z.string(),
          fees: z.string(),
          funding: z.string(),
          net_pnl: z.string(),
          fill_ids: z.array(z.string()),
        })
        .strict(),
    ),
    fees: z.string(),
    funding: z.string(),
    funding_entries: z.array(z.lazy(() => AccountIncomeSchema.shape.items.element)),
    cost_assumptions: RawJsonObjectSchema,
    unmodeled_costs: z.array(z.string()),
    data_coverage: RawJsonObjectSchema,
  })
  .strict();

export const BacktestRunStatusSchema = z
  .object({
    run_id: z.string(),
    params: BacktestParamsSchema,
    config_snapshot: BacktestSnapshotSchema,
    model_evidence: z.array(BacktestModelEvidenceSchema),
    incomplete_fields: z.array(z.string()),
    status: z.enum(['queued', 'running', 'completed', 'failed', 'canceled', 'interrupted']),
    progress: z.number(),
    started_at: z.string(),
    finished_at: z.string().nullable().optional(),
    error: z.string().nullable().optional(),
    result: BacktestResultSchema.nullable().optional(),
  })
  .strict();

export const BacktestRunsSchema = z
  .object({ items: z.array(BacktestRunStatusSchema), limit: z.number(), offset: z.number(), has_next: z.boolean() })
  .strict();
export const BacktestComparisonSchema = z
  .object({
    comparable: z.boolean(),
    condition_differences: z.record(
      z.string(),
      z.object({ left: RawJsonValueSchema, right: RawJsonValueSchema, reason: z.string().nullable().optional() }).strict(),
    ),
    configuration_differences: z.record(
      z.string(),
      z.object({ left: RawJsonValueSchema, right: RawJsonValueSchema, reason: z.string().nullable().optional() }).strict(),
    ),
    left: BacktestRunStatusSchema,
    right: BacktestRunStatusSchema,
  })
  .strict();

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
      const owner =
        value.kind === 'boolean'
          ? 'boolean_value'
          : value.kind === 'number'
            ? 'number_value'
            : value.kind === 'string'
              ? 'string_value'
              : value.kind === 'datetime'
                ? 'datetime_value'
                : value.kind === 'pair'
                  ? 'pair_value'
                  : null;
      for (const field of scalars) {
        if (field === owner ? value[field] === null : value[field] !== null)
          ctx.addIssue({
            code: z.ZodIssueCode.custom,
            path: [field],
            message: 'JsonValue envelope field does not match kind',
          });
      }
      if (value.kind === 'array' ? value.entries.length !== 0 : value.items.length !== 0)
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          path: [value.kind === 'array' ? 'entries' : 'items'],
          message: 'JsonValue envelope container does not match kind',
        });
      if (value.kind !== 'object' && value.entries.length !== 0)
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          path: ['entries'],
          message: 'JsonValue envelope entries do not match kind',
        });
    }),
);
export const JsonEntrySchema: z.ZodType<{ key: string; value: JsonValueOut }> = z.lazy(() =>
  z.object({ key: z.string(), value: JsonValueSchema }).strict(),
);
const LocalizedTextSchema = z.object({ zh_CN: z.string(), en_US: z.string() }).strict();
export const ConfigurationFieldSchema = z
  .object({
    key: z.string(),
    label: LocalizedTextSchema,
    description: LocalizedTextSchema,
    kind: z.enum(['text', 'number', 'integer', 'boolean', 'select', 'string_list']),
    default_value: JsonValueSchema,
    required: z.boolean(),
    minimum: z.number().nullable(),
    maximum: z.number().nullable(),
    step: z.number().nullable(),
    unit: z.string().nullable(),
    exclusive_minimum: z.number().nullable(),
    exclusive_maximum: z.number().nullable(),
    advanced: z.boolean(),
    options: z.array(z.object({ value: z.string(), label: LocalizedTextSchema }).strict()),
  })
  .strict();
export const ConfigurationEnvironmentSchema = z
  .object({
    id: z.string(),
    label: LocalizedTextSchema,
    capital_scope: z.enum(['simulated', 'real']),
  })
  .strict();
export const CredentialFieldSchema = z
  .object({
    key: z.string(),
    label: LocalizedTextSchema,
    description: LocalizedTextSchema,
    required: z.boolean(),
  })
  .strict();
export const PluginDefinitionSchema = z
  .object({
    id: z.string(),
    label: LocalizedTextSchema,
    description: LocalizedTextSchema,
    fields: z.array(ConfigurationFieldSchema),
    environments: z.array(ConfigurationEnvironmentSchema),
    credential_fields: z.array(CredentialFieldSchema),
    margin_modes: z.array(z.string()),
  })
  .strict();
export const ConfigurationCatalogSchema = z
  .object({
    components: z.array(PluginDefinitionSchema),
    venues: z.array(PluginDefinitionSchema),
    market_sources: z.array(PluginDefinitionSchema),
  })
  .strict();
export const VenueEnvironmentDefinitionSchema = z
  .object({
    id: z.string(),
    label: LocalizedTextSchema,
    description: LocalizedTextSchema,
    environment: ConfigurationEnvironmentSchema,
    fields: z.array(ConfigurationFieldSchema),
    credential_fields: z.array(CredentialFieldSchema),
    margin_modes: z.array(z.string()),
    leverage_minimum: z.number().int(),
    leverage_maximum: z.number().int().nullable(),
    account_read: z.boolean(),
    capabilities: z
      .object({
        market_types: z.array(z.string()),
        native_protection: z.boolean(),
        hedge_mode: z.boolean(),
        reduce_only: z.boolean(),
        supported_order_types: z.array(z.string()),
        account_reads: z.array(z.string()),
        exit_operations: z.array(z.string()),
        history_initial_days: z.number().int().positive().nullable(),
        unknown_fields: z.array(z.enum(['account_reads', 'exit_operations', 'history_initial_days'])),
      })
      .strict()
      .nullable(),
  })
  .strict();
export const RuntimeConnectionSchema = z
  .object({
    id: z.string(),
    label: z.string(),
    adapter_id: z.string(),
    environment: z.string(),
    enabled: z.boolean(),
    credential_configured: z.boolean(),
    credential_updated_at: z.string().nullable(),
    leverage: z.number().int(),
    margin_mode: z.string(),
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
export const NotificationEventSchema = z.enum([
  'approval_pending',
  'execution_failed',
  'protection_failed',
  'risk_adjusted',
  'component_failed',
  'connection_failed',
  'daily_summary',
]);

const RuntimeDocumentSchema = strictRecord({
  security: strictRecord({
    enabled: z.boolean(),
    access_credential_configured: z.boolean(),
    access_credential_updated_at: z.string().nullable(),
  }),
  market_data: strictRecord({
    source_id: z.string(),
    timeframe: z.string(),
    parameters: z.array(JsonEntrySchema),
    news_credential_configured: z.boolean(),
    news_credential_updated_at: z.string().nullable(),
  }),
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
    evaluation_interval: z.string().nullable(),
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
  }),
  risk: strictRecord({
    position: strictRecord({
      max_single_pct: z.number(),
      max_total_exposure_pct: z.number(),
      max_margin_used_pct: z.number(),
    }),
    loss: strictRecord({ max_drawdown_pct: z.number() }),
  }),
  execution: strictRecord({
    pairs: z.array(z.string()),
    connections: z.array(RuntimeConnectionSchema),
    books: z.array(RuntimeBookSchema),
    live_order_execution_enabled: z.boolean(),
  }),
  hitl: strictRecord({ approval_ttl_minutes: z.number().int() }),
  accounts: strictRecord({ sync_interval_seconds: z.number().int().min(1).max(86400) }),
  scheduler: strictRecord({
    automation_enabled: z.boolean(),
    enabled: z.boolean(),
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
    events: z.array(NotificationEventSchema),
  }),
  infrastructure: strictRecord({ redis_url: z.string() }),
  observability: strictRecord({ otlp_endpoint: z.string() }),
});
export const AlertSchema = strictRecord({
  id: z.string(), event_key: z.string(), type: NotificationEventSchema, occurred_at: z.string(), capital_scope: z.enum(['simulated', 'real']).nullable(), decision_id: z.string().nullable(), book_id: z.string().nullable(), connection_id: z.string().nullable(), operation_id: z.string().nullable(), pair: z.string().nullable(), message: z.string(), read_at: z.string().nullable(), resolution: z.enum(['open', 'applied', 'approval_processed', 'expired', 'recovered', 'exit_completed', 'informational']), resolved_at: z.string().nullable(),
});
export const DeliverySchema = strictRecord({ id: z.string(), alert_id: z.string(), channel: z.literal('webhook'), status: z.enum(['pending', 'sending', 'failed', 'delivered']), attempts: z.number().int(), last_error: z.string().nullable(), last_attempt_at: z.string().nullable(), delivered_at: z.string().nullable() });
export const AlertListSchema = strictRecord({ items: z.array(AlertSchema), total: z.number().int() });
export const AlertOverviewSchema = strictRecord({ alerts: z.array(AlertSchema), deliveries: z.array(DeliverySchema) });

export const RuntimeConfigSchema = strictRecord({
  revision: z.number().int(),
  updated_at: z.string(),
  apply_status: z.enum(['pending', 'applied', 'failed']),
  applied_revision: z.number().int().nullable(),
  apply_error: z.string().nullable(),
  document: RuntimeDocumentSchema,
});

const ReadinessReasonSchema = strictRecord({ code: z.string(), message: z.string(), path: z.string() });
const CapabilitySchema = strictRecord({ ready: z.boolean(), reasons: z.array(ReadinessReasonSchema) });
export const ReadinessSchema = strictRecord({
  analysis: CapabilitySchema,
  trading: CapabilitySchema,
  components: z.array(
    strictRecord({
      component_id: z.string(),
      enabled: z.boolean(),
      ready: z.boolean(),
      reasons: z.array(ReadinessReasonSchema),
      dependencies: z.array(
        strictRecord({
          kind: z.enum(['market', 'model_service', 'local_artifact', 'context']),
          key: z.string(),
          label: z.string(),
          ready: z.boolean(),
          reasons: z.array(ReadinessReasonSchema),
        }),
      ),
    }),
  ),
  saved_revision: z.number().int(),
  applied_revision: z.number().int().nullable(),
  apply_error: z.string().nullable(),
  automation_enabled: z.boolean(),
  latest_run_at: z.string().nullable(),
  execution_pairs: z.array(z.string()),
});
export const TradingScopeSchema = strictRecord({
  pair: z.string(),
  saved_revision: z.number().int(),
  ready: z.boolean(),
  reasons: z.array(ReadinessReasonSchema),
  books: z.array(
    strictRecord({
      book_id: z.string(),
      label: z.string(),
      capital_scope: z.enum(['simulated', 'real']),
      enabled: z.boolean(),
      eligible: z.boolean(),
      reasons: z.array(ReadinessReasonSchema),
      hitl_required: z.boolean(),
      connections: z.array(
        strictRecord({ connection_id: z.string(), label: z.string(), environment: z.string(), enabled: z.boolean() }),
      ),
    }),
  ),
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
  checked_at: z.string(),
  healthy: z.boolean(),
  environment: z.string(),
  credential_configured: z.boolean(),
  error_code: z.string().nullable(),
  capabilities: strictRecord({
    market_types: z.array(z.string()),
    native_protection: z.boolean(),
    hedge_mode: z.boolean(),
    reduce_only: z.boolean(),
    supported_order_types: z.array(z.string()),
    account_reads: z.array(z.string()),
    exit_operations: z.array(z.string()),
    history_initial_days: z.number().int().positive().nullable(),
    unknown_fields: z.array(z.enum(['account_reads', 'exit_operations', 'history_initial_days'])),
  }).nullable(),
});

export const BacktestRunResponseSchema = z
  .object({
    run_id: z.string(),
    status: z.literal('queued'),
  })
  .strict();

export const BacktestCancelResponseSchema = z.object({
  canceled: z.boolean(),
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
const ScheduleRuleBaseSchema = z
  .object({
    id: z.string(),
    name: z.string(),
    pair: z.string(),
    cooldown_minutes: z.number(),
    enabled: z.boolean(),
    ttl_expires_at: z.string().nullable(),
    created_by: z.string(),
    schedule_depth: z.number(),
    created_at: z.string(),
    updated_at: z.string(),
    in_cooldown: z.boolean(),
    last_triggered_at: z.string().nullable(),
  })
  .strict();
const PriceParametersSchema = z.object({ direction: z.enum(['above', 'below']), price: z.number().positive() }).strict();
const ChangeParametersSchema = z.object({ window_minutes: z.number().int().min(1), threshold_pct: z.number().positive() }).strict();
const CandleParametersSchema = z
  .object({
    interval: z.string().min(1),
    consecutive_count: z.number().int().min(1),
    direction: z.enum(['bearish', 'bullish']),
  })
  .strict();
const FundingParametersSchema = z.object({ threshold_pct: z.number().positive() }).strict();
export const ScheduleRuleSchema = z.discriminatedUnion('trigger_type', [
  ScheduleRuleBaseSchema.extend({
    trigger_type: z.literal('price_threshold'),
    parameters: PriceParametersSchema,
  }),
  ScheduleRuleBaseSchema.extend({
    trigger_type: z.literal('pct_change'),
    parameters: ChangeParametersSchema,
  }),
  ScheduleRuleBaseSchema.extend({
    trigger_type: z.literal('candle_pattern'),
    parameters: CandleParametersSchema,
  }),
  ScheduleRuleBaseSchema.extend({
    trigger_type: z.literal('funding_rate'),
    parameters: FundingParametersSchema,
  }),
]);
export const ScheduleRuleListSchema = z.array(ScheduleRuleSchema);
export const TriggerEventSchema = z
  .object({
    id: z.string(),
    rule_id: z.string(),
    triggered_at: z.string(),
    trigger_reason: z.string(),
    price_snapshot: z.object({ pair: z.string().min(1), price: z.number(), ts: z.number() }).strict(),
    analysis_commit_id: z.string().nullable(),
    schedule_depth: z.number(),
    cooldown_skipped: z.boolean(),
  })
  .strict();
export const PaginatedTriggerEventsSchema = z.object({
  items: z.array(TriggerEventSchema),
  total: z.number(),
  page: z.number(),
  size: z.number(),
});

// ── §8 HITL Approvals ──

const PairSymbolSchema = z.object({ symbol: z.string() }).strict();
const ConnectionPlanSchema = z
  .object({
    book_id: z.string(),
    connection_id: z.string(),
    pair: PairSymbolSchema,
    current_signed_notional: z.string(),
    target_signed_notional: z.string(),
    delta_signed_notional: z.string(),
    current_signed_amount: z.string(),
    target_signed_amount: z.string(),
    delta_signed_amount: z.string(),
    post_fill_signed_amount: z.string(),
    quote: z.object({ pair: PairSymbolSchema, bid: z.string(), ask: z.string(), last: z.string() }).strict(),
    execution_price: z.string(),
    amount: z.string(),
    side: z.string(),
    reduce_only: z.boolean(),
    market_type: z.string(),
    stop_loss: z.string().nullable(),
    take_profit: z.string().nullable(),
    old_protection_ids: z.array(z.string()),
    capabilities: z
      .object({
        market_types: z.array(z.string()),
        native_protection: z.boolean(),
        hedge_mode: z.boolean(),
        reduce_only: z.boolean(),
        supported_order_types: z.array(z.string()),
        account_reads: z.array(z.string()),
        exit_operations: z.array(z.string()),
        history_initial_days: z.number().int().positive().nullable(),
        unknown_fields: z.array(z.enum(['account_reads', 'exit_operations', 'history_initial_days'])),
      })
      .strict(),
  })
  .strict();
export const BookRiskStateSchema = z
  .object({
    book_id: z.string(),
    capital_scope: z.enum(['simulated', 'real']),
    valuation_currency: z.string(),
    observed_at: z.string(),
    equity: z.string().nullable(),
    peak_equity: z.string().nullable(),
    positions_by_instrument: z.array(
      z.object({ instrument: z.string(), signed_notional: z.string().nullable() }).strict(),
    ),
    pending_increase_notional: z.string().nullable(),
    gross_notional: z.string().nullable(),
    net_notional: z.string().nullable(),
    used_margin: z.string().nullable(),
    available_margin: z.string().nullable(),
    completeness: z.array(z.string()),
  })
  .strict();
const BookProposalSchema = z
  .object({
    version: z.literal(1),
    book_id: z.string(),
    capital_scope: z.enum(['simulated', 'real']),
    config_revision: z.number().int(),
    pair: PairSymbolSchema,
    requested_target_exposure: z.string(),
    target_exposure: z.string(),
    risk: z
      .object({
        passed: z.boolean(),
        requested_target_exposure: z.string(),
        capped_target_exposure: z.string(),
        connection_weights: z.array(z.string()),
        connection_targets: z.array(
          z
            .object({
              book_id: z.string(),
              connection_id: z.string(),
              weight: z.string(),
              book_equity: z.string().nullable(),
              target_exposure: z.string(),
              target_signed_notional: z.string(),
            })
            .strict(),
        ),
        rejected_by: z.string(),
        reason: z.string(),
        cap_source: z.string(),
        state: BookRiskStateSchema.nullable(),
      })
      .strict(),
    connection_risks: z.array(
      z
        .object({
          connection_id: z.string(),
          passed: z.boolean(),
          risk_increase: z.boolean(),
          reason: z.string(),
          operation: z.string(),
        })
        .strict(),
    ),
    connection_plans: z.array(ConnectionPlanSchema),
    unavailable_connections: z.array(z.string()),
    errors: z.array(z.string()),
    ready: z.boolean(),
  })
  .strict();
export const ApprovalRequestSchema = z
  .object({
    approval_id: z.string(),
    cycle_id: z.string(),
    book_id: z.string(),
    pair: z.string(),
    config_revision: z.number().int(),
    proposal: BookProposalSchema,
    status: z.enum(['pending', 'approved', 'rejected', 'invalidated', 'executed']),
    created_at: z.string(),
    decided_at: z.string().nullable(),
  })
  .strict();

export const HitlPendingListSchema = z.array(ApprovalRequestSchema);

export const HitlRespondSchema = z
  .object({
    approval_id: z.string(),
    cycle_id: z.string(),
    approval_status: z.string(),
    cycle_status: z.string(),
    execution_status: z.string(),
    requires_attention: z.boolean(),
  })
  .strict();

const ConnectionPortfolioSchema = z
  .object({
    connection_id: z.string(),
    equity: z.string().nullable(),
    balances: z.array(z.object({ asset: z.string(), amount: z.string() }).strict()),
    position: z
      .object({
        pair: z.string(),
        signed_amount: z.string(),
        signed_notional: z.string(),
        entry_price: z.string().nullable(),
      })
      .strict(),
  })
  .strict();
const PortfolioBookSchema = z
  .object({
    book_id: z.string(),
    capital_scope: z.enum(['simulated', 'real']),
    pair: z.string(),
    total_equity: z.string().nullable(),
    total_signed_notional: z.string(),
    connections: z.array(ConnectionPortfolioSchema),
  })
  .strict();
export const AccountMoneySchema = z
  .object({ amount: z.string().nullable(), currency: z.string(), unavailable_reason: z.string().nullable() })
  .strict();
const AccountInstrumentSchema = z
  .object({
    venue_symbol: z.string(),
    pair: z.string().nullable(),
    market_type: z.string(),
    tradable: z.boolean(),
    reason: z.string().nullable(),
  })
  .strict();
const AccountPositionSchema = z
  .object({
    instrument: AccountInstrumentSchema,
    signed_amount: z.string(),
    available_amount: z.string().nullable(),
    signed_notional: AccountMoneySchema,
    entry_price: z.string().nullable(),
    unrealized_pnl: AccountMoneySchema,
  })
  .strict();
const AccountOrderSchema = z
  .object({
    connection_id: z.string(),
    venue_order_id: z.string(),
    instrument: AccountInstrumentSchema,
    side: z.string(),
    order_type: z.string(),
    amount: z.string(),
    filled_amount: z.string(),
    average_price: z.string().nullable(),
    status: z.string(),
    reduce_only: z.boolean(),
    protection: z.boolean(),
    client_order_id: z.string().nullable(),
    observed_at: z.string(),
    remaining_notional: AccountMoneySchema,
  })
  .strict();
const AttributionSchema = z
  .object({
    source: z.string(),
    book_id: z.string().nullable(),
    decision_id: z.string().nullable(),
    operation_id: z.string().nullable(),
  })
  .strict();
const AccountCoverageSchema = z
  .object({
    coverage_start: z.string(),
    coverage_end: z.string(),
    complete: z.boolean(),
    from_inception: z.boolean(),
  })
  .strict();
export const AccountSchema = z
  .object({
    archived: z.boolean().default(false),
    connection_id: z.string(),
    label: z.string(),
    adapter_id: z.string(),
    environment: z.string(),
    capital_scope: z.enum(['simulated', 'real']),
    enabled: z.boolean(),
    book_ids: z.array(z.string()),
    snapshot: z
      .object({
        connection_id: z.string(),
        observed_at: z.string(),
        capital_scope: z.enum(['simulated', 'real']),
        equity: AccountMoneySchema,
        balances: z.array(AccountMoneySchema),
        positions: z.array(AccountPositionSchema),
        orders: z.array(AccountOrderSchema),
        used_margin: AccountMoneySchema,
        available_margin: AccountMoneySchema,
        completeness: z.array(z.string()),
        valuation_notes: z.array(z.string()),
      })
      .strict()
      .nullable(),
    last_success_at: z.string().nullable(),
    last_failure_at: z.string().nullable(),
    failure_reason: z.string().nullable(),
    coverage: z.object({ fills: AccountCoverageSchema.nullable(), funding: AccountCoverageSchema.nullable() }).strict(),
    orders: z.array(
      AccountOrderSchema.extend({ attribution: AttributionSchema, currently_open: z.boolean() }).strict(),
    ),
  })
  .strict();
export const AccountsSchema = z
  .object({ items: z.array(AccountSchema), simulated: z.array(AccountMoneySchema), real: z.array(AccountMoneySchema) })
  .strict();
export const AccountFillsSchema = z
  .object({
    items: z.array(
      z
        .object({
          connection_id: z.string(),
          venue_fill_id: z.string(),
          venue_order_id: z.string(),
          instrument: AccountInstrumentSchema,
          side: z.string(),
          amount: z.string(),
          price: z.string(),
          occurred_at: z.string(),
          fee: AccountMoneySchema,
          realized_pnl: AccountMoneySchema,
          source: z.enum(['platform', 'local_calculation']),
          client_order_id: z.string().nullable(),
          attribution: AttributionSchema,
        })
        .strict(),
    ),
    total: z.number(),
    offset: z.number(),
    limit: z.number(),
  })
  .strict();
export const AccountIncomeSchema = z
  .object({
    start: z.string(),
    end: z.string(),
    realized_gross: z.array(AccountMoneySchema),
    fees: z.array(AccountMoneySchema),
    funding: z.array(AccountMoneySchema),
    unrealized: z.array(AccountMoneySchema),
    unrealized_as_of: z.string().nullable(),
    net_trading: z.array(AccountMoneySchema),
    completeness: z.array(z.string()),
    methodology: z.string(),
    items: z.array(
      z
        .object({
          connection_id: z.string(),
          venue_entry_id: z.string(),
          instrument: AccountInstrumentSchema,
          amount: AccountMoneySchema,
          occurred_at: z.string(),
        })
        .strict(),
    ),
    total: z.number(),
    offset: z.number(),
    limit: z.number(),
  })
  .strict();
export const AccountBookSchema = z
  .object({
    book_id: z.string(),
    label: z.string(),
    capital_scope: z.enum(['simulated', 'real']),
    enabled: z.boolean(),
    total_equity: z.array(AccountMoneySchema),
    total_signed_notional: z.array(AccountMoneySchema),
    connections: z.array(AccountSchema),
    risk_state: BookRiskStateSchema.nullable(),
  })
  .strict();
const AccountScopeSchema = z
  .object({
    books: z.array(AccountBookSchema),
    equity: z.array(AccountMoneySchema),
    signed_notional: z.array(AccountMoneySchema),
  })
  .strict();
export const PortfolioBooksSchema = z.object({ simulated: AccountScopeSchema, real: AccountScopeSchema }).strict();
const ConnectionRiskSchema = z
  .object({
    connection_id: z.string(),
    passed: z.boolean(),
    risk_increase: z.boolean(),
    reason: z.string(),
    operation: z.string(),
  })
  .strict();
const NormalizedOrderSchema = z
  .object({
    id: z.string(),
    pair: PairSymbolSchema,
    side: z.string(),
    order_type: z.string(),
    amount: z.string(),
    filled_amount: z.string(),
    average_price: z.string().nullable(),
    status: z.string(),
    reduce_only: z.boolean(),
    client_order_id: z.string().nullable(),
  })
  .strict();
const ProtectionStateSchema = z
  .object({
    actual_order_ids: z.array(z.string()),
    protection_ids: z.array(z.string()),
    pair: PairSymbolSchema,
    position_side: z.string(),
    amount: z.string(),
    stop_loss: z.string().nullable(),
    take_profit: z.string().nullable(),
    active: z.boolean(),
    triggered: z.boolean(),
  })
  .strict();
const FinalPositionSchema = z
  .object({
    position: z
      .object({
        pair: PairSymbolSchema,
        signed_amount: z.string(),
        signed_notional: z.string(),
        entry_price: z.string().nullable(),
      })
      .strict(),
    protected: z.boolean(),
    protection_ids: z.array(z.string()),
    protections: z.array(ProtectionStateSchema),
  })
  .strict();
const CompensationSchema = z
  .object({
    attempted: z.boolean(),
    succeeded: z.boolean(),
    order: NormalizedOrderSchema.nullable(),
    operation: z.string(),
    safe_signed_amount: z.string().nullable(),
    required_protection: ProtectionStateSchema.nullable(),
  })
  .strict();
const ConnectionExecutionSchema = z
  .object({
    book_id: z.string(),
    connection_id: z.string(),
    pair: PairSymbolSchema,
    target_signed_notional: z.string(),
    target_signed_amount: z.string(),
    status: z.string(),
    orders: z.array(NormalizedOrderSchema),
    protection: ProtectionStateSchema.nullable(),
    compensation: CompensationSchema,
    final_position: FinalPositionSchema.nullable(),
    error_operation: z.string(),
    requires_attention: z.boolean(),
    trace: z.array(z.string()),
    execution_quote: z
      .object({ pair: PairSymbolSchema, bid: z.string(), ask: z.string(), last: z.string() })
      .strict()
      .nullable(),
    quantity_frozen: z.boolean().nullable(),
  })
  .strict();
const CycleConnectionSchema = z
  .object({
    connection_id: z.string(),
    portfolio_before: ConnectionPortfolioSchema.nullable(),
    portfolio_after: ConnectionPortfolioSchema.nullable(),
    risk: ConnectionRiskSchema.nullable(),
    plan: ConnectionPlanSchema.nullable(),
    execution: ConnectionExecutionSchema.nullable(),
    unavailable: z.boolean(),
  })
  .strict();
const BookRiskSchema = z
  .object({
    passed: z.boolean(),
    requested_target_exposure: z.string(),
    capped_target_exposure: z.string(),
    connection_weights: z.array(z.string()),
    connection_targets: z.array(
      z
        .object({
          book_id: z.string(),
          connection_id: z.string(),
          weight: z.string(),
          book_equity: z.string().nullable(),
          target_exposure: z.string(),
          target_signed_notional: z.string(),
        })
        .strict(),
    ),
    rejected_by: z.string(),
    reason: z.string(),
    cap_source: z.string(),
    state: BookRiskStateSchema.nullable(),
  })
  .strict();
const CycleBookSchema = z
  .object({
    book_id: z.string(),
    capital_scope: z.enum(['simulated', 'real']),
    config_revision: z.number().int(),
    pair: z.string(),
    market_type: z.string(),
    status: z.string(),
    hitl: z
      .object({ approval_id: z.string().nullable(), status: z.string(), config_revision: z.number().int() })
      .strict(),
    failure: z.object({ stage: z.string() }).strict().nullable(),
    requested_target_exposure: z.string().nullable(),
    target_exposure: z.string().nullable(),
    risk: BookRiskSchema.nullable(),
    ready: z.boolean().nullable(),
    errors: z.array(z.string()),
    execution: z
      .object({ status: z.string(), requires_attention: z.boolean(), reallocated: z.boolean() })
      .strict()
      .nullable(),
    portfolio_before: PortfolioBookSchema.nullable(),
    portfolio_after: PortfolioBookSchema.nullable(),
    portfolio_after_available: z.boolean().nullable(),
    reconciliation_required: z.boolean().nullable(),
    connections: z.array(CycleConnectionSchema),
  })
  .strict();
const PlainTextSchema = z
  .string()
  .refine((value) => !/<\s*\/?\s*[a-zA-Z][^>]*>|javascript\s*:/i.test(value), '仅支持普通文本');
const ResultScalarSchema = z.union([PlainTextSchema, z.number().finite(), z.boolean(), z.null()]);
const SavedTimeSchema = z.string().datetime({ offset: true });
export const ResultBlockSchema = z.discriminatedUnion('kind', [
  z.object({ kind: z.literal('text'), title: PlainTextSchema, body: PlainTextSchema }).strict(),
  z
    .object({
      kind: z.literal('metrics'),
      title: PlainTextSchema,
      metrics: z.array(
        z
          .object({
            key: PlainTextSchema,
            value: ResultScalarSchema,
            unit: PlainTextSchema.nullable(),
            note: PlainTextSchema.nullable(),
          })
          .strict(),
      ),
    })
    .strict(),
  z
    .object({
      kind: z.literal('series'),
      title: PlainTextSchema,
      forecast_start: SavedTimeSchema.nullable(),
      evaluation_target: z.literal('candle_close').nullable(),
      series: z.array(
        z
          .object({
            name: PlainTextSchema,
            unit: PlainTextSchema.nullable(),
            points: z.array(
              z
                .object({
                  time: SavedTimeSchema,
                  value: z
                    .string()
                    .refine((value) => Number.isFinite(Number(value)))
                    .nullable(),
                })
                .strict(),
            ),
          })
          .strict(),
      ),
    })
    .strict(),
  z
    .object({
      kind: z.literal('table'),
      title: PlainTextSchema,
      columns: z.array(z.object({ key: PlainTextSchema, label: PlainTextSchema }).strict()),
      rows: z.array(
        z
          .object({ cells: z.array(z.object({ column_key: PlainTextSchema, value: ResultScalarSchema }).strict()) })
          .strict(),
      ),
    })
    .strict(),
  z
    .object({
      kind: z.literal('timeline'),
      title: PlainTextSchema,
      entries: z.array(z.object({ time: SavedTimeSchema, actor: PlainTextSchema, body: PlainTextSchema }).strict()),
    })
    .strict(),
]);
export const EvaluationReferenceSchema = z
  .object({
    reference_time: SavedTimeSchema,
    reference_price: z.string(),
    due_at: SavedTimeSchema,
    interval: PlainTextSchema,
    market_source_id: PlainTextSchema,
  })
  .strict();
export const PredictionComparisonSchema = z
  .object({
    title: PlainTextSchema,
    name: PlainTextSchema,
    timeframe: PlainTextSchema,
    points: z.array(
      z
        .object({
          time: SavedTimeSchema,
          close_time: SavedTimeSchema,
          predicted: z.string().nullable(),
          actual: z.string().nullable(),
          difference: z.string().nullable(),
          status: z.enum(['pending', 'matched', 'missing_market']),
        })
        .strict(),
    ),
    matched: z.number().int().nonnegative(),
    total: z.number().int().nonnegative(),
    mae: z.string().nullable(),
    rmse: z.string().nullable(),
  })
  .strict();
export const ComponentEvaluationSchema = z
  .object({
    decision_id: z.string(),
    component_id: z.string(),
    pair: z.string().nullable(),
    mode: z.enum(['analysis', 'trading', 'backtest']),
    config_revision: z.number().int(),
    interval: z.string().nullable(),
    created_at: SavedTimeSchema,
    status: z.enum(['pending', 'evaluated', 'missing_market', 'not_directional', 'skipped', 'failed']),
    direction: z.enum(['long', 'short', 'neutral']),
    reference: EvaluationReferenceSchema.nullable(),
    actual_price: z.string().nullable(),
    actual_time: SavedTimeSchema.nullable(),
    hit: z.boolean().nullable(),
    return_ratio: z.string().nullable(),
    reason: z.string().nullable(),
    comparisons: z.array(PredictionComparisonSchema),
    cost: z.string().nullable(),
  })
  .strict();
export const EvaluationGroupSchema = z
  .object({
    component_id: z.string(),
    pair: z.string().nullable(),
    mode: z.enum(['analysis', 'trading', 'backtest']),
    config_revision: z.number().int(),
    interval: z.string().nullable(),
    total: z.number().int().nonnegative(),
    pending: z.number().int().nonnegative(),
    matured_directional: z.number().int().nonnegative(),
    hits: z.number().int().nonnegative(),
    neutral: z.number().int().nonnegative(),
    skipped: z.number().int().nonnegative(),
    failed: z.number().int().nonnegative(),
    missing_market: z.number().int().nonnegative(),
    hit_rate: z.number().min(0).max(1).nullable(),
  })
  .strict();
export const ComponentEvaluationsSchema = z
  .object({
    items: z.array(ComponentEvaluationSchema),
    summary: z.object({ groups: z.array(EvaluationGroupSchema) }).strict(),
    total: z.number().int(),
    limit: z.number().int(),
    offset: z.number().int(),
    has_next: z.boolean(),
  })
  .strict();
export const SavedComponentSignalSchema = z
  .object({
    component_id: z.string(),
    direction: z.enum(['long', 'short', 'neutral']),
    confidence: z.number(),
    reasoning: z.string(),
    details: z.array(JsonEntrySchema),
    blocks: z.array(ResultBlockSchema),
    evaluation_reference: EvaluationReferenceSchema.nullable(),
    status: z.enum(['completed', 'skipped', 'failed']),
    duration_ms: z.number().int().nonnegative().nullable(),
    usage: z
      .object({ input_tokens: z.number().int().nonnegative(), output_tokens: z.number().int().nonnegative() })
      .strict()
      .nullable(),
    cost: z.string().nullable(),
  })
  .strict();
export const CycleSchema = z
  .object({
    cycle_id: z.string(),
    config_revision: z.number().int(),
    market_data_source_id: z.string(),
    shared_signals: z
      .object({
        components: z.array(SavedComponentSignalSchema),
        fused: z
          .object({
            score: z.number(),
            reasoning: z.string(),
            contributions: z.array(
              z
                .object({
                  component_id: z.string(),
                  weight: z.number(),
                  signed_score: z.number(),
                  weighted_score: z.number(),
                })
                .strict(),
            ),
          })
          .strict()
          .nullable(),
        target_position: z.object({ side: z.string(), size_ratio: z.number() }).strict().nullable(),
      })
      .strict(),
    books: z.array(CycleBookSchema),
    cycle_status: z.string(),
    execution_status: z.string(),
    requires_attention: z.boolean(),
    created_at: z.string(),
  })
  .strict();
export const PaginatedCyclesSchema = z
  .object({
    items: z.array(CycleSchema),
    total: z.number().int(),
    page: z.number().int(),
    size: z.number().int(),
    has_next: z.boolean(),
  })
  .strict();

export const DecisionSchema = z
  .object({
    decision_id: z.string(),
    pair: z.string().nullable(),
    mode: z.enum(['analysis', 'trading', 'backtest']),
    origin: z.enum(['manual', 'scheduled', 'trigger', 'backtest']).nullable(),
    config_revision: z.number().int(),
    config_snapshot: z.array(JsonEntrySchema),
    created_at: SavedTimeSchema,
    finished_at: SavedTimeSchema.nullable(),
    status: z.string(),
    components: z.array(SavedComponentSignalSchema),
    fusion: CycleSchema.shape.shared_signals.shape.fused,
    target: CycleSchema.shape.shared_signals.shape.target_position,
    books: z.array(CycleBookSchema),
    failure: z.object({ code: z.string(), stage: z.string(), message: z.string() }).strict().nullable(),
    incomplete_fields: z.array(z.string()),
  })
  .strict();
export const DecisionListSchema = z
  .object({
    items: z.array(DecisionSchema),
    total: z.number().int(),
    limit: z.number().int(),
    offset: z.number().int(),
    has_next: z.boolean(),
  })
  .strict();
export const AnalysisQueuedSchema = z.object({ decision_id: z.string() }).strict();

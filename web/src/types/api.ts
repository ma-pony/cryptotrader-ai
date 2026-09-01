import type { z } from 'zod';

// Use z.output (post-parse type where .default() fields are required)
// instead of z.infer (input type where .default() fields are optional).

import type {
  ComponentEvaluationSchema,
  EvaluationGroupSchema,
  AccountOperationSchema,
  AccountMoneySchema,
  AccountSchema,
  AccountsSchema,
  AccountBookSchema,
  BookRiskStateSchema,
  AccountFillsSchema,
  AccountIncomeSchema,
  ReadinessSchema,
  TradingScopeSchema,
  DecisionSchema,
  ResultBlockSchema,
  SavedComponentSignalSchema,
  ConfigurationCatalogSchema,
  ConfigurationFieldSchema,
  PluginDefinitionSchema,
  ApprovalRequestSchema,
  PortfolioBooksSchema,
  CycleSchema,
  PaginatedCyclesSchema,
  BacktestMetricsSchema,
  BacktestEquityPointSchema,
  BacktestParamsSchema,
  BacktestResultSchema,
  BacktestRunStatusSchema,
  BacktestComparisonSchema,
  CommitteeDebateTurnSchema,
  ComponentContributionSchema,
  ConsensusMetricsSchema,
  CycleRiskResultSchema,
  CycleStatusSchema,
  DailyCostPointSchema,
  FusedSignalSchema,
  HitlRespondSchema,
  LatencyHistogramBucketSchema,
  MetricsCountersSchema,
  MetricsPercentilesSchema,
  MetricsSummarySchema,
  PaginatedTriggerEventsSchema,
  ScheduleRuleSchema,
  SchedulerStatusSchema,
  SignalProfileSchema,
  RuntimeConfigSchema,
  RuntimeConnectionSchema,
  RuntimeBookSchema,
  VenueMutationSchema,
  CredentialMutationSchema,
  RuntimeTokenMutationSchema,
  ConnectionHealthSchema,
  TargetPositionSchema,
  TriggerEventSchema,
  TriggerTypeSchema,
} from './api.schema';

export type ComponentEvaluation = z.output<typeof ComponentEvaluationSchema>;
export type EvaluationGroup = z.output<typeof EvaluationGroupSchema>;

export type AccountOperation = z.output<typeof AccountOperationSchema>;

export type AccountMoney = z.output<typeof AccountMoneySchema>;
export type Account = z.output<typeof AccountSchema>;
export type Accounts = z.output<typeof AccountsSchema>;
export type AccountBook = z.output<typeof AccountBookSchema>;
export type BookRiskState = z.output<typeof BookRiskStateSchema>;
export type AccountFills = z.output<typeof AccountFillsSchema>;
export type AccountIncome = z.output<typeof AccountIncomeSchema>;

// §2 Scheduler
export type SchedulerStatus = z.output<typeof SchedulerStatusSchema>;

// §3 Decisions
export type CycleStatus = z.output<typeof CycleStatusSchema>;
export type CommitteeDebateTurn = z.output<typeof CommitteeDebateTurnSchema>;
export type ConsensusMetrics = z.output<typeof ConsensusMetricsSchema>;
export type ResultBlock = z.output<typeof ResultBlockSchema>;
export type SavedComponentSignal = z.output<typeof SavedComponentSignalSchema>;
export type ComponentContribution = z.output<typeof ComponentContributionSchema>;
export type FusedSignal = z.output<typeof FusedSignalSchema>;
export type TargetPosition = z.output<typeof TargetPositionSchema>;
export type CycleRiskResult = z.output<typeof CycleRiskResultSchema>;

// §4 Backtest
export type BacktestParams = z.output<typeof BacktestParamsSchema>;
export type BacktestMetrics = z.output<typeof BacktestMetricsSchema>;
export type BacktestResult = z.output<typeof BacktestResultSchema>;
export type BacktestRunStatus = z.output<typeof BacktestRunStatusSchema>;
export type BacktestComparison = z.output<typeof BacktestComparisonSchema>;
export type EquityPoint = z.output<typeof BacktestEquityPointSchema>;

// Signal strategy profile
export type SignalProfile = z.output<typeof SignalProfileSchema>;
export type SignalProfileUpdate = Omit<SignalProfile, 'installed_components' | 'revision' | 'updated_at'>;
export type RuntimeConfig = z.output<typeof RuntimeConfigSchema>;
export type ConfigurationCatalog = z.output<typeof ConfigurationCatalogSchema>;
export type ConfigurationField = z.output<typeof ConfigurationFieldSchema>;
export type PluginDefinition = z.output<typeof PluginDefinitionSchema>;
/** A cleared number is a real local draft, never coerced to zero. Not an API document. */
export type ConfigurationDraft<T> = T extends number
  ? number | ''
  : T extends Array<infer U>
    ? ConfigurationDraft<U>[]
    : T extends object
      ? { [K in keyof T]: ConfigurationDraft<T[K]> }
      : T;
export type RuntimeConnection = z.output<typeof RuntimeConnectionSchema>;
export type RuntimeBook = z.output<typeof RuntimeBookSchema>;
export type RuntimeJsonPrimitive = null | boolean | number | string;
export type RuntimeJsonValue = RuntimeJsonPrimitive | RuntimeJsonValue[] | RuntimeJsonObject;
export type RuntimeJsonObject = { [key: string]: RuntimeJsonValue };
type RuntimeResponseDocument = RuntimeConfig['document'];
export type RuntimeDocument = Omit<
  RuntimeResponseDocument,
  'market_data' | 'signals' | 'execution' | 'security' | 'llm'
> & {
  security: Pick<RuntimeResponseDocument['security'], 'enabled'>;
  llm: Omit<RuntimeResponseDocument['llm'], 'gateway_credential_configured' | 'gateway_credential_updated_at'>;
  market_data: Omit<
    RuntimeResponseDocument['market_data'],
    'parameters' | 'news_credential_configured' | 'news_credential_updated_at'
  > & { parameters: RuntimeJsonObject };
  signals: Omit<RuntimeResponseDocument['signals'], 'components'> & {
    components: Array<
      Omit<RuntimeResponseDocument['signals']['components'][number], 'parameters'> & { parameters: RuntimeJsonObject }
    >;
  };
  execution: Omit<RuntimeResponseDocument['execution'], 'connections'> & {
    connections: Array<
      Omit<
        RuntimeResponseDocument['execution']['connections'][number],
        'credential_configured' | 'credential_updated_at' | 'parameters'
      > & { parameters: RuntimeJsonObject }
    >;
  };
};
export type VenueMutation = z.output<typeof VenueMutationSchema>;
export type CredentialMutation = z.output<typeof CredentialMutationSchema>;
export type RuntimeTokenMutation = z.output<typeof RuntimeTokenMutationSchema>;
export type ConnectionHealth = z.output<typeof ConnectionHealthSchema>;

// §6 Metrics
export type MetricsCounters = z.output<typeof MetricsCountersSchema>;
export type MetricsPercentiles = z.output<typeof MetricsPercentilesSchema>;
export type MetricsSummary = z.output<typeof MetricsSummarySchema>;
export type LatencyHistogramBucket = z.output<typeof LatencyHistogramBucketSchema>;
export type DailyCostPoint = z.output<typeof DailyCostPointSchema>;

// §7 Triggers
export type TriggerType = z.output<typeof TriggerTypeSchema>;
export type ScheduleRule = z.output<typeof ScheduleRuleSchema>;
export type TriggerEvent = z.output<typeof TriggerEventSchema>;
export type PaginatedTriggerEvents = z.output<typeof PaginatedTriggerEventsSchema>;

// §8 HITL Approvals
export type ApprovalRequest = z.output<typeof ApprovalRequestSchema>;
export type HitlRespond = z.output<typeof HitlRespondSchema>;
export type PortfolioBooks = z.output<typeof PortfolioBooksSchema>;
export type Cycle = z.output<typeof CycleSchema>;
export type PaginatedCycles = z.output<typeof PaginatedCyclesSchema>;
export type Decision = z.output<typeof DecisionSchema>;

// Filters
export interface DecisionListFilter {
  pair?: string;
  status?: CycleStatus;
  page?: number;
  size?: number;
}
export type Readiness = z.output<typeof ReadinessSchema>;
export type TradingScope = z.output<typeof TradingScopeSchema>;

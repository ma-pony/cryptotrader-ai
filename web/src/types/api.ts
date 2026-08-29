import type { z } from 'zod';

// Use z.output (post-parse type where .default() fields are required)
// instead of z.infer (input type where .default() fields are optional).

import type {
  ApprovalRequestSchema,
  PortfolioBooksSchema,
  CycleSchema,
  PaginatedCyclesSchema,
  BacktestMetricsSchema,
  BacktestParamsSchema,
  BacktestResultSchema,
  BacktestRunStatusSchema,
  BacktestSessionDetailSchema,
  CircuitBreakerStatusSchema,
  CommitteeAgentAnalysisSchema,
  CommitteeDebateTurnSchema,
  ComponentContributionSchema,
  ComponentSignalSchema,
  ConsensusMetricsSchema,
  CooldownSchema,
  CorrelationGroupSchema,
  CycleRiskResultSchema,
  CycleStatusSchema,
  DailyCostPointSchema,
  EquityCurveSchema,
  EquityPointSchema,
  FusedSignalSchema,
  HitlRespondSchema,
  LatencyHistogramBucketSchema,
  MetricsCountersSchema,
  MetricsPercentilesSchema,
  MetricsSummarySchema,
  PaginatedTriggerEventsSchema,
  PortfolioSchema,
  PositionSchema,
  RecentBlockSchema,
  RiskStatusSchema,
  RiskThresholdsSchema,
  ScheduleRuleSchema,
  SchedulerStatusSchema,
  SignalProfileSchema,
  RuntimeConfigSchema,
  RuntimeConnectionSchema,
  RuntimeBookSchema,
  VenueMutationSchema,
  CredentialMutationSchema,
  ConnectionHealthSchema,
  TargetPositionSchema,
  TriggerEventSchema,
  TriggerTypeSchema,
} from './api.schema';

// §1 Portfolio
export type Position = z.output<typeof PositionSchema>;
export type Portfolio = z.output<typeof PortfolioSchema>;
export type EquityPoint = z.output<typeof EquityPointSchema>;
export type EquityCurve = z.output<typeof EquityCurveSchema>;
export type RangeWindow = '24h' | '7d' | '30d' | 'all';

// §2 Scheduler
export type SchedulerStatus = z.output<typeof SchedulerStatusSchema>;

// §3 Decisions
export type CycleStatus = z.output<typeof CycleStatusSchema>;
export type CommitteeAgentAnalysis = z.output<typeof CommitteeAgentAnalysisSchema>;
export type CommitteeDebateTurn = z.output<typeof CommitteeDebateTurnSchema>;
export type ConsensusMetrics = z.output<typeof ConsensusMetricsSchema>;
export type ComponentSignal = z.output<typeof ComponentSignalSchema>;
export type ComponentContribution = z.output<typeof ComponentContributionSchema>;
export type FusedSignal = z.output<typeof FusedSignalSchema>;
export type TargetPosition = z.output<typeof TargetPositionSchema>;
export type CycleRiskResult = z.output<typeof CycleRiskResultSchema>;

// §4 Backtest
export type BacktestParams = z.output<typeof BacktestParamsSchema>;
export type BacktestMetrics = z.output<typeof BacktestMetricsSchema>;
export type BacktestResult = z.output<typeof BacktestResultSchema>;
export type BacktestRunStatus = z.output<typeof BacktestRunStatusSchema>;
export type BacktestSessionDetail = z.output<typeof BacktestSessionDetailSchema>;

// Signal strategy profile
export type SignalProfile = z.output<typeof SignalProfileSchema>;
export type SignalProfileUpdate = Omit<SignalProfile, 'installed_components' | 'revision' | 'updated_at'>;
export type RuntimeConfig = z.output<typeof RuntimeConfigSchema>;
export type RuntimeConnection = z.output<typeof RuntimeConnectionSchema>;
export type RuntimeBook = z.output<typeof RuntimeBookSchema>;
export type RuntimeJsonPrimitive = null | boolean | number | string;
export type RuntimeJsonValue = RuntimeJsonPrimitive | RuntimeJsonValue[] | RuntimeJsonObject;
export type RuntimeJsonObject = { [key: string]: RuntimeJsonValue };
type RuntimeResponseDocument = RuntimeConfig['document'];
export type RuntimeDocument = Omit<RuntimeResponseDocument, 'market_data' | 'signals' | 'execution'> & {
  market_data: Omit<RuntimeResponseDocument['market_data'], 'parameters'> & { parameters: RuntimeJsonObject };
  signals: Omit<RuntimeResponseDocument['signals'], 'components'> & { components: Array<Omit<RuntimeResponseDocument['signals']['components'][number], 'parameters'> & { parameters: RuntimeJsonObject }> };
  execution: Omit<RuntimeResponseDocument['execution'], 'connections'> & { connections: Array<Omit<RuntimeResponseDocument['execution']['connections'][number], 'credential_configured' | 'credential_updated_at' | 'parameters'> & { parameters: RuntimeJsonObject }> };
};
export type VenueMutation = z.output<typeof VenueMutationSchema>;
export type CredentialMutation = z.output<typeof CredentialMutationSchema>;
export type ConnectionHealth = z.output<typeof ConnectionHealthSchema>;

// §5 Risk
export type CircuitBreakerStatus = z.output<typeof CircuitBreakerStatusSchema>;
export type RiskThresholds = z.output<typeof RiskThresholdsSchema>;
export type RiskStatus = z.output<typeof RiskStatusSchema>;
export type CorrelationGroup = z.output<typeof CorrelationGroupSchema>;
export type Cooldown = z.output<typeof CooldownSchema>;
export type RecentBlock = z.output<typeof RecentBlockSchema>;

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

// §9 Chat (P2 — stub types for store compatibility)
export type ChatRole = 'user' | 'assistant' | 'system';
export interface ChatMessage {
  id: string;
  role: ChatRole;
  ts: string;
  content_md?: string;
}

// Filters
export interface DecisionListFilter {
  pair?: string;
  status?: CycleStatus;
  page?: number;
  size?: number;
}

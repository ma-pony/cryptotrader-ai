import type { z } from 'zod';

// Use z.output (post-parse type where .default() fields are required)
// instead of z.infer (input type where .default() fields are optional).

import type {
  AgentAnalysisSchema,
  AgentBiasSchema,
  BacktestMetricsSchema,
  BiasSchema,
  BacktestParamsSchema,
  BacktestResultSchema,
  BacktestRunStatusSchema,
  BacktestSessionDetailSchema,
  CircuitBreakerStatusSchema,
  ConsensusMetricsSchema,
  CooldownSchema,
  CorrelationGroupSchema,
  DailyCostPointSchema,
  DebateGateSchema,
  DebateRoundSchema,
  DebateTurnSchema,
  DecisionDetailSchema,
  DecisionListItemSchema,
  EquityCurveSchema,
  EquityPointSchema,
  ExecutionSchema,
  ExperienceMemoryRefSchema,
  LatencyBreakdownSchema,
  LatencyHistogramBucketSchema,
  MetricsCountersSchema,
  MetricsPercentilesSchema,
  MetricsSummarySchema,
  NodeTimelineEntrySchema,
  PaginatedDecisionsSchema,
  PaginatedTriggerEventsSchema,
  PortfolioSchema,
  PositionSchema,
  RecentBlockSchema,
  RiskCheckSchema,
  RiskGateSchema,
  RiskStatusSchema,
  RiskThresholdsSchema,
  ScheduleRuleSchema,
  SchedulerStatusSchema,
  TokenUsageSchema,
  TriggerEventSchema,
  TriggerTypeSchema,
  VerdictSlimSchema,
  AgentAnalysisSummarySchema,
  ApprovalRequestSchema,
  HitlRespondSchema,
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
export type VerdictSlim = z.output<typeof VerdictSlimSchema>;
export type DecisionListItem = z.output<typeof DecisionListItemSchema>;
export type PaginatedDecisions = z.output<typeof PaginatedDecisionsSchema>;
export type AgentAnalysis = z.output<typeof AgentAnalysisSchema>;
export type DebateRound = z.output<typeof DebateRoundSchema>;
export type DebateTurn = z.output<typeof DebateTurnSchema>;
export type DebateGate = z.output<typeof DebateGateSchema>;
export type ConsensusMetrics = z.output<typeof ConsensusMetricsSchema>;
export type LatencyBreakdown = z.output<typeof LatencyBreakdownSchema>;
export type TokenUsage = z.output<typeof TokenUsageSchema>;
export type RiskCheck = z.output<typeof RiskCheckSchema>;
export type RiskGate = z.output<typeof RiskGateSchema>;
export type Execution = z.output<typeof ExecutionSchema>;
export type NodeTimelineEntry = z.output<typeof NodeTimelineEntrySchema>;
export type ExperienceMemoryRef = z.output<typeof ExperienceMemoryRefSchema>;
export type DecisionDetail = z.output<typeof DecisionDetailSchema>;
export type AgentBias = z.output<typeof AgentBiasSchema>;
export type Bias = z.output<typeof BiasSchema>;

// §4 Backtest
export type BacktestParams = z.output<typeof BacktestParamsSchema>;
export type BacktestMetrics = z.output<typeof BacktestMetricsSchema>;
export type BacktestResult = z.output<typeof BacktestResultSchema>;
export type BacktestRunStatus = z.output<typeof BacktestRunStatusSchema>;
export type BacktestSessionDetail = z.output<typeof BacktestSessionDetailSchema>;

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
export type AgentAnalysisSummary = z.output<typeof AgentAnalysisSummarySchema>;
export type ApprovalRequest = z.output<typeof ApprovalRequestSchema>;
export type HitlRespond = z.output<typeof HitlRespondSchema>;

// §9 Chat (P2 — stub types for store compatibility)
export type ChatRole = 'user' | 'assistant' | 'system';
export interface ChatMessage {
  id: string;
  role: ChatRole;
  ts: string;
  content_md?: string;
  tool_calls?: Array<{ id: string; name: string; args: Record<string, unknown> }>;
  tool_results?: Array<{ tool_call_id: string; output_md: string }>;
  inline_widgets?: Array<{ widget_id: string; html: string; height_px?: number }>;
}

// Filters
export interface DecisionListFilter {
  pair?: string;
  from?: string;
  to?: string;
  page?: number;
  size?: number;
}

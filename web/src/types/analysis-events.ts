import type { ComponentSignal, FusedSignal } from './api';

export type AnalysisEventType =
  | 'session_start'
  | 'cycle_started'
  | 'component_started'
  | 'component_completed'
  | 'component_failed'
  | 'committee_agent_started'
  | 'committee_agent_completed'
  | 'committee_agent_failed'
  | 'debate_round_started'
  | 'debate_round_completed'
  | 'committee_summary_completed'
  | 'fusion_completed'
  | 'cycle_awaiting_approval'
  | 'cycle_completed'
  | 'cycle_failed'
  | 'cycle_cancelled'
  | 'stream_resume'
  | 'stream_done'
  | 'stream_error'
  | 'interrupt_received'
  | 'interrupt_noop'
  | 'interrupt_rejected'
  | 'steer_queued'
  | 'steer_too_late'
  | 'steer_truncated';

export interface SSEEnvelope<T = Record<string, unknown>> {
  event_id: number;
  type: AnalysisEventType;
  ts: string;
  session_id: string;
  data: T;
}

export interface CycleStartedData {
  cycle_id: string;
  pair: string;
  mode: 'live' | 'paper' | 'backtest';
}

export interface ComponentStartedData {
  component_id: string;
}

export interface ComponentCompletedData {
  component_id: string;
  signal: ComponentSignal;
}

export interface ComponentFailedData {
  component_id: string;
  error_type: string;
  error: string;
}

export interface CommitteeAgentCompletedData {
  agent_id: string;
  analysis: {
    direction: string;
    confidence: number;
  };
}

export interface DebateRoundData {
  round_number: number;
}

export interface FusionCompletedData {
  cycle_id: string;
  fusion: FusedSignal;
}

export interface CycleFinishedData {
  cycle_id: string;
  status: string;
  error?: string | null;
}

export interface SteerQueuedData {
  target: string;
  queue_position: number;
}

export interface StreamResumeData {
  session_id: string;
  last_event_id: number;
}

export interface StreamErrorData {
  error: string;
}

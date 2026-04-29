export type AnalysisEventType =
  | 'session_start'
  | 'session_replaced'
  | 'node_started'
  | 'node_done'
  | 'agent_thinking'
  | 'agent_analysis'
  | 'debate_started'
  | 'debate_round_done'
  | 'verdict_ready'
  | 'verdict_partial'
  | 'risk_checked'
  | 'checkpoint_saved'
  | 'interrupt_received'
  | 'interrupt_noop'
  | 'interrupt_rejected'
  | 'steer_queued'
  | 'steer_too_late'
  | 'steer_truncated'
  | 'stream_resume'
  | 'stream_done'
  | 'stream_error';

export type LegacyEventType =
  | 'session'
  | 'message_start'
  | 'content_delta'
  | 'tool_call'
  | 'message_end'
  | 'done';

export interface SSEEnvelope<T = Record<string, unknown>> {
  event_id: number;
  type: AnalysisEventType;
  ts: string;
  session_id: string;
  data: T;
}

export interface NodeStartedData {
  node_name: string;
}

export interface NodeDoneData {
  node_name: string;
  duration_ms: number;
}

export interface AgentThinkingData {
  agent_id: string;
}

export interface AgentAnalysisData {
  agent_id: string;
  direction: string;
  confidence: number;
  steered: boolean;
}

export interface DebateStartedData {
  round_number: number;
}

export interface DebateRoundDoneData {
  round_number: number;
  updated_positions: Record<string, { direction: string; confidence: number }>;
}

export interface VerdictReadyData {
  action: string;
  confidence: number;
  position_scale: number;
  reasoning: string;
}

export interface VerdictPartialData {
  action: string;
  confidence: number;
  position_scale: number;
  reasoning: string;
  is_partial: true;
  completed_agents: string[];
  missing_agents: string[];
}

export interface RiskCheckedData {
  allowed: boolean;
  flags: string[];
  reason: string;
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

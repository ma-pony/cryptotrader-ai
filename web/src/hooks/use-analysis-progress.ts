import { useCallback, useRef, useState } from 'react';

import type { SSEEvent } from '@/lib/stream-fetch';
import type { FusedSignal } from '@/types/api';
import type {
  AgentAnalysisCompletedData,
  ComponentCompletedData,
  ComponentFailedData,
  ComponentStartedData,
  CycleFinishedData,
  CycleStartedData,
  DebateRoundData,
  FusionCompletedData,
  SSEEnvelope,
} from '@/types/analysis-events';

export interface ComponentProgressSignal {
  direction: ComponentCompletedData['direction'];
  confidence: number;
}

export interface ComponentProgress {
  status: 'running' | 'done' | 'failed';
  signal?: ComponentProgressSignal;
  error?: string;
}

export interface AgentProgress {
  status: 'thinking' | 'done' | 'failed';
  direction: string;
  confidence: number;
}

export interface AnalysisProgressState {
  cycleId: string | null;
  status: 'idle' | 'running' | 'awaiting_approval' | 'completed' | 'failed' | 'cancelled';
  components: Record<string, ComponentProgress>;
  agents: Record<string, AgentProgress>;
  debateRound: number;
  fusion: FusedSignal | null;
  cancelled: boolean;
  lastEventId: number;
}

const INITIAL_STATE: AnalysisProgressState = {
  cycleId: null,
  status: 'idle',
  components: {},
  agents: {},
  debateRound: 0,
  fusion: null,
  cancelled: false,
  lastEventId: 0,
};

export function useAnalysisProgress() {
  const [progress, setProgress] = useState<AnalysisProgressState>(INITIAL_STATE);
  const lastEventIdRef = useRef(0);

  const handleProgressEvent = useCallback((event: SSEEvent) => {
    const envelope = event.data as SSEEnvelope;
    lastEventIdRef.current = envelope.event_id;
    const lastEventId = lastEventIdRef.current;
    const payload = envelope.data;

    switch (event.event) {
      case 'cycle_started': {
        const data = payload as unknown as CycleStartedData;
        setProgress({ ...INITIAL_STATE, cycleId: data.cycle_id, status: 'running', lastEventId });
        break;
      }
      case 'component_started': {
        const data = payload as unknown as ComponentStartedData;
        setProgress((current) => ({ ...current, components: { ...current.components, [data.component_id]: { status: 'running' } }, lastEventId }));
        break;
      }
      case 'component_completed': {
        const data = payload as unknown as ComponentCompletedData;
        setProgress((current) => ({
          ...current,
          components: {
            ...current.components,
            [data.component_id]: {
              status: 'done',
              signal: { direction: data.direction, confidence: data.confidence },
            },
          },
          lastEventId,
        }));
        break;
      }
      case 'component_failed': {
        const data = payload as unknown as ComponentFailedData;
        setProgress((current) => ({ ...current, components: { ...current.components, [data.component_id]: { status: 'failed', error: data.error } }, lastEventId }));
        break;
      }
      case 'committee_agent_started': {
        const { agent_id } = payload as { agent_id: string };
        setProgress((current) => ({ ...current, agents: { ...current.agents, [agent_id]: { status: 'thinking', direction: '', confidence: 0 } }, lastEventId }));
        break;
      }
      case 'agent_analysis_completed': {
        const data = payload as unknown as AgentAnalysisCompletedData;
        setProgress((current) => ({ ...current, agents: { ...current.agents, [data.agent_id]: { status: 'done', direction: data.analysis.direction, confidence: data.analysis.confidence } }, lastEventId }));
        break;
      }
      case 'committee_agent_failed': {
        const { agent_id } = payload as { agent_id: string };
        setProgress((current) => ({ ...current, agents: { ...current.agents, [agent_id]: { status: 'failed', direction: '', confidence: 0 } }, lastEventId }));
        break;
      }
      case 'debate_round_started':
      case 'debate_round_completed': {
        const data = payload as unknown as DebateRoundData;
        setProgress((current) => ({ ...current, debateRound: data.round_number, lastEventId }));
        break;
      }
      case 'fusion_completed': {
        const data = payload as unknown as FusionCompletedData;
        setProgress((current) => ({ ...current, cycleId: data.cycle_id, fusion: data.fusion, lastEventId }));
        break;
      }
      case 'cycle_awaiting_approval': {
        const data = payload as unknown as CycleFinishedData;
        setProgress((current) => ({ ...current, cycleId: data.cycle_id, status: 'awaiting_approval', lastEventId }));
        break;
      }
      case 'cycle_completed': {
        const data = payload as unknown as CycleFinishedData;
        setProgress((current) => ({ ...current, cycleId: data.cycle_id, status: 'completed', lastEventId }));
        break;
      }
      case 'cycle_failed': {
        const data = payload as unknown as CycleFinishedData;
        setProgress((current) => ({ ...current, cycleId: data.cycle_id, status: 'failed', lastEventId }));
        break;
      }
      case 'cycle_cancelled': {
        const data = payload as Partial<CycleFinishedData>;
        setProgress((current) => ({ ...current, cycleId: data.cycle_id ?? current.cycleId, status: 'cancelled', cancelled: true, lastEventId }));
        break;
      }
    }
  }, []);

  const reset = useCallback(() => {
    setProgress(INITIAL_STATE);
    lastEventIdRef.current = 0;
  }, []);

  return { progress, handleProgressEvent, reset };
}

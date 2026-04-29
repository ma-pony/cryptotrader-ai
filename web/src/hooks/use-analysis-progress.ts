/**
 * useAnalysisProgress — tracks structured analysis events for the progress panel.
 */
import { useCallback, useRef, useState } from 'react';

import { env } from '@/lib/env';
import type { SSEEvent } from '@/lib/stream-fetch';
import { useSettingsStore } from '@/stores/use-settings-store';
import type {
  AgentAnalysisData,
  DebateRoundDoneData,
  NodeDoneData,
  RiskCheckedData,
  VerdictPartialData,
  VerdictReadyData,
} from '@/types/analysis-events';

export interface NodeProgress {
  status: 'pending' | 'running' | 'done';
  duration_ms: number;
}

export interface AgentProgress {
  status: 'pending' | 'thinking' | 'done';
  direction: string;
  confidence: number;
  steered: boolean;
}

export interface AnalysisProgressState {
  nodes: Record<string, NodeProgress>;
  agents: Record<string, AgentProgress>;
  debateRound: number;
  verdict: VerdictReadyData | VerdictPartialData | null;
  riskCheck: RiskCheckedData | null;
  interrupted: boolean;
  lastEventId: number;
}

const INITIAL_STATE: AnalysisProgressState = {
  nodes: {},
  agents: {},
  debateRound: 0,
  verdict: null,
  riskCheck: null,
  interrupted: false,
  lastEventId: 0,
};

export interface UseAnalysisProgressReturn {
  progress: AnalysisProgressState;
  handleProgressEvent: (event: SSEEvent) => void;
  reset: () => void;
  sendInterrupt: (sessionId: string) => Promise<void>;
  sendSteer: (sessionId: string, target: string, instruction: string) => Promise<void>;
}

export function useAnalysisProgress(): UseAnalysisProgressReturn {
  const [progress, setProgress] = useState<AnalysisProgressState>(INITIAL_STATE);
  const lastEventIdRef = useRef(0);

  const handleProgressEvent = useCallback((event: SSEEvent) => {
    const envelope = event.data as { event_id?: number };
    if (typeof envelope.event_id === 'number') {
      lastEventIdRef.current = envelope.event_id;
    }

    switch (event.event) {
      case 'node_started': {
        const { node_name } = event.data as { node_name: string };
        setProgress((prev) => ({
          ...prev,
          nodes: { ...prev.nodes, [node_name]: { status: 'running', duration_ms: 0 } },
          lastEventId: lastEventIdRef.current,
        }));
        break;
      }

      case 'node_done': {
        const d = event.data as NodeDoneData;
        setProgress((prev) => ({
          ...prev,
          nodes: { ...prev.nodes, [d.node_name]: { status: 'done', duration_ms: d.duration_ms } },
          lastEventId: lastEventIdRef.current,
        }));
        break;
      }

      case 'agent_thinking': {
        const { agent_id } = event.data as { agent_id: string };
        setProgress((prev) => ({
          ...prev,
          agents: {
            ...prev.agents,
            [agent_id]: { status: 'thinking', direction: '', confidence: 0, steered: false },
          },
          lastEventId: lastEventIdRef.current,
        }));
        break;
      }

      case 'agent_analysis': {
        const a = event.data as AgentAnalysisData;
        setProgress((prev) => ({
          ...prev,
          agents: {
            ...prev.agents,
            [a.agent_id]: {
              status: 'done',
              direction: a.direction,
              confidence: a.confidence,
              steered: a.steered,
            },
          },
          lastEventId: lastEventIdRef.current,
        }));
        break;
      }

      case 'debate_started':
      case 'debate_round_done': {
        const dr = event.data as DebateRoundDoneData;
        setProgress((prev) => ({
          ...prev,
          debateRound: dr.round_number,
          lastEventId: lastEventIdRef.current,
        }));
        break;
      }

      case 'verdict_ready': {
        const v = event.data as VerdictReadyData;
        setProgress((prev) => ({
          ...prev,
          verdict: v,
          lastEventId: lastEventIdRef.current,
        }));
        break;
      }

      case 'verdict_partial': {
        const vp = event.data as VerdictPartialData;
        setProgress((prev) => ({
          ...prev,
          verdict: vp,
          interrupted: true,
          lastEventId: lastEventIdRef.current,
        }));
        break;
      }

      case 'risk_checked': {
        const rc = event.data as RiskCheckedData;
        setProgress((prev) => ({
          ...prev,
          riskCheck: rc,
          lastEventId: lastEventIdRef.current,
        }));
        break;
      }

      case 'checkpoint_saved':
      case 'interrupt_received': {
        setProgress((prev) => ({
          ...prev,
          interrupted: true,
          lastEventId: lastEventIdRef.current,
        }));
        break;
      }
    }
  }, []);

  const reset = useCallback(() => {
    setProgress(INITIAL_STATE);
    lastEventIdRef.current = 0;
  }, []);

  const sendInterrupt = useCallback(async (sessionId: string) => {
    const apiKey = useSettingsStore.getState().apiKey;
    await fetch(`${env.VITE_API_BASE_URL}/api/chat/interrupt/${encodeURIComponent(sessionId)}`, {
      method: 'POST',
      headers: { 'X-API-Key': apiKey },
    });
  }, []);

  const sendSteer = useCallback(async (sessionId: string, target: string, instruction: string) => {
    const apiKey = useSettingsStore.getState().apiKey;
    await fetch(`${env.VITE_API_BASE_URL}/api/chat/steer/${encodeURIComponent(sessionId)}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': apiKey,
      },
      body: JSON.stringify({ target, instruction }),
    });
  }, []);

  return { progress, handleProgressEvent, reset, sendInterrupt, sendSteer };
}

import { act, renderHook } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import type { SSEEvent } from '@/lib/stream-fetch';

import { useAnalysisProgress } from './use-analysis-progress';

const event = (type: string, data: Record<string, unknown>, eventId = 1): SSEEvent => ({
  event: type,
  data: {
    event_id: eventId,
    type,
    ts: '2026-08-28T00:00:00Z',
    session_id: 'session-1',
    data,
  },
});

describe('useAnalysisProgress', () => {
  it('reduces cycle component, fusion, and cancellation events without a verdict state', () => {
    const { result } = renderHook(() => useAnalysisProgress());

    act(() => result.current.handleProgressEvent(event('cycle_started', { cycle_id: 'cycle-1', pair: 'BTC/USDT', mode: 'paper' })));
    act(() => result.current.handleProgressEvent(event('component_completed', {
      component_id: 'kronos',
      signal: { component_id: 'kronos', direction: 'long', confidence: 0.8, reasoning: 'trend', details: {} },
    }, 2)));
    act(() => result.current.handleProgressEvent(event('fusion_completed', {
      cycle_id: 'cycle-1',
      fusion: {
        score: 0.48,
        reasoning: 'weighted',
        contributions: [{ component_id: 'kronos', weight: 0.6, signed_score: 0.8, weighted_score: 0.48 }],
      },
    }, 3)));

    expect(result.current.progress.components.kronos?.signal?.direction).toBe('long');
    expect(result.current.progress.fusion?.score).toBe(0.48);
    expect(result.current.progress).not.toHaveProperty('verdict');

    act(() => result.current.handleProgressEvent(event('cycle_cancelled', { cycle_id: 'cycle-1', status: 'cancelled' }, 4)));
    expect(result.current.progress.cancelled).toBe(true);
    expect(result.current.progress.status).toBe('cancelled');
  });
});

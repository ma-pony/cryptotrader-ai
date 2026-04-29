import { renderHook } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { useAdaptivePolling } from './use-adaptive-polling';

describe('useAdaptivePolling', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('returns false when WS is connected', () => {
    const { result } = renderHook(() =>
      useAdaptivePolling({ wsStatus: 'connected', priceChangePercent: 2 }),
    );
    expect(result.current.refetchInterval).toBe(false);
  });

  it('returns false when WS is connecting', () => {
    const { result } = renderHook(() =>
      useAdaptivePolling({ wsStatus: 'connecting' }),
    );
    expect(result.current.refetchInterval).toBe(false);
  });

  it('returns 10_000 when degraded with high volatility outside funding window', () => {
    vi.setSystemTime(new Date('2026-04-18T09:00:00Z'));
    const { result } = renderHook(() =>
      useAdaptivePolling({ wsStatus: 'degraded', priceChangePercent: 2.0 }),
    );
    expect(result.current.refetchInterval).toBe(10_000);
  });

  it('returns 60_000 when degraded with low volatility outside funding window', () => {
    vi.setSystemTime(new Date('2026-04-18T09:00:00Z'));
    const { result } = renderHook(() =>
      useAdaptivePolling({ wsStatus: 'degraded', priceChangePercent: 0.5 }),
    );
    expect(result.current.refetchInterval).toBe(60_000);
  });

  it('returns 10_000 when degraded inside funding window (UTC 07:45)', () => {
    vi.setSystemTime(new Date('2026-04-18T07:45:00Z'));
    const { result } = renderHook(() =>
      useAdaptivePolling({ wsStatus: 'degraded', priceChangePercent: 0.5 }),
    );
    expect(result.current.refetchInterval).toBe(10_000);
  });

  it('returns 60_000 when degraded outside funding window (UTC 09:00) with low volatility', () => {
    vi.setSystemTime(new Date('2026-04-18T09:00:00Z'));
    const { result } = renderHook(() =>
      useAdaptivePolling({ wsStatus: 'degraded', priceChangePercent: 0.5 }),
    );
    expect(result.current.refetchInterval).toBe(60_000);
  });

  it('returns 60_000 when degraded and priceChangePercent is undefined', () => {
    vi.setSystemTime(new Date('2026-04-18T09:00:00Z'));
    const { result } = renderHook(() =>
      useAdaptivePolling({ wsStatus: 'degraded' }),
    );
    expect(result.current.refetchInterval).toBe(60_000);
  });

  it('returns 10_000 when disconnected', () => {
    const { result } = renderHook(() =>
      useAdaptivePolling({ wsStatus: 'disconnected' }),
    );
    expect(result.current.refetchInterval).toBe(10_000);
  });

  it('returns 10_000 when reconnecting', () => {
    const { result } = renderHook(() =>
      useAdaptivePolling({ wsStatus: 'reconnecting' }),
    );
    expect(result.current.refetchInterval).toBe(10_000);
  });

  it('respects custom volatility threshold', () => {
    vi.setSystemTime(new Date('2026-04-18T09:00:00Z'));
    const { result } = renderHook(() =>
      useAdaptivePolling({ wsStatus: 'degraded', priceChangePercent: 1.5, volatilityThreshold: 2.0 }),
    );
    expect(result.current.refetchInterval).toBe(60_000);
  });

  it('returns 10_000 during midnight funding window (23:45 UTC)', () => {
    vi.setSystemTime(new Date('2026-04-18T23:45:00Z'));
    const { result } = renderHook(() =>
      useAdaptivePolling({ wsStatus: 'degraded', priceChangePercent: 0.3 }),
    );
    expect(result.current.refetchInterval).toBe(10_000);
  });

  it('returns 10_000 during midnight funding window wrap (00:15 UTC)', () => {
    vi.setSystemTime(new Date('2026-04-19T00:15:00Z'));
    const { result } = renderHook(() =>
      useAdaptivePolling({ wsStatus: 'degraded', priceChangePercent: 0.3 }),
    );
    expect(result.current.refetchInterval).toBe(10_000);
  });
});

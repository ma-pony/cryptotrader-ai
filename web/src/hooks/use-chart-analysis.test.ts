import { renderHook, act } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { createRef } from 'react';

import type { CandlestickChartHandle } from '@/types/chart-analysis';
import type { SSEEvent, StreamFetchOptions } from '@/lib/stream-fetch';

const mockNavigate = vi.fn();
vi.mock('react-router', () => ({ useNavigate: () => mockNavigate }));

vi.mock('@/lib/stream-fetch', () => ({
  streamFetch: vi.fn(),
}));

vi.mock('@/lib/indicators', () => ({
  generateDescription: vi.fn(() => 'mock-description'),
}));

describe('useChartAnalysis', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  function makeChartRef(screenshot: string | null = 'data:image/png;base64,abc') {
    const ref = createRef<CandlestickChartHandle>();
    (ref as { current: CandlestickChartHandle }).current = {
      captureScreenshot: () => screenshot,
    };
    return ref;
  }

  const bars = [
    { time: 1000000, open: 100, high: 110, low: 90, close: 105, volume: 1000 },
  ];

  it('starts with idle status', async () => {
    const { useChartAnalysis } = await import('./use-chart-analysis');
    const { result } = renderHook(() => useChartAnalysis());

    expect(result.current.result.status).toBe('idle');
    expect(result.current.result.contentMd).toBe('');
    expect(result.current.result.screenshotFailed).toBe(false);
  });

  it('triggerFast sets loading then streams content', async () => {
    const { streamFetch } = await import('@/lib/stream-fetch');
    const mockStreamFetch = vi.mocked(streamFetch);

    mockStreamFetch.mockImplementation((_path: string, opts: StreamFetchOptions) => {
      opts.onEvent?.({ event: 'content_delta', data: { delta: 'Hello' } } as SSEEvent);
      opts.onEvent?.({ event: 'content_delta', data: { delta: ' world' } } as SSEEvent);
      opts.onEvent?.({ event: 'done', data: {} } as SSEEvent);
      return Promise.resolve();
    });

    const { useChartAnalysis } = await import('./use-chart-analysis');
    const { result } = renderHook(() => useChartAnalysis());
    const chartRef = makeChartRef();

    act(() => {
      result.current.triggerFast(chartRef, bars, 'BTC/USDT', '1h');
    });

    expect(result.current.result.status).toBe('done');
    expect(result.current.result.contentMd).toBe('Hello world');
    expect(result.current.result.screenshotFailed).toBe(false);
  });

  it('triggerFast sets screenshotFailed when screenshot is null', async () => {
    const { streamFetch } = await import('@/lib/stream-fetch');
    const mockStreamFetch = vi.mocked(streamFetch);

    mockStreamFetch.mockImplementation((_path: string, opts: StreamFetchOptions) => {
      opts.onEvent?.({ event: 'done', data: {} } as SSEEvent);
      return Promise.resolve();
    });

    const { useChartAnalysis } = await import('./use-chart-analysis');
    const { result } = renderHook(() => useChartAnalysis());
    const chartRef = makeChartRef(null);

    act(() => {
      result.current.triggerFast(chartRef, bars, 'BTC/USDT', '1h');
    });

    expect(result.current.result.screenshotFailed).toBe(true);
  });

  it('triggerFast handles context_notice event', async () => {
    const { streamFetch } = await import('@/lib/stream-fetch');
    const mockStreamFetch = vi.mocked(streamFetch);

    mockStreamFetch.mockImplementation((_path: string, opts: StreamFetchOptions) => {
      opts.onEvent?.({ event: 'context_notice', data: { type: 'image_too_large' } } as SSEEvent);
      opts.onEvent?.({ event: 'done', data: {} } as SSEEvent);
      return Promise.resolve();
    });

    const { useChartAnalysis } = await import('./use-chart-analysis');
    const { result } = renderHook(() => useChartAnalysis());
    const chartRef = makeChartRef();

    act(() => {
      result.current.triggerFast(chartRef, bars, 'BTC/USDT', '1h');
    });

    expect(result.current.result.contextNotice).toBe('image_too_large');
  });

  it('triggerFast handles errors', async () => {
    const { streamFetch } = await import('@/lib/stream-fetch');
    const mockStreamFetch = vi.mocked(streamFetch);

    mockStreamFetch.mockImplementation((_path: string, opts: StreamFetchOptions) => {
      opts.onError?.(new Error('network failure'));
      return Promise.resolve();
    });

    const { useChartAnalysis } = await import('./use-chart-analysis');
    const { result } = renderHook(() => useChartAnalysis());
    const chartRef = makeChartRef();

    act(() => {
      result.current.triggerFast(chartRef, bars, 'BTC/USDT', '1h');
    });

    expect(result.current.result.status).toBe('error');
    expect(result.current.result.error).toBe('network failure');
  });

  it('triggerDeep navigates to /chat with context', async () => {
    const { useChartAnalysis } = await import('./use-chart-analysis');
    const { result } = renderHook(() => useChartAnalysis());
    const chartRef = makeChartRef();

    act(() => {
      result.current.triggerDeep(chartRef, bars, 'BTC/USDT', '4h');
    });

    expect(mockNavigate).toHaveBeenCalledWith('/chat', {
      state: {
        additionalContext: expect.objectContaining({
          payloads: [
            expect.objectContaining({
              symbol: 'BTC/USDT',
              timeframe: '4h',
              exchange: 'binance',
            }),
          ],
        }),
      },
    });
  });

  it('stop aborts and sets done', async () => {
    const { streamFetch } = await import('@/lib/stream-fetch');
    const mockStreamFetch = vi.mocked(streamFetch);

    mockStreamFetch.mockImplementation((_path: string, opts: StreamFetchOptions) => {
      opts.onEvent?.({ event: 'content_delta', data: { delta: 'partial' } } as SSEEvent);
      return Promise.resolve();
    });

    const { useChartAnalysis } = await import('./use-chart-analysis');
    const { result } = renderHook(() => useChartAnalysis());
    const chartRef = makeChartRef();

    act(() => {
      result.current.triggerFast(chartRef, bars, 'BTC/USDT', '1h');
    });

    act(() => {
      result.current.stop();
    });

    expect(result.current.result.status).toBe('done');
  });

  it('resetResult returns to idle state', async () => {
    const { streamFetch } = await import('@/lib/stream-fetch');
    const mockStreamFetch = vi.mocked(streamFetch);

    mockStreamFetch.mockImplementation((_path: string, opts: StreamFetchOptions) => {
      opts.onEvent?.({ event: 'done', data: {} } as SSEEvent);
      return Promise.resolve();
    });

    const { useChartAnalysis } = await import('./use-chart-analysis');
    const { result } = renderHook(() => useChartAnalysis());
    const chartRef = makeChartRef();

    act(() => {
      result.current.triggerFast(chartRef, bars, 'BTC/USDT', '1h');
    });

    act(() => {
      result.current.resetResult();
    });

    expect(result.current.result.status).toBe('idle');
    expect(result.current.result.contentMd).toBe('');
  });
});

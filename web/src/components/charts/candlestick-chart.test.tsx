import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, cleanup, act } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { CandlestickChartHandle } from '@/types/chart-analysis';
import { createRef } from 'react';

const mockRemove = vi.fn();
const mockTakeScreenshot = vi.fn();
const mockSetData = vi.fn();
const mockFitContent = vi.fn();
const mockSubscribe = vi.fn();
const mockApplyOptions = vi.fn();

vi.mock('lightweight-charts', () => ({
  ColorType: { Solid: 'Solid' },
  createChart: vi.fn(() => ({
    addCandlestickSeries: vi.fn(() => ({ setData: mockSetData })),
    addLineSeries: vi.fn(() => ({ setData: mockSetData })),
    timeScale: vi.fn(() => ({
      fitContent: mockFitContent,
      subscribeVisibleTimeRangeChange: mockSubscribe,
    })),
    applyOptions: mockApplyOptions,
    takeScreenshot: mockTakeScreenshot,
    remove: mockRemove,
  })),
}));

function wrapper({ children }: { children: React.ReactNode }) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={qc}>{children}</QueryClientProvider>;
}

describe('CandlestickChart', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    cleanup();
  });

  it('captureScreenshot returns null when not ready', async () => {
    const { CandlestickChart } = await import('./candlestick-chart');
    const ref = createRef<CandlestickChartHandle>();
    render(<CandlestickChart ref={ref} symbol="BTC/USDT" />, { wrapper });

    expect(ref.current?.captureScreenshot()).toBeNull();
  });

  it('captureScreenshot returns dataURL when ready', async () => {
    const mockCanvas = { toDataURL: vi.fn(() => 'data:image/png;base64,abc') };
    mockTakeScreenshot.mockReturnValue(mockCanvas);

    const { CandlestickChart } = await import('./candlestick-chart');
    const ref = createRef<CandlestickChartHandle>();
    render(<CandlestickChart ref={ref} symbol="BTC/USDT" />, { wrapper });

    const cb = mockSubscribe.mock.calls[0]?.[0];
    if (cb) act(() => { cb(); });

    expect(ref.current?.captureScreenshot()).toBe('data:image/png;base64,abc');
  });

  it('calls chart.remove on unmount', async () => {
    const { CandlestickChart } = await import('./candlestick-chart');
    const { unmount } = render(<CandlestickChart symbol="BTC/USDT" />, { wrapper });
    unmount();
    expect(mockRemove).toHaveBeenCalled();
  });
});

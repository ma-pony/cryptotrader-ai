import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, renderHook } from '@testing-library/react';
import { type ReactNode, useCallback } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { useMarketDataWS } from '@/hooks/use-market-data-ws';

import { MarketDataProvider } from './market-data-provider';

type WsEventHandler = ((ev: Event) => void) | null;
type WsMessageHandler = ((ev: MessageEvent) => void) | null;

class MockWebSocket {
  static instances: MockWebSocket[] = [];
  url: string;
  onopen: WsEventHandler = null;
  onmessage: WsMessageHandler = null;
  onerror: WsEventHandler = null;
  onclose: WsEventHandler = null;
  readyState = 0;
  closed = false;

  constructor(url: string) {
    this.url = url;
    MockWebSocket.instances.push(this);
  }

  close() {
    this.closed = true;
    this.readyState = 3;
  }

  send(_data: string) {}

  simulateOpen() {
    this.readyState = 1;
    this.onopen?.(new Event('open'));
  }

  simulateMessage(data: unknown) {
    this.onmessage?.(new MessageEvent('message', { data: JSON.stringify(data) }));
  }

  simulateClose() {
    this.readyState = 3;
    this.onclose?.(new Event('close'));
  }
}

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });

  return function Wrapper({ children }: { children: ReactNode }) {
    const factory = useCallback((url: string) => new MockWebSocket(url) as unknown as WebSocket, []);
    return (
      <QueryClientProvider client={queryClient}>
        <MarketDataProvider createWebSocket={factory}>{children}</MarketDataProvider>
      </QueryClientProvider>
    );
  };
}

function makeTicker(pair: string, price: string, pct: string) {
  return {
    stream: `${pair.toLowerCase()}@ticker`,
    data: { s: pair, c: price, P: pct, v: '1000', E: String(Date.now()) },
  };
}

function latestWs(): MockWebSocket {
  return MockWebSocket.instances.at(-1)!;
}

describe('MarketDataProvider + useMarketDataWS', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    MockWebSocket.instances = [];
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('connectionStatus becomes connected after WS open + 2s delay', () => {
    const { result } = renderHook(() => useMarketDataWS('BTCUSDT'), {
      wrapper: createWrapper(),
    });

    expect(MockWebSocket.instances.length).toBeGreaterThan(0);
    const ws = latestWs();

    void act(() => ws.simulateOpen());
    expect(result.current.connectionStatus).not.toBe('connected');

    void act(() => vi.advanceTimersByTime(2100));
    expect(result.current.connectionStatus).toBe('connected');
  });

  it('tickerData updates after WS message + throttle', () => {
    const { result } = renderHook(() => useMarketDataWS('BTCUSDT'), {
      wrapper: createWrapper(),
    });

    const ws = latestWs();
    void act(() => ws.simulateOpen());
    void act(() => vi.advanceTimersByTime(2100));

    void act(() => ws.simulateMessage(makeTicker('BTCUSDT', '43250.00', '1.5')));
    void act(() => vi.advanceTimersByTime(250));

    expect(result.current.tickerData).toBeDefined();
    expect(result.current.tickerData?.price).toBe(43250);
    expect(result.current.tickerData?.priceChangePercent).toBe(1.5);
  });

  it('unsubscribe closes WS on unmount', () => {
    const { unmount } = renderHook(() => useMarketDataWS('BTCUSDT'), {
      wrapper: createWrapper(),
    });

    const ws = latestWs();
    void act(() => ws.simulateOpen());
    unmount();

    expect(ws.closed).toBe(true);
  });

  it('becomes degraded when reconnects keep failing', () => {
    const { result } = renderHook(() => useMarketDataWS('BTCUSDT'), {
      wrapper: createWrapper(),
    });

    const ws1 = latestWs();
    void act(() => ws1.simulateOpen());
    void act(() => vi.advanceTimersByTime(2100));
    expect(result.current.connectionStatus).toBe('connected');

    void act(() => ws1.simulateClose());

    for (let i = 0; i < 10; i++) {
      void act(() => vi.advanceTimersByTime(31_000));
      const wsN = latestWs();
      if (wsN && !wsN.closed) {
        void act(() => wsN.simulateClose());
      }
    }

    void act(() => vi.advanceTimersByTime(5_000));
    expect(result.current.connectionStatus).toBe('degraded');
  });

  it('parses price strings to numbers', () => {
    const { result } = renderHook(() => useMarketDataWS('BTCUSDT'), {
      wrapper: createWrapper(),
    });

    const ws = latestWs();
    void act(() => ws.simulateOpen());
    void act(() => vi.advanceTimersByTime(2100));

    void act(() =>
      ws.simulateMessage({
        stream: 'btcusdt@ticker',
        data: { s: 'BTCUSDT', c: '43250.00', P: '-2.35', v: '15000.5', E: '1713456000000' },
      }),
    );
    void act(() => vi.advanceTimersByTime(250));

    expect(result.current.tickerData).toBeDefined();
    expect(typeof result.current.tickerData?.price).toBe('number');
    expect(typeof result.current.tickerData?.priceChangePercent).toBe('number');
    expect(typeof result.current.tickerData?.volume24h).toBe('number');
  });

  it('cleanup closes WS on Provider unmount', () => {
    const { unmount } = renderHook(() => useMarketDataWS('ETHUSDT'), {
      wrapper: createWrapper(),
    });

    const ws = latestWs();
    void act(() => ws.simulateOpen());
    unmount();

    expect(ws.closed).toBe(true);
  });

  it('throttle: rapid messages result in last value', () => {
    const { result } = renderHook(() => useMarketDataWS('BTCUSDT'), {
      wrapper: createWrapper(),
    });

    const ws = latestWs();
    void act(() => ws.simulateOpen());
    void act(() => vi.advanceTimersByTime(2100));

    void act(() => {
      ws.simulateMessage(makeTicker('BTCUSDT', '43250', '1.0'));
      ws.simulateMessage(makeTicker('BTCUSDT', '43260', '1.1'));
      ws.simulateMessage(makeTicker('BTCUSDT', '43270', '1.2'));
    });

    void act(() => vi.advanceTimersByTime(250));

    expect(result.current.tickerData?.price).toBe(43270);
  });

  it('no pair means no subscription', () => {
    const { result } = renderHook(() => useMarketDataWS(), {
      wrapper: createWrapper(),
    });

    expect(result.current.connectionStatus).toBe('disconnected');
    expect(result.current.tickerData).toBeUndefined();
    expect(MockWebSocket.instances.length).toBe(0);
  });

  it('reconnects after WS close with new WS instance', () => {
    const { result } = renderHook(() => useMarketDataWS('BTCUSDT'), {
      wrapper: createWrapper(),
    });

    const ws1 = latestWs();
    void act(() => ws1.simulateOpen());
    void act(() => vi.advanceTimersByTime(2100));
    expect(result.current.connectionStatus).toBe('connected');

    const countBefore = MockWebSocket.instances.length;
    void act(() => ws1.simulateClose());
    void act(() => vi.advanceTimersByTime(1100));

    expect(MockWebSocket.instances.length).toBeGreaterThan(countBefore);

    const ws2 = latestWs();
    void act(() => ws2.simulateOpen());
    void act(() => vi.advanceTimersByTime(2100));

    expect(result.current.connectionStatus).toBe('connected');
  });

  it('combined stream URL includes all subscribed pairs', () => {
    renderHook(
      () => {
        const btc = useMarketDataWS('BTCUSDT');
        const eth = useMarketDataWS('ETHUSDT');
        return { btc, eth };
      },
      { wrapper: createWrapper() },
    );

    const ws = latestWs();
    expect(ws.url).toContain('btcusdt@ticker');
    expect(ws.url).toContain('ethusdt@ticker');
  });
});

import { type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';

import { MarketDataContext, type ConnectionStatus, type MarketDataContextValue, type TickerData } from './market-data-context';

const WS_BASE = 'wss://stream.binance.com:9443/stream?streams=';
const THROTTLE_MS = 200;
const RECONNECT_INITIAL_MS = 1_000;
const RECONNECT_MAX_MS = 30_000;
const MAX_RETRIES = 10;
const DEGRADED_DELAY_MS = 3_000;
const CONNECTED_DELAY_MS = 2_000;

interface MarketDataProviderProps {
  children: ReactNode;
  createWebSocket?: (url: string) => WebSocket;
}

export function MarketDataProvider({ children, createWebSocket }: MarketDataProviderProps) {
  const queryClient = useQueryClient();
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('disconnected');

  const subscriptionsRef = useRef(new Map<string, number>());
  const priceMapRef = useRef<Record<string, TickerData>>({});
  const listenersRef = useRef(new Map<string, Set<() => void>>());

  const wsRef = useRef<WebSocket | null>(null);
  const retryCountRef = useRef(0);
  const retryTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const degradedTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const connectedTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const throttleTimersRef = useRef(new Map<string, ReturnType<typeof setTimeout>>());
  const isPausedRef = useRef(false);
  const mountedRef = useRef(true);

  const [subscriptionVersion, setSubscriptionVersion] = useState(0);

  const notifyListeners = useCallback((pair: string) => {
    const set = listenersRef.current.get(pair);
    if (set) {
      for (const cb of set) cb();
    }
  }, []);

  const clearAllTimers = useCallback(() => {
    if (retryTimerRef.current) {
      clearTimeout(retryTimerRef.current);
      retryTimerRef.current = null;
    }
    if (degradedTimerRef.current) {
      clearTimeout(degradedTimerRef.current);
      degradedTimerRef.current = null;
    }
    if (connectedTimerRef.current) {
      clearTimeout(connectedTimerRef.current);
      connectedTimerRef.current = null;
    }
    for (const timer of throttleTimersRef.current.values()) {
      clearTimeout(timer);
    }
    throttleTimersRef.current.clear();
  }, []);

  const closeWs = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.onopen = null;
      wsRef.current.onmessage = null;
      wsRef.current.onerror = null;
      wsRef.current.onclose = null;
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  const connectWs = useCallback(() => {
    const subs = subscriptionsRef.current;
    if (subs.size === 0) {
      closeWs();
      clearAllTimers();
      setConnectionStatus('disconnected');
      return;
    }

    closeWs();
    if (degradedTimerRef.current) {
      clearTimeout(degradedTimerRef.current);
      degradedTimerRef.current = null;
    }
    if (connectedTimerRef.current) {
      clearTimeout(connectedTimerRef.current);
      connectedTimerRef.current = null;
    }

    const streams = [...subs.keys()].map((p) => `${p.toLowerCase()}@ticker`).join('/');
    const url = `${WS_BASE}${streams}`;

    setConnectionStatus('connecting');
    console.info('[WSMarketData] connecting', url);

    const factory = createWebSocket ?? ((u: string) => new WebSocket(u));
    const ws = factory(url);
    wsRef.current = ws;

    ws.onopen = () => {
      if (!mountedRef.current) return;
      console.info('[WSMarketData] onopen');
      retryCountRef.current = 0;

      connectedTimerRef.current = setTimeout(() => {
        if (mountedRef.current) {
          setConnectionStatus('connected');
          console.info('[WSMarketData] status → connected');
        }
      }, CONNECTED_DELAY_MS);
    };

    ws.onmessage = (event: MessageEvent) => {
      if (!mountedRef.current || isPausedRef.current) return;

      try {
        const msg = JSON.parse(event.data as string) as { stream?: string; data?: Record<string, unknown> };
        const d = msg.data;
        if (!d) return;

        const rawPair = d.s;
        if (typeof rawPair !== 'string' || !rawPair) return;
        const pair = rawPair.toUpperCase();
        if (!pair) return;

        const parsed: TickerData = {
          pair,
          price: Number(d.c),
          priceChangePercent: Number(d.P),
          volume24h: Number(d.v),
          ts: Number(d.E),
        };

        const existing = throttleTimersRef.current.get(pair);
        if (existing) clearTimeout(existing);

        throttleTimersRef.current.set(
          pair,
          setTimeout(() => {
            priceMapRef.current[pair] = parsed;
            notifyListeners(pair);
            throttleTimersRef.current.delete(pair);
          }, THROTTLE_MS),
        );
      } catch {
        // malformed message, ignore
      }
    };

    const handleDisconnect = () => {
      if (!mountedRef.current) return;
      console.info('[WSMarketData] disconnected, starting degraded timer');

      if (connectedTimerRef.current) {
        clearTimeout(connectedTimerRef.current);
        connectedTimerRef.current = null;
      }

      if (!degradedTimerRef.current) {
        degradedTimerRef.current = setTimeout(() => {
          if (mountedRef.current) {
            setConnectionStatus('degraded');
            console.info('[WSMarketData] status → degraded');
          }
          degradedTimerRef.current = null;
        }, DEGRADED_DELAY_MS);
      }

      if (retryCountRef.current < MAX_RETRIES) {
        setConnectionStatus('reconnecting');
        const delay = Math.min(RECONNECT_INITIAL_MS * 2 ** retryCountRef.current, RECONNECT_MAX_MS);
        retryCountRef.current += 1;
        console.info(`[WSMarketData] reconnect attempt ${String(retryCountRef.current)} in ${String(delay)}ms`);

        retryTimerRef.current = setTimeout(() => {
          if (mountedRef.current && subscriptionsRef.current.size > 0) {
            connectWs();
          }
        }, delay);
      }
    };

    ws.onerror = () => {
      console.info('[WSMarketData] onerror');
    };

    ws.onclose = () => {
      console.info('[WSMarketData] onclose');
      handleDisconnect();
    };
  }, [closeWs, clearAllTimers, createWebSocket, notifyListeners]);

  // Visibility handling (FR-020)
  useEffect(() => {
    const handler = () => {
      if (document.visibilityState === 'hidden') {
        isPausedRef.current = true;
      } else {
        isPausedRef.current = false;
        void queryClient.invalidateQueries({ queryKey: ['portfolio-snapshot'] });
      }
    };
    document.addEventListener('visibilitychange', handler);
    return () => document.removeEventListener('visibilitychange', handler);
  }, [queryClient]);

  // WS lifecycle tied to subscription changes
  useEffect(() => {
    connectWs();
  }, [subscriptionVersion, connectWs]);

  // Cleanup on unmount
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      closeWs();
      clearAllTimers();
    };
  }, [closeWs, clearAllTimers]);

  const subscribe = useCallback((pair: string) => {
    const normalized = pair.toUpperCase();
    const subs = subscriptionsRef.current;
    const count = subs.get(normalized) ?? 0;
    subs.set(normalized, count + 1);
    if (count === 0) {
      setSubscriptionVersion((v) => v + 1);
    }
  }, []);

  const unsubscribe = useCallback((pair: string) => {
    const normalized = pair.toUpperCase();
    const subs = subscriptionsRef.current;
    const count = subs.get(normalized) ?? 0;
    if (count <= 1) {
      subs.delete(normalized);
      setSubscriptionVersion((v) => v + 1);
    } else {
      subs.set(normalized, count - 1);
    }
  }, []);

  const getPrice = useCallback((pair: string): TickerData | undefined => {
    return priceMapRef.current[pair.toUpperCase()];
  }, []);

  const subscribeToPrice = useCallback((pair: string, callback: () => void): (() => void) => {
    const normalized = pair.toUpperCase();
    let set = listenersRef.current.get(normalized);
    if (!set) {
      set = new Set();
      listenersRef.current.set(normalized, set);
    }
    set.add(callback);
    return () => {
      set.delete(callback);
      if (set.size === 0) {
        listenersRef.current.delete(normalized);
      }
    };
  }, []);

  const value: MarketDataContextValue = useMemo(
    () => ({
      connectionStatus,
      subscribe,
      unsubscribe,
      getPrice,
      subscribeToPrice,
    }),
    [connectionStatus, subscribe, unsubscribe, getPrice, subscribeToPrice],
  );

  return <MarketDataContext.Provider value={value}>{children}</MarketDataContext.Provider>;
}

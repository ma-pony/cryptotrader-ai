import { useCallback, useContext, useEffect, useSyncExternalStore } from 'react';

import { MarketDataContext, type ConnectionStatus, type TickerData } from '@/contexts/market-data';

export function useMarketDataWS(pair?: string): {
  connectionStatus: ConnectionStatus;
  tickerData: TickerData | undefined;
} {
  const ctx = useContext(MarketDataContext);
  if (!ctx) {
    throw new Error('useMarketDataWS must be used within <MarketDataProvider>');
  }

  const { connectionStatus, subscribe: ctxSubscribe, unsubscribe: ctxUnsubscribe, getPrice, subscribeToPrice } = ctx;

  useEffect(() => {
    if (!pair) return;
    ctxSubscribe(pair);
    return () => ctxUnsubscribe(pair);
  }, [pair, ctxSubscribe, ctxUnsubscribe]);

  const subscribeFn = useCallback(
    (onStoreChange: () => void) => {
      if (!pair) return () => {};
      return subscribeToPrice(pair, onStoreChange);
    },
    [pair, subscribeToPrice],
  );

  const getSnapshot = useCallback(() => {
    if (!pair) return undefined;
    return getPrice(pair);
  }, [pair, getPrice]);

  const tickerData = useSyncExternalStore(subscribeFn, getSnapshot, getSnapshot);

  return { connectionStatus, tickerData };
}

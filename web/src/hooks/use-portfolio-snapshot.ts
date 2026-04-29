import { useContext } from 'react';
import { useQuery } from '@tanstack/react-query';

import { MarketDataContext } from '@/contexts/market-data';
import { useAdaptivePolling } from '@/hooks/use-adaptive-polling';
import { useMarketDataWS } from '@/hooks/use-market-data-ws';
import { apiClient } from '@/lib/api-client';
import { PortfolioSchema } from '@/types/api.schema';
import type { Portfolio } from '@/types/api';

export const usePortfolioSnapshot = () => {
  const { connectionStatus } = useMarketDataWS();
  const ctx = useContext(MarketDataContext);
  const getPrice = ctx?.getPrice;

  const { refetchInterval } = useAdaptivePolling({
    wsStatus: connectionStatus,
    priceChangePercent: undefined,
  });

  const query = useQuery({
    queryKey: ['portfolio-snapshot'],
    queryFn: () => apiClient.get('/api/portfolio/snapshot', PortfolioSchema),
    refetchInterval: refetchInterval === false ? false : refetchInterval,
    select: (data): Portfolio => {
      if (connectionStatus !== 'connected' || !getPrice) return data;

      let equity = data.cash;
      const positions = data.positions.map((pos) => {
        const pairKey = pos.pair.replace('/', '');
        const ticker = getPrice(pairKey);
        if (!ticker) {
          equity += pos.avg_price * pos.size;
          return pos;
        }
        const latestPrice = ticker.price;
        const unrealizedPnl =
          pos.side === 'long'
            ? (latestPrice - pos.avg_price) * pos.size
            : (pos.avg_price - latestPrice) * pos.size;
        const unrealizedPnlPct = pos.avg_price > 0 ? unrealizedPnl / (pos.avg_price * pos.size) : 0;
        equity += latestPrice * pos.size;
        return {
          ...pos,
          unrealized_pnl: unrealizedPnl,
          unrealized_pnl_pct: unrealizedPnlPct,
        };
      });

      return { ...data, equity, positions };
    },
  });

  return { ...query, connectionStatus };
};

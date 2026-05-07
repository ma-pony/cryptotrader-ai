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

      // Mirror backend `_equity_contribution` (175089c, 2026-05-06): perp /
      // swap positions contribute their unrealized P&L only — NOT the
      // signed notional `size * price`. Margin already lives in cash; adding
      // notional double-counts (and signed notional makes a short look
      // like a -$50K liability, producing the "总权益 $50,159" bug
      // visible in v0.x dashboards). Spot positions are real assets so we
      // keep `size * latest_price` for them.
      const isDerivative = (pos: Portfolio['positions'][number]): boolean =>
        pos.market_type === 'swap' || pos.market_type === 'future' || pos.pair.includes(':');

      let equity = data.cash;
      const positions = data.positions.map((pos) => {
        const pairKey = pos.pair.replace('/', '');
        const ticker = getPrice(pairKey);
        const sign = pos.side === 'short' ? -1 : 1;
        const absSize = Math.abs(pos.size);

        if (!ticker) {
          // Fallback when WS has no live tick: trust server's unrealized_pnl.
          equity += isDerivative(pos) ? pos.unrealized_pnl : pos.avg_price * pos.size;
          return pos;
        }

        const latestPrice = ticker.price;
        const unrealizedPnl = (latestPrice - pos.avg_price) * absSize * sign;
        const unrealizedPnlPct =
          pos.avg_price > 0 && absSize > 0
            ? unrealizedPnl / (pos.avg_price * absSize)
            : 0;

        equity += isDerivative(pos)
          ? unrealizedPnl // perp: only the P&L moves equity, margin stays in cash
          : latestPrice * pos.size; // spot: full mark-to-market

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

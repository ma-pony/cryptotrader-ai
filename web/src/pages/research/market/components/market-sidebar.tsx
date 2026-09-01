import { useQuery } from '@tanstack/react-query';
import { type FC } from 'react';
import { useTranslation } from 'react-i18next';
import { z } from 'zod';

import { WSStatusIndicator } from '@/components/ws-status-indicator';
import { Card, CardContent } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { useAdaptivePolling } from '@/hooks/use-adaptive-polling';
import { useMarketDataWS } from '@/hooks/use-market-data-ws';
import { apiClient } from '@/lib/api-client';

const MarketDataSchema = z.object({
  funding_rate: z.number().nullable().default(null),
  open_interest: z.number().nullable().default(null),
  liquidations_24h: z
    .object({
      long: z.number().default(0),
      short: z.number().default(0),
    })
    .default({ long: 0, short: 0 }),
});

type MarketData = z.output<typeof MarketDataSchema>;

interface MarketSidebarProps {
  pair: string;
  exchange: 'binance' | 'okx';
}

const formatUSD = (value: number) =>
  new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    notation: 'compact',
    maximumFractionDigits: 1,
  }).format(value);

const fundingRateTone = (rate: number | null): { color: string; label: string } => {
  if (rate == null) return { color: 'hsl(var(--muted-foreground))', label: '—' };
  const pct = Math.abs(rate);
  if (pct > 0.0003) return { color: 'var(--trade-short)', label: '偏热' };
  if (pct > 0.0002) return { color: 'var(--amber-500)', label: '接近阈值' };
  return { color: 'var(--trade-long)', label: '中性' };
};

const StatLabel = ({ children }: { children: React.ReactNode }) => (
  <div className="text-[10px] uppercase tracking-wider font-medium text-muted-foreground">
    {children}
  </div>
);

export const MarketSidebar: FC<MarketSidebarProps> = ({ pair, exchange }) => {
  const { t } = useTranslation('market');
  const encodedPair = pair.replace('/', '-');

  const wsPair = exchange === 'binance' ? pair.replace('/', '') : undefined;
  const { connectionStatus, tickerData } = useMarketDataWS(wsPair);

  const { refetchInterval } = useAdaptivePolling({
    wsStatus: exchange === 'binance' ? connectionStatus : 'disconnected',
    priceChangePercent: tickerData?.priceChangePercent,
  });

  const effectiveInterval = exchange === 'okx' ? 30_000 : refetchInterval;

  const { data, isLoading } = useQuery({
    queryKey: ['market-data', pair, exchange],
    queryFn: () => apiClient.get(`/api/market/${encodedPair}?exchange=${exchange}`, MarketDataSchema),
    refetchInterval: effectiveInterval,
  });

  const loading = isLoading || !data;
  const fundingTone = fundingRateTone(data?.funding_rate ?? null);

  return (
    <div className="space-y-3">
      <Card>
        <CardContent className="flex flex-col gap-2 p-4">
          <div className="flex items-center justify-between">
            <StatLabel>
              {t('side_panel.funding_rate', { defaultValue: '资金费率 · 8h' })}
            </StatLabel>
            {exchange === 'binance' ? (
              <WSStatusIndicator
                status={connectionStatus}
                refetchInterval={typeof effectiveInterval === 'number' ? effectiveInterval : undefined}
              />
            ) : null}
          </div>
          {loading ? (
            <Skeleton className="h-7 w-24" />
          ) : (
            <>
              <div
                className="font-mono text-lg font-semibold tracking-tight leading-none"
                style={{ color: fundingTone.color }}
              >
                {data.funding_rate != null
                  ? `${data.funding_rate >= 0 ? '+' : ''}${(data.funding_rate * 100).toFixed(4)}%`
                  : '—'}
              </div>
              <div className="text-[10px] text-muted-foreground">
                阈值 0.0300% · {fundingTone.label}
              </div>
            </>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardContent className="flex flex-col gap-2 p-4">
          <StatLabel>{t('side_panel.open_interest', { defaultValue: '持仓量 (OI)' })}</StatLabel>
          {loading ? (
            <Skeleton className="h-7 w-24" />
          ) : (
            <>
              <div className="font-mono text-lg font-semibold tracking-tight leading-none">
                {data.open_interest != null ? formatUSD(data.open_interest) : '—'}
              </div>
              <div className="text-[10px] text-muted-foreground">24h 快照</div>
            </>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardContent className="flex flex-col gap-2 p-4">
          <StatLabel>{t('side_panel.liquidations', { defaultValue: '24h 清算' })}</StatLabel>
          {loading ? <Skeleton className="h-14 w-full" /> : <LiquidationBar data={data} />}
        </CardContent>
      </Card>
    </div>
  );
};

const LiquidationBar: FC<{ data: MarketData }> = ({ data }) => {
  const { liquidations_24h } = data;
  const total = liquidations_24h.long + liquidations_24h.short;
  const longPct = total > 0 ? (liquidations_24h.long / total) * 100 : 50;

  return (
    <div className="space-y-2">
      <div className="flex h-16 items-end gap-2">
        <div className="flex-1 text-center">
          <div
            className="rounded border border-trade-long/60 bg-trade-long-soft"
            style={{ height: `${longPct}%`, minHeight: 4 }}
          />
          <div className="mt-1 font-mono text-[11px] text-trade-long">
            {formatUSD(liquidations_24h.long)}
          </div>
          <div className="text-[10px] text-muted-foreground">多头爆仓</div>
        </div>
        <div className="flex-1 text-center">
          <div
            className="rounded border border-trade-short/60 bg-trade-short-soft"
            style={{ height: `${100 - longPct}%`, minHeight: 4 }}
          />
          <div className="mt-1 font-mono text-[11px] text-trade-short">
            {formatUSD(liquidations_24h.short)}
          </div>
          <div className="text-[10px] text-muted-foreground">空头爆仓</div>
        </div>
      </div>
    </div>
  );
};

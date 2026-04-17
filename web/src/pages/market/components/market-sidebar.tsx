import { useQuery } from '@tanstack/react-query';
import { type FC } from 'react';
import { useTranslation } from 'react-i18next';
import { z } from 'zod';

import { apiClient } from '@/lib/api-client';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';

const MarketDataSchema = z.object({
  funding_rate: z.number().nullable().default(null),
  open_interest: z.number().nullable().default(null),
  liquidations_24h: z.object({
    long: z.number().default(0),
    short: z.number().default(0),
  }).default({ long: 0, short: 0 }),
});

type MarketData = z.output<typeof MarketDataSchema>;

interface MarketSidebarProps {
  pair: string;
  exchange: 'binance' | 'okx';
}

const formatNumber = (value: number | null, opts?: { style?: string; maximumFractionDigits?: number }) => {
  if (value === null) return '—';
  return new Intl.NumberFormat('en-US', {
    maximumFractionDigits: opts?.maximumFractionDigits ?? 2,
    ...(opts?.style === 'percent' ? { style: 'percent' } : {}),
  }).format(opts?.style === 'percent' ? value : value);
};

const formatUSD = (value: number) =>
  new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', notation: 'compact', maximumFractionDigits: 1 }).format(value);

export const MarketSidebar: FC<MarketSidebarProps> = ({ pair, exchange }) => {
  const { t } = useTranslation('market');
  const encodedPair = pair.replace('/', '-');

  const { data, isLoading } = useQuery({
    queryKey: ['market-data', pair, exchange],
    queryFn: () => apiClient.get(`/api/market/${encodedPair}?exchange=${exchange}`, MarketDataSchema),
    refetchInterval: 30_000,
  });

  const loading = isLoading || !data;

  return (
    <div className="space-y-4">
      {/* Funding rate */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">{t('side_panel.funding_rate')}</CardTitle>
        </CardHeader>
        <CardContent>
          {loading ? (
            <Skeleton className="h-8 w-24" />
          ) : (
            <p className="text-2xl font-bold tabular-nums">
              {formatNumber(data.funding_rate, { style: 'percent', maximumFractionDigits: 4 })}
            </p>
          )}
        </CardContent>
      </Card>

      {/* Open interest */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">{t('side_panel.open_interest')}</CardTitle>
        </CardHeader>
        <CardContent>
          {loading ? (
            <Skeleton className="h-8 w-24" />
          ) : (
            <p className="text-2xl font-bold tabular-nums">
              {data.open_interest !== null ? formatUSD(data.open_interest) : '—'}
            </p>
          )}
        </CardContent>
      </Card>

      {/* Liquidations 24h */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">{t('side_panel.liquidations')}</CardTitle>
        </CardHeader>
        <CardContent>
          {loading ? (
            <Skeleton className="h-12 w-full" />
          ) : (
            <LiquidationBar data={data} />
          )}
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
    <div className="space-y-1.5">
      <div className="flex justify-between text-xs text-muted-foreground">
        <span>Long {formatUSD(liquidations_24h.long)}</span>
        <span>Short {formatUSD(liquidations_24h.short)}</span>
      </div>
      <div className="flex h-3 overflow-hidden rounded-full">
        <div className="bg-success" style={{ width: `${String(longPct)}%` }} />
        <div className="bg-destructive flex-1" />
      </div>
    </div>
  );
};

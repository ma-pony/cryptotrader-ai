import { ChevronRight } from 'lucide-react';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

import { PairBadge } from '@/components/PairBadge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { DirChip } from '@/components/ui/dir-chip';
import { Skeleton } from '@/components/ui/skeleton';
import { StatusPill } from '@/components/ui/status-pill';
import { useMarketDataWS } from '@/hooks/use-market-data-ws';
import { cn } from '@/lib/cn';
import { formatCurrency } from '@/lib/format';
import type { Position } from '@/types/api';

const fmtAgo = (iso: string | null | undefined): string => {
  if (!iso) return '—';
  const diffMs = Date.now() - new Date(iso).getTime();
  if (!Number.isFinite(diffMs) || diffMs < 0) return '刚开仓';
  const mins = Math.floor(diffMs / 60_000);
  if (mins < 60) return `${mins}m`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ${mins % 60}m`;
  const days = Math.floor(hours / 24);
  return `${days}d ${hours % 24}h`;
};

interface PositionRowProps {
  pos: Position;
}

const PositionRow = memo(function PositionRow({ pos }: PositionRowProps) {
  // Strip ccxt derivative suffix and the slash → "BTC/USDT:USDT" → "BTCUSDT"
  // for ticker subscription (spec 013: only spot tickers exist on the WS feed).
  const colon = pos.pair.indexOf(':');
  const spotPair = colon === -1 ? pos.pair : pos.pair.slice(0, colon);
  const pairKey = spotPair.replace('/', '');
  const { tickerData } = useMarketDataWS(pairKey);

  let pnl = pos.unrealized_pnl;
  let pnlPct = pos.unrealized_pnl_pct;
  let currentPrice: number | null = null;

  if (tickerData) {
    currentPrice = tickerData.price;
    pnl =
      pos.side === 'long'
        ? (tickerData.price - pos.avg_price) * pos.size
        : (pos.avg_price - tickerData.price) * pos.size;
    pnlPct = pos.avg_price > 0 ? pnl / (pos.avg_price * pos.size) : 0;
  }

  const notional = pos.avg_price * pos.size;
  const pnlTone: 'long' | 'short' = pnl >= 0 ? 'long' : 'short';

  return (
    <div
      className={cn(
        'grid items-center gap-3 border-b border-border px-4 py-3 transition-colors hover:bg-muted/40',
        'grid-cols-[minmax(140px,140px)_minmax(72px,72px)_1fr_minmax(90px,90px)_minmax(90px,90px)_minmax(24px,24px)]',
      )}
    >
      <PairBadge pair={pos.pair} pairDisplay={pos.pair_display} marketType={pos.market_type} />
      <DirChip dir={pos.side} />
      <div className="min-w-0">
        <div className="truncate text-xs text-muted-foreground">
          {pos.size} @ <span className="font-mono">{formatCurrency(pos.avg_price)}</span>
          {currentPrice != null ? (
            <>
              {' '}
              → <span className="font-mono">{formatCurrency(currentPrice)}</span>
            </>
          ) : null}
        </div>
        <div className="mt-0.5 text-[10px] text-muted-foreground">
          开仓 {fmtAgo(pos.opened_at)}
        </div>
      </div>
      <div className="text-right font-mono text-xs tabular-nums">
        ${notional.toLocaleString('en-US', { maximumFractionDigits: 0 })}
      </div>
      <div className="text-right">
        <div
          className={cn(
            'font-mono text-xs font-medium',
            pnlTone === 'long' ? 'text-trade-long' : 'text-trade-short',
          )}
        >
          {pnl >= 0 ? '+' : '-'}${Math.abs(pnl).toLocaleString('en-US', { maximumFractionDigits: 0 })}
        </div>
        <div
          className={cn(
            'font-mono text-[10px]',
            pnlTone === 'long' ? 'text-trade-long' : 'text-trade-short',
          )}
        >
          {pnl >= 0 ? '+' : ''}
          {(pnlPct * 100).toFixed(2)}%
        </div>
      </div>
      <ChevronRight size={14} className="text-muted-foreground" />
    </div>
  );
});

interface PositionsTableProps {
  positions: Position[] | undefined;
  isLoading: boolean;
}

export const PositionsTable = ({ positions, isLoading }: PositionsTableProps) => {
  const { t } = useTranslation('dashboard');

  if (isLoading) {
    return (
      <Card>
        <CardContent className="space-y-3 p-4">
          {[0, 1, 2].map((i) => (
            <Skeleton key={i} className="h-12 w-full" />
          ))}
        </CardContent>
      </Card>
    );
  }

  const sorted = [...(positions ?? [])].sort(
    (a, b) => Math.abs(b.unrealized_pnl) - Math.abs(a.unrealized_pnl),
  );

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between p-4 pb-2">
        <CardTitle className="text-sm">
          {t('positions.title', { defaultValue: '当前仓位' })}
          <span className="ml-2 text-[11px] font-normal text-muted-foreground">
            {sorted.length} 个
          </span>
        </CardTitle>
        <StatusPill tone="cyan" live>
          {t('positions.realtime', { defaultValue: '实时' })}
        </StatusPill>
      </CardHeader>
      <CardContent className="p-0">
        {sorted.length === 0 ? (
          <p className="p-4 text-sm text-muted-foreground">
            {t('positions.empty', { defaultValue: '暂无持仓' })}
          </p>
        ) : (
          <>
            <div
              className={cn(
                'grid items-center gap-3 border-b border-border bg-muted/30 px-4 py-2',
                'grid-cols-[minmax(140px,140px)_minmax(72px,72px)_1fr_minmax(90px,90px)_minmax(90px,90px)_minmax(24px,24px)]',
                'text-[10px] uppercase tracking-wider font-medium text-muted-foreground',
              )}
            >
              <div>{t('positions.pair', { defaultValue: '交易对' })}</div>
              <div>{t('positions.side', { defaultValue: '方向' })}</div>
              <div>{t('positions.size_price', { defaultValue: '数量 · 均价' })}</div>
              <div className="text-right">{t('positions.notional', { defaultValue: '净值' })}</div>
              <div className="text-right">{t('positions.pnl', { defaultValue: '盈亏' })}</div>
              <div />
            </div>
            <div>
              {sorted.map((pos) => (
                <PositionRow key={pos.pair} pos={pos} />
              ))}
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
};

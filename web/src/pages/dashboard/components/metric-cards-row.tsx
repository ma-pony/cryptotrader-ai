import { memo } from 'react';
import { useTranslation } from 'react-i18next';

import type { ConnectionStatus } from '@/contexts/market-data';
import { useEquityCurve } from '@/hooks/use-equity-curve';
import { formatCurrency, formatPercent } from '@/lib/format';
import type { Portfolio } from '@/types/api';

import { MetricCard } from './metric-card';

interface MetricCardsRowProps {
  data: Portfolio | undefined;
  isLoading: boolean;
  connectionStatus?: ConnectionStatus;
}

export const MetricCardsRow = memo(function MetricCardsRow({
  data,
  isLoading,
  connectionStatus,
}: MetricCardsRowProps) {
  const { t } = useTranslation('dashboard');
  const equity = useEquityCurve('30d');
  const trendEquity = equity.data?.points.map((p) => p.equity);

  const pnlDir: 'up' | 'down' | 'neutral' = !data
    ? 'neutral'
    : data.pnl_24h > 0
      ? 'up'
      : data.pnl_24h < 0
        ? 'down'
        : 'neutral';
  const totalReturnDir: 'up' | 'down' | 'neutral' = !data
    ? 'neutral'
    : data.total_return > 0
      ? 'up'
      : data.total_return < 0
        ? 'down'
        : 'neutral';
  const isLive = connectionStatus === 'connected';

  return (
    <div
      className="grid grid-cols-4 gap-3"
      role="region"
      aria-label={t('metrics.region', { defaultValue: '核心指标' })}
    >
      <MetricCard
        label={
          <span className="inline-flex items-center gap-1.5">
            {t('metrics.total_equity', { defaultValue: '总权益' })}
            {isLive ? (
              <span
                className="h-1.5 w-1.5 rounded-full bg-trade-long animate-ct-pulse"
                aria-label={t('metrics.realtime', { defaultValue: '实时数据' })}
              />
            ) : null}
          </span>
        }
        value={data ? formatCurrency(data.equity) : '--'}
        delta={data ? formatPercent(data.pnl_24h_pct) : undefined}
        deltaDirection={pnlDir}
        trend={trendEquity}
        trendColor="var(--amber-500)"
        isLoading={isLoading}
      />
      <MetricCard
        label={t('metrics.available_cash', { defaultValue: '可用现金' })}
        value={data ? formatCurrency(data.cash) : '--'}
        isLoading={isLoading}
      />
      <MetricCard
        label={t('metrics.sharpe', { defaultValue: 'Sharpe (90D)' })}
        value={data?.sharpe_90d != null ? data.sharpe_90d.toFixed(2) : '—'}
        sub={data ? `胜率 ${data.win_rate != null ? `${(data.win_rate * 100).toFixed(1)}%` : '—'} · ${data.total_trades} 笔` : undefined}
        isLoading={isLoading}
      />
      <MetricCard
        label={t('metrics.total_return', { defaultValue: '总收益' })}
        value={
          data
            ? `${data.total_return >= 0 ? '+' : ''}${formatCurrency(data.total_return)}`
            : '--'
        }
        delta={data ? formatPercent(data.total_return_pct) : undefined}
        deltaDirection={totalReturnDir}
        sub={
          data
            ? `30D 已实现 ${data.realized_pnl_30d >= 0 ? '+' : ''}${formatCurrency(data.realized_pnl_30d)} · 回撤 ${formatPercent(-data.drawdown)}`
            : undefined
        }
        isLoading={isLoading}
      />
    </div>
  );
});

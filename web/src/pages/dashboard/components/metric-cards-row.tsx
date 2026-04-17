import { useTranslation } from 'react-i18next';

import { formatCurrency, formatPercent, pnlClass } from '@/lib/format';
import type { Portfolio } from '@/types/api';

import { MetricCard } from './metric-card';

interface MetricCardsRowProps {
  data: Portfolio | undefined;
  isLoading: boolean;
}

export const MetricCardsRow = ({ data, isLoading }: MetricCardsRowProps) => {
  const { t } = useTranslation('dashboard');

  const pnlDir = !data ? 'neutral' : data.pnl_24h > 0 ? 'up' : data.pnl_24h < 0 ? 'down' : 'neutral';
  const ddDir = !data ? 'neutral' : data.drawdown > 0 ? 'down' : 'neutral';
  const ddClass = pnlClass(-Math.abs(data?.drawdown ?? 0));
  void ddClass; // used for aria semantics via deltaDirection

  return (
    <div className="grid grid-cols-4 gap-4" role="region" aria-label={t('metrics.region', { defaultValue: '核心指标' })}>
      <MetricCard
        label={t('metrics.total_equity', { defaultValue: '总权益' })}
        value={data ? formatCurrency(data.equity) : '--'}
        isLoading={isLoading}
      />
      <MetricCard
        label={t('metrics.available_cash', { defaultValue: '可用现金' })}
        value={data ? formatCurrency(data.cash) : '--'}
        isLoading={isLoading}
      />
      <MetricCard
        label={t('metrics.pnl_24h', { defaultValue: '24h PnL' })}
        value={data ? formatCurrency(data.pnl_24h) : '--'}
        delta={data ? formatPercent(data.pnl_24h_pct) : undefined}
        deltaDirection={pnlDir}
        isLoading={isLoading}
      />
      <MetricCard
        label={t('metrics.drawdown', { defaultValue: '当前回撤' })}
        value={data ? formatPercent(-data.drawdown) : '--'}
        deltaDirection={ddDir}
        isLoading={isLoading}
      />
    </div>
  );
};

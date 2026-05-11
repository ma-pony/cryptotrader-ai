import { useTranslation } from 'react-i18next';

import { Card, CardContent } from '@/components/ui/card';
import { cn } from '@/lib/cn';
import { formatPercent } from '@/lib/format';
import type { BacktestMetrics } from '@/types/api';

interface Props {
  metrics: BacktestMetrics;
}

const MetricItem = ({
  label,
  value,
  tone = 'neutral',
}: {
  label: string;
  value: string;
  tone?: 'neutral' | 'long' | 'short';
}) => (
  <Card>
    <CardContent className="flex flex-col gap-1.5 p-4">
      <div className="text-[10px] uppercase tracking-wider font-medium text-muted-foreground">
        {label}
      </div>
      <div
        className={cn(
          'font-mono text-xl font-semibold tabular-nums leading-none',
          tone === 'long' && 'text-trade-long',
          tone === 'short' && 'text-trade-short',
        )}
      >
        {value}
      </div>
    </CardContent>
  </Card>
);

export const BacktestMetricsRow = ({ metrics }: Props) => {
  const { t } = useTranslation('backtest');
  const returnTone: 'long' | 'short' | 'neutral' =
    metrics.total_return_pct > 0 ? 'long' : metrics.total_return_pct < 0 ? 'short' : 'neutral';
  const sharpeTone: 'long' | 'short' | 'neutral' =
    metrics.sharpe > 1 ? 'long' : metrics.sharpe < 0 ? 'short' : 'neutral';

  return (
    <div className="grid grid-cols-5 gap-3">
      <MetricItem
        label={t('metrics.total_return', { defaultValue: '收益率' })}
        value={formatPercent(metrics.total_return_pct)}
        tone={returnTone}
      />
      <MetricItem
        label={t('metrics.sharpe', { defaultValue: 'Sharpe' })}
        value={metrics.sharpe.toFixed(2)}
        tone={sharpeTone}
      />
      <MetricItem
        label={t('metrics.max_drawdown', { defaultValue: '最大回撤' })}
        value={formatPercent(-Math.abs(metrics.max_drawdown_pct))}
        tone="short"
      />
      <MetricItem
        label={t('metrics.win_rate', { defaultValue: '胜率' })}
        value={formatPercent(metrics.win_rate, 1)}
      />
      <MetricItem
        label={t('metrics.trades', { defaultValue: '交易数' })}
        value={String(metrics.trades_count)}
      />
    </div>
  );
};

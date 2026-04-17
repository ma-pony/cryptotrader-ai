import { useTranslation } from 'react-i18next';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { cn } from '@/lib/cn';
import { formatPercent } from '@/lib/format';
import type { BacktestMetrics } from '@/types/api';

interface Props {
  metrics: BacktestMetrics;
}

const MetricItem = ({ label, value, colored }: { label: string; value: string; colored?: boolean }) => (
  <Card>
    <CardHeader className="p-3 pb-1">
      <CardTitle className="text-xs text-muted-foreground">{label}</CardTitle>
    </CardHeader>
    <CardContent className="p-3 pt-0">
      <span className={cn('text-lg font-semibold tabular-nums', colored && (parseFloat(value) >= 0 ? 'text-success' : 'text-destructive'))}>
        {value}
      </span>
    </CardContent>
  </Card>
);

export const BacktestMetricsRow = ({ metrics }: Props) => {
  const { t } = useTranslation('backtest');

  return (
    <div className="grid grid-cols-5 gap-3">
      <MetricItem label={t('metrics.total_return')} value={formatPercent(metrics.total_return_pct)} colored />
      <MetricItem label={t('metrics.sharpe')} value={metrics.sharpe.toFixed(2)} />
      <MetricItem label={t('metrics.max_drawdown')} value={formatPercent(-Math.abs(metrics.max_drawdown_pct))} />
      <MetricItem label={t('metrics.win_rate')} value={formatPercent(metrics.win_rate * 100, 1)} />
      <MetricItem label={t('metrics.trades')} value={String(metrics.trades_count)} />
    </div>
  );
};

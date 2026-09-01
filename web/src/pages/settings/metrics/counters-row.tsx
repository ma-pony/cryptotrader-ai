import { useTranslation } from 'react-i18next';

import { Card, CardContent } from '@/components/ui/card';
import type { MetricsCounters } from '@/types/api';

interface Props {
  counters: MetricsCounters;
}

const Kpi = ({ label, value, sub }: { label: string; value: number | string; sub?: string }) => (
  <Card>
    <CardContent className="flex flex-col gap-1.5 p-4">
      <div className="text-[10px] uppercase tracking-wider font-medium text-muted-foreground">
        {label}
      </div>
      <div className="font-mono text-xl font-semibold tabular-nums leading-none">{value}</div>
      {sub ? <div className="text-[10px] text-muted-foreground">{sub}</div> : null}
    </CardContent>
  </Card>
);

export const CountersRow = ({ counters }: Props) => {
  const { t } = useTranslation('metrics');
  const failRate =
    counters.orders_placed > 0
      ? ((counters.orders_failed / counters.orders_placed) * 100).toFixed(1)
      : '0.0';

  return (
    <div className="grid grid-cols-5 gap-3">
      <Kpi label={t('counters.trades', { defaultValue: '总交易' })} value={counters.trades_total} />
      <Kpi
        label={t('counters.orders_placed', { defaultValue: '下单数' })}
        value={counters.orders_placed}
      />
      <Kpi
        label={t('counters.orders_failed', { defaultValue: '失败订单' })}
        value={counters.orders_failed}
        sub={`失败率 ${failRate}%`}
      />
      <Kpi
        label={t('counters.risk_rejections', { defaultValue: '风控拒绝' })}
        value={counters.risk_rejections}
      />
      <Kpi
        label={t('counters.debate_skipped', { defaultValue: '辩论跳过' })}
        value={counters.debate_skipped_total}
      />
    </div>
  );
};

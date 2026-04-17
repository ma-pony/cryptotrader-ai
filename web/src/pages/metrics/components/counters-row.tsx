import { useTranslation } from 'react-i18next';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import type { MetricsCounters } from '@/types/api';

interface Props {
  counters: MetricsCounters;
}

export const CountersRow = ({ counters }: Props) => {
  const { t } = useTranslation('metrics');

  const items = [
    { label: t('counters.trades'), value: counters.trades_total },
    { label: t('counters.orders_placed', { defaultValue: '下单数' }), value: counters.orders_placed },
    { label: t('counters.orders_failed', { defaultValue: '失败订单' }), value: counters.orders_failed },
    { label: t('counters.risk_rejections', { defaultValue: '风控拒绝' }), value: counters.risk_rejections },
    { label: t('counters.debate_skipped', { defaultValue: '辩论跳过' }), value: counters.debate_skipped_total },
  ];

  return (
    <div className="grid grid-cols-5 gap-3">
      {items.map((item) => (
        <Card key={item.label}>
          <CardHeader className="p-3 pb-1">
            <CardTitle className="text-xs text-muted-foreground">{item.label}</CardTitle>
          </CardHeader>
          <CardContent className="p-3 pt-0">
            <span className="text-lg font-semibold tabular-nums">{item.value}</span>
          </CardContent>
        </Card>
      ))}
    </div>
  );
};

import { useTranslation } from 'react-i18next';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import type { RiskThresholds } from '@/types/api';

interface Props {
  thresholds: RiskThresholds;
}

export const ThresholdsCard = ({ thresholds }: Props) => {
  const { t } = useTranslation('risk');

  const items = [
    { label: t('thresholds.max_position'), value: `${(thresholds.max_position_pct * 100).toFixed(0)}%` },
    { label: t('thresholds.daily_loss_limit'), value: `${(thresholds.max_daily_loss_pct * 100).toFixed(1)}%` },
    { label: t('thresholds.per_trade_risk'), value: `${(thresholds.max_stop_loss_pct * 100).toFixed(1)}%` },
    { label: t('thresholds.max_trades_hour', { defaultValue: '每小时最大交易' }), value: String(thresholds.max_trades_per_hour) },
    { label: t('thresholds.max_trades_day', { defaultValue: '每日最大交易' }), value: String(thresholds.max_trades_per_day) },
    { label: t('thresholds.cooldown', { defaultValue: '亏损冷却' }), value: `${Math.round(thresholds.post_loss_cooldown_seconds / 60)} min` },
  ];

  return (
    <Card>
      <CardHeader className="p-4 pb-2">
        <CardTitle className="text-sm">{t('thresholds.title')}</CardTitle>
      </CardHeader>
      <CardContent className="p-4 pt-0">
        <div className="grid grid-cols-2 gap-2 text-xs">
          {items.map((item) => (
            <div key={item.label} className="flex justify-between gap-2">
              <span className="text-muted-foreground">{item.label}</span>
              <span className="font-medium tabular-nums">{item.value}</span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};

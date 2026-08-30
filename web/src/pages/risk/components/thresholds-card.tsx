import { Link } from 'react-router';
import { useTranslation } from 'react-i18next';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import type { RiskThresholds } from '@/types/api';

interface Props {
  thresholds: RiskThresholds;
}

export const ThresholdsCard = ({ thresholds }: Props) => {
  const { t } = useTranslation('risk');

  const items = [
    { label: t('thresholds.concentration'), value: thresholds.max_single_pct },
    { label: t('thresholds.exposure'), value: thresholds.max_total_exposure_pct },
    { label: t('thresholds.margin'), value: thresholds.max_margin_used_pct },
    { label: t('thresholds.drawdown'), value: thresholds.max_drawdown_pct },
  ];

  return (
    <Card>
      <CardHeader className="p-4 pb-2">
        <CardTitle className="text-sm">{t('thresholds.title')}</CardTitle>
        <Link to="/settings/risk" className="text-sm text-primary underline underline-offset-4">
          {t('settings_link')}
        </Link>
      </CardHeader>
      <CardContent className="p-4 pt-0">
        <div className="grid grid-cols-2 gap-2 text-xs">
          {items.map((item) => (
            <div key={item.label} className="flex justify-between gap-2">
              <span className="text-muted-foreground">{item.label}</span>
              <span className="font-medium tabular-nums">{Number((item.value * 100).toFixed(2))}%</span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};

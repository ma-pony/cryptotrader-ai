import { useState } from 'react';
import { useTranslation } from 'react-i18next';

import { EquityChart } from '@/components/charts/equity-chart';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { useEquityCurve } from '@/hooks/use-equity-curve';
import type { RangeWindow } from '@/types/api';

const RANGES: RangeWindow[] = ['24h', '7d', '30d', 'all'];

export const EquityChartSection = () => {
  const { t } = useTranslation('dashboard');
  const [range, setRange] = useState<RangeWindow>('24h');
  const { data, isLoading } = useEquityCurve(range);

  return (
    <Card>
      <CardHeader className="pb-2 flex flex-row items-center justify-between">
        <CardTitle className="text-base">{t('equity.title', { defaultValue: '权益曲线' })}</CardTitle>
        <div className="flex gap-1" role="tablist" aria-label={t('equity.range_label', { defaultValue: '时间范围' })}>
          {RANGES.map((r) => (
            <button
              key={r}
              role="tab"
              aria-selected={r === range}
              className={`px-3 py-1 text-xs rounded-md transition-colors ${
                r === range
                  ? 'bg-primary text-primary-foreground'
                  : 'text-muted-foreground hover:text-foreground hover:bg-muted'
              }`}
              onClick={() => setRange(r)}
            >
              {r}
            </button>
          ))}
        </div>
      </CardHeader>
      <CardContent className="pt-0">
        {isLoading ? (
          <Skeleton className="h-[320px] w-full" />
        ) : (
          <EquityChart data={data?.points ?? []} height={320} />
        )}
      </CardContent>
    </Card>
  );
};

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
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <div>
          <CardTitle className="text-sm">
            {t('equity.title', { defaultValue: '权益曲线' })}
          </CardTitle>
          <div className="mt-0.5 text-[11px] text-muted-foreground">
            {t('equity.subtitle', { defaultValue: '实时权益 · 多窗口对比' })}
          </div>
        </div>
        <div
          className="flex gap-0.5 rounded-md bg-muted/60 p-0.5"
          role="tablist"
          aria-label={t('equity.range_label', { defaultValue: '时间范围' })}
        >
          {RANGES.map((r) => (
            <button
              key={r}
              id={`equity-range-tab-${r}`}
              role="tab"
              aria-selected={r === range}
              aria-controls="equity-range-panel"
              tabIndex={r === range ? 0 : -1}
              className={`rounded px-2.5 py-1 text-[11px] font-medium uppercase tracking-wider transition-colors ${
                r === range
                  ? 'bg-card text-foreground shadow-sm'
                  : 'text-muted-foreground hover:text-foreground'
              }`}
              onClick={() => setRange(r)}
            >
              {r}
            </button>
          ))}
        </div>
      </CardHeader>
      <CardContent
        className="pt-0"
        role="tabpanel"
        id="equity-range-panel"
        aria-labelledby={`equity-range-tab-${range}`}
      >
        {isLoading ? (
          <Skeleton className="h-[320px] w-full" />
        ) : (
          <EquityChart data={data?.points ?? []} height={320} />
        )}
      </CardContent>
    </Card>
  );
};

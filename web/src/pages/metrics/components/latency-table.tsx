import { useTranslation } from 'react-i18next';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import type { MetricsPercentiles } from '@/types/api';

interface Props {
  percentiles: MetricsPercentiles;
}

export const LatencyTable = ({ percentiles }: Props) => {
  const { t } = useTranslation('metrics');

  const rows = [
    { label: t('latency.decision'), p50: percentiles.pipeline_p50_ms, p95: percentiles.pipeline_p95_ms },
    { label: t('latency.execution'), p50: percentiles.execution_p50_ms, p95: percentiles.execution_p95_ms },
  ];

  return (
    <Card>
      <CardHeader className="p-4 pb-2">
        <CardTitle className="text-sm">{t('latency.title', { defaultValue: '延迟' })}</CardTitle>
      </CardHeader>
      <CardContent className="p-4 pt-0">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-border text-muted-foreground">
              <th className="py-1 text-left font-medium" />
              <th className="py-1 text-right font-medium">{t('latency.p50')}</th>
              <th className="py-1 text-right font-medium">{t('latency.p95')}</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.label} className="border-b border-border/50">
                <td className="py-1.5">{r.label}</td>
                <td className="py-1.5 text-right tabular-nums">{r.p50.toFixed(0)} {t('latency.ms')}</td>
                <td className="py-1.5 text-right tabular-nums">{r.p95.toFixed(0)} {t('latency.ms')}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </CardContent>
    </Card>
  );
};

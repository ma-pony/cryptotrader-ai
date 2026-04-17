import { Suspense, useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { TrendChart, type TrendSeries } from '@/components/charts/trend-chart';
import { ErrorBoundary } from '@/components/error-boundary';
import { ErrorState } from '@/components/ui/error-state';
import { Skeleton } from '@/components/ui/skeleton';
import { useMetricsSummary } from '@/hooks/use-metrics-summary';
import { appendSample, loadHistory } from '@/lib/metrics-history';

import { CountersRow } from './components/counters-row';
import { LatencyTable } from './components/latency-table';

const MetricsContent = () => {
  const { t } = useTranslation('metrics');
  const { data, isLoading, isError, refetch } = useMetricsSummary();
  const [trendSeries, setTrendSeries] = useState<TrendSeries[]>([]);

  const refreshHistory = useCallback(async () => {
    const history = await loadHistory();
    setTrendSeries([
      { id: 'pipeline_p50', name: 'Pipeline P50', color: '#22c55e', points: history.map((h) => ({ ts: h.ts, value: h.pipeline_p50_ms })) },
      { id: 'pipeline_p95', name: 'Pipeline P95', color: '#eab308', points: history.map((h) => ({ ts: h.ts, value: h.pipeline_p95_ms })) },
      { id: 'exec_p50', name: 'Execution P50', color: '#3b82f6', points: history.map((h) => ({ ts: h.ts, value: h.execution_p50_ms })) },
      { id: 'exec_p95', name: 'Execution P95', color: '#ef4444', points: history.map((h) => ({ ts: h.ts, value: h.execution_p95_ms })) },
    ]);
  }, []);

  useEffect(() => {
    void refreshHistory();
  }, [refreshHistory]);

  useEffect(() => {
    if (!data) return;
    void appendSample(data).then(refreshHistory);
  }, [data, refreshHistory]);

  if (isLoading) return <Skeleton className="h-64 w-full" />;
  if (isError || !data) return <ErrorState title={t('title')} onRetry={() => void refetch()} />;

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold text-foreground">{t('title')}</h1>
      <CountersRow counters={data.counters} />
      <LatencyTable percentiles={data.percentiles} />
      <section aria-label={t('trend.title')}>
        <h3 className="text-sm font-medium mb-2">{t('trend.title')}</h3>
        {trendSeries.length > 0 && trendSeries[0]!.points.length > 1 ? (
          <TrendChart series={trendSeries} height={280} />
        ) : (
          <p className="text-sm text-muted-foreground py-8 text-center">{t('trend.empty')}</p>
        )}
      </section>
    </div>
  );
};

const MetricsPage = () => (
  <ErrorBoundary>
    <Suspense fallback={<Skeleton className="h-96 w-full" />}>
      <MetricsContent />
    </Suspense>
  </ErrorBoundary>
);

export default MetricsPage;

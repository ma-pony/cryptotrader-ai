import { useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { TrendChart, type TrendSeries } from '@/components/charts/trend-chart';
import { Card, CardContent } from '@/components/ui/card';
import { EmptyState } from '@/components/ui/empty-state';
import { PageBoundary } from '@/components/ui/page-boundary';
import { PageHeader } from '@/components/ui/page-header';
import { useMetricsSummary } from '@/hooks/use-metrics-summary';
import { appendSample, loadHistory } from '@/lib/metrics-history';

import { CountersRow } from './components/counters-row';
import { LatencyTable } from './components/latency-table';

const Kpi = ({ label, value, sub }: { label: string; value: string; sub?: string }) => (
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

const MetricsContent = () => {
  const { t } = useTranslation('metrics');
  const { data } = useMetricsSummary();
  const [trendSeries, setTrendSeries] = useState<TrendSeries[]>([]);

  const refreshHistory = useCallback(async () => {
    const history = await loadHistory();
    setTrendSeries([
      {
        id: 'pipeline_p50',
        name: 'Pipeline P50',
        color: '#d9a74f',
        points: history.map((h) => ({ ts: h.ts, value: h.pipeline_p50_ms })),
      },
      {
        id: 'pipeline_p95',
        name: 'Pipeline P95',
        color: '#e68650',
        points: history.map((h) => ({ ts: h.ts, value: h.pipeline_p95_ms })),
      },
      {
        id: 'exec_p50',
        name: 'Execution P50',
        color: '#5ea9cb',
        points: history.map((h) => ({ ts: h.ts, value: h.execution_p50_ms })),
      },
      {
        id: 'exec_p95',
        name: 'Execution P95',
        color: '#c45b5b',
        points: history.map((h) => ({ ts: h.ts, value: h.execution_p95_ms })),
      },
    ]);
  }, []);

  useEffect(() => {
    void refreshHistory();
  }, [refreshHistory]);

  useEffect(() => {
    if (!data) return;
    void appendSample(data).then(refreshHistory);
  }, [data, refreshHistory]);

  // MetricsPage's PageBoundary handles loading + error; null data here is a
  // transient gap (background refetch).
  if (!data) return null;

  // Server returns a raw Prometheus cumulative histogram (upper-bound ascending).
  // For the bar chart we want per-bucket (non-cumulative) counts.
  const rawBuckets = data.latency_histogram;
  const maxHist = Math.max(1, ...rawBuckets.map((b) => b.count));

  const costMax = Math.max(0.001, ...data.cost_14d.map((p) => p.cost_usd));

  return (
    <div className="space-y-6">
      <PageHeader title={t('title')} />

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Kpi
          label={t('summary.calls_24h', { defaultValue: 'LLM 调用 (24h)' })}
          value={data.llm_calls_24h.toLocaleString()}
        />
        <Kpi
          label={t('summary.cost_24h', { defaultValue: '成本 (24h)' })}
          value={`$${data.llm_cost_24h.toFixed(2)}`}
          sub={
            data.llm_calls_24h > 0
              ? `均 $${((data.llm_cost_24h / data.llm_calls_24h) * 1000).toFixed(3)}/千调用`
              : '—'
          }
        />
        <Kpi
          label={t('summary.cache_hit', { defaultValue: '缓存命中率' })}
          value={`${(data.cache_hit_rate * 100).toFixed(1)}%`}
          sub={`决策/天 ${data.decisions_per_day.toFixed(1)}`}
        />
        <Kpi
          label={t('summary.p95', { defaultValue: 'Pipeline P95' })}
          value={`${(data.percentiles.pipeline_p95_ms / 1000).toFixed(2)}s`}
          sub={`P50 ${(data.percentiles.pipeline_p50_ms / 1000).toFixed(2)}s`}
        />
      </div>

      <CountersRow counters={data.counters} />
      <LatencyTable percentiles={data.percentiles} />

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Card>
          <CardContent className="p-4">
            <div className="mb-2 text-sm font-medium">
              {t('histogram.title', { defaultValue: '延迟分布直方图' })}
            </div>
            {rawBuckets.length === 0 ? (
              <EmptyState
                size="compact"
                className="h-32"
                title={t('histogram.empty', { defaultValue: '尚无观测样本' })}
              />
            ) : (
              <div className="flex h-40 items-end gap-1">
                {rawBuckets.map((b, i) => {
                  const h = (b.count / maxHist) * 100;
                  const label =
                    b.upper_bound_s >= 1e11 ? '+∞' : `<${b.upper_bound_s.toFixed(2)}s`;
                  const bg =
                    i < rawBuckets.length * 0.5
                      ? 'linear-gradient(180deg, var(--cyan-500), var(--cyan-600))'
                      : i < rawBuckets.length * 0.75
                        ? 'linear-gradient(180deg, var(--amber-500), var(--amber-600))'
                        : 'linear-gradient(180deg, var(--trade-short), color-mix(in oklch, var(--trade-short) 50%, black))';
                  return (
                    <div key={i} className="flex flex-1 flex-col items-center gap-1.5">
                      <div className="font-mono text-[9px] text-muted-foreground">{b.count}</div>
                      <div
                        className="w-full rounded-t"
                        style={{ height: `${Math.max(2, h)}%`, background: bg, minHeight: 2 }}
                      />
                      <div className="whitespace-nowrap text-[9px] text-muted-foreground">
                        {label}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="mb-2 text-sm font-medium">
              {t('cost_14d.title', { defaultValue: 'LLM 成本 · 最近 14 天' })}
            </div>
            {data.cost_14d.length === 0 ? (
              <EmptyState
                size="compact"
                className="h-32"
                title={t('cost_14d.empty', { defaultValue: '尚无成本数据' })}
              />
            ) : (
              <div className="flex h-32 items-end gap-1">
                {data.cost_14d.map((p, i) => {
                  const h = (p.cost_usd / costMax) * 100;
                  return (
                    <div key={p.ts} className="flex flex-1 flex-col items-center gap-1">
                      <div
                        className="w-full rounded-t bg-amber-500"
                        style={{
                          height: `${Math.max(2, h)}%`,
                          minHeight: 2,
                          opacity: 0.35 + (i / 14) * 0.65,
                        }}
                      />
                      <div className="font-mono text-[9px] text-muted-foreground">
                        ${p.cost_usd.toFixed(2)}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </CardContent>
        </Card>

        {/* spec 020a FR-Z19: Cache Hit Rate panel */}
        <Card>
          <CardContent className="p-4">
            <div className="mb-1 text-sm font-medium">
              {t('cache.title', { defaultValue: 'Prompt Cache 命中率 (24h)' })}
            </div>
            <div className="font-mono text-2xl font-semibold tabular-nums">
              {(data.cache_hit_rate * 100).toFixed(1)}%
            </div>
            <div className="mt-1 text-[10px] text-muted-foreground">
              {t('cache.hint', { defaultValue: '24h 滑动窗口均值 · Anthropic ephemeral cache' })}
            </div>
            <div className="mt-3 h-2 w-full overflow-hidden rounded-full bg-muted">
              <div
                className="h-full rounded-full bg-emerald-500 transition-all"
                style={{ width: `${Math.min(100, data.cache_hit_rate * 100).toFixed(1)}%` }}
              />
            </div>
          </CardContent>
        </Card>

        {/* spec 020a FR-Z19: IVE Failure Rate panel */}
        <Card>
          <CardContent className="p-4">
            <div className="mb-1 text-sm font-medium">
              {t('ive.title', { defaultValue: 'IVE 分类失败率 (1h)' })}
            </div>
            <div className="font-mono text-2xl font-semibold tabular-nums">
              {((data as { ive_failure_rate?: number }).ive_failure_rate != null
                ? ((data as { ive_failure_rate: number }).ive_failure_rate * 100).toFixed(1)
                : '0.0')}%
            </div>
            <div className="mt-1 text-[10px] text-muted-foreground">
              {t('ive.hint', { defaultValue: '1h 滑动窗口 · IVE classify_case 失败比例' })}
            </div>
            <div className="mt-3 h-2 w-full overflow-hidden rounded-full bg-muted">
              <div
                className="h-full rounded-full bg-rose-500 transition-all"
                style={{
                  width: `${Math.min(
                    100,
                    (data as { ive_failure_rate?: number }).ive_failure_rate != null
                      ? ((data as { ive_failure_rate: number }).ive_failure_rate * 100)
                      : 0,
                  ).toFixed(1)}%`,
                }}
              />
            </div>
          </CardContent>
        </Card>
      </div>

      <section aria-label={t('trend.title')}>
        <h3 className="text-sm font-medium mb-2 text-muted-foreground uppercase tracking-wider text-[10px]">
          {t('trend.title', { defaultValue: '历史趋势 · 本地会话' })}
        </h3>
        {trendSeries.length > 0 && trendSeries[0]!.points.length > 1 ? (
          <TrendChart series={trendSeries} height={260} />
        ) : (
          <EmptyState
            size="compact"
            title={t('trend.empty', { defaultValue: '尚无足够样本' })}
            description={t('trend.empty_hint', {
              defaultValue: '运行几个调度周期后，曲线会自动出现在这里',
            })}
          />
        )}
      </section>
    </div>
  );
};

const MetricsPage = () => {
  const { t } = useTranslation('metrics');
  const { isLoading, isError, refetch } = useMetricsSummary();
  return (
    <PageBoundary
      loading={isLoading}
      isError={isError}
      onRetry={() => void refetch()}
      errorTitle={t('title')}
    >
      <MetricsContent />
    </PageBoundary>
  );
};

export default MetricsPage;

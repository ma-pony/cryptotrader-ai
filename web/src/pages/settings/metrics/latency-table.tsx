import { useTranslation } from 'react-i18next';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import type { MetricsPercentiles } from '@/types/api';

interface Props {
  percentiles: MetricsPercentiles;
}

const LatencyBar = ({
  label,
  p50,
  p95,
  max,
}: {
  label: string;
  p50: number;
  p95: number;
  max: number;
}) => {
  const p50Pct = Math.min(100, (p50 / max) * 100);
  const p95Pct = Math.min(100, (p95 / max) * 100);
  const tone = p95 > 10_000 ? 'short' : p95 > 5_000 ? 'amber' : 'cyan';
  const toneVar =
    tone === 'short' ? 'var(--trade-short)' : tone === 'amber' ? 'var(--amber-500)' : 'var(--cyan-500)';

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <span className="text-[10px] uppercase tracking-wider font-medium text-muted-foreground">
          {label}
        </span>
        <span className="font-mono text-xs text-muted-foreground">
          P95 <span style={{ color: toneVar }}>{(p95 / 1000).toFixed(2)}s</span>
        </span>
      </div>
      <div className="relative h-5 overflow-hidden rounded bg-muted">
        <div
          className="absolute left-0 top-0 bottom-0 opacity-45"
          style={{ width: `${p95Pct}%`, background: toneVar }}
        />
        <div
          className="absolute left-0 top-0 bottom-0"
          style={{ width: `${p50Pct}%`, background: toneVar }}
        />
        <div className="absolute inset-0 flex items-center justify-between px-2 text-[10px] font-mono font-medium">
          <span className="text-foreground">P50 {(p50 / 1000).toFixed(2)}s</span>
          <span className="text-foreground/80">P95 {(p95 / 1000).toFixed(2)}s</span>
        </div>
      </div>
    </div>
  );
};

export const LatencyTable = ({ percentiles }: Props) => {
  const { t } = useTranslation('metrics');
  const max = Math.max(
    percentiles.pipeline_p95_ms,
    percentiles.execution_p95_ms,
    1000,
  );

  return (
    <Card>
      <CardHeader className="p-4 pb-2">
        <CardTitle className="text-sm">
          {t('latency.title', { defaultValue: '延迟分布' })}
        </CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-4 p-4 pt-1">
        <LatencyBar
          label={t('latency.decision', { defaultValue: '决策流水线' })}
          p50={percentiles.pipeline_p50_ms}
          p95={percentiles.pipeline_p95_ms}
          max={max}
        />
        <LatencyBar
          label={t('latency.execution', { defaultValue: '订单执行' })}
          p50={percentiles.execution_p50_ms}
          p95={percentiles.execution_p95_ms}
          max={max}
        />
      </CardContent>
    </Card>
  );
};

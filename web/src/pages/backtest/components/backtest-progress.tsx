import { Loader2, Pause } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { StatusPill } from '@/components/ui/status-pill';
import { useCancelBacktest } from '@/hooks/use-backtest';
import type { BacktestRunStatus } from '@/types/api';

interface Props {
  run: BacktestRunStatus;
}

const toneFor = (s: string): 'amber' | 'success' | 'danger' | 'default' => {
  if (s === 'completed') return 'success';
  if (s === 'failed') return 'danger';
  if (s === 'running') return 'amber';
  if (s === 'queued') return 'amber';
  return 'default';
};

export const BacktestProgress = ({ run }: Props) => {
  const { t } = useTranslation('backtest');
  const cancel = useCancelBacktest();
  const isActive = run.status === 'queued' || run.status === 'running';
  const pct = Math.round(run.progress * 100);
  const pair = run.params.pair;
  const startedAt = run.started_at ? new Date(run.started_at).getTime() : null;
  const elapsedSec =
    startedAt != null && isActive ? Math.max(0, Math.floor((Date.now() - startedAt) / 1000)) : null;
  const etaSec =
    elapsedSec != null && run.progress > 0 && run.progress < 1
      ? Math.round(elapsedSec * (1 - run.progress) / run.progress)
      : null;

  return (
    <Card
      className="overflow-hidden"
      style={{
        background: isActive
          ? 'linear-gradient(135deg, color-mix(in oklch, var(--amber-500) 5%, transparent), hsl(var(--card)) 60%)'
          : 'hsl(var(--card))',
        borderColor: isActive
          ? 'color-mix(in oklch, var(--amber-500) 25%, transparent)'
          : 'hsl(var(--border))',
      }}
    >
      <CardContent className="flex flex-col gap-3.5 p-5">
        <div className="flex items-center gap-3">
          {isActive ? (
            <span className="h-2.5 w-2.5 animate-ct-pulse rounded-full bg-amber-500 shadow-[0_0_10px_var(--amber-glow)]" />
          ) : (
            <Loader2 className="h-4 w-4 text-muted-foreground" />
          )}
          <div className="flex-1 min-w-0">
            <div className="text-sm font-semibold">
              {pair} · {run.params.mode === 'llm' ? 'LLM 驱动' : 'SMA 对照组'}
            </div>
            <div className="font-mono text-[11px] text-muted-foreground truncate">
              {run.run_id}
            </div>
          </div>
          <StatusPill tone={toneFor(run.status)} live={isActive}>
            {t(`status.${run.status.toUpperCase()}`, { defaultValue: run.status })}
          </StatusPill>
          {isActive ? (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => cancel.mutate(run.run_id)}
              disabled={cancel.isPending}
              aria-label={t('form.cancel', { defaultValue: '取消' })}
            >
              <Pause className="h-3.5 w-3.5" />
            </Button>
          ) : null}
        </div>

        <div className="flex flex-col gap-1.5">
          <div className="flex items-center justify-between text-[11px] text-muted-foreground">
            <span>
              {t('progress.label', { defaultValue: '进度' })} · {pair}
            </span>
            <span className="font-mono">{pct}%</span>
          </div>
          <div className="relative h-2 overflow-hidden rounded bg-muted">
            <div
              className="h-full rounded bg-amber-500 transition-[width] duration-300"
              style={{ width: `${pct}%` }}
            />
          </div>
        </div>

        {isActive ? (
          <div className="grid grid-cols-3 gap-4">
            <KV label={t('progress.elapsed', { defaultValue: '已运行' })} value={fmtDuration(elapsedSec)} />
            <KV label={t('progress.eta', { defaultValue: 'ETA' })} value={fmtDuration(etaSec)} />
            <KV
              label={t('progress.capital', { defaultValue: '初始资金' })}
              value={`$${run.params.initial_capital.toLocaleString()}`}
            />
          </div>
        ) : null}

        {run.error ? (
          <div className="rounded-md border border-trade-short/40 bg-trade-short-soft p-2.5 text-xs text-trade-short">
            {run.error}
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
};

const KV = ({ label, value }: { label: string; value: string }) => (
  <div className="flex flex-col gap-1">
    <div className="text-[10px] uppercase tracking-wider font-medium text-muted-foreground">
      {label}
    </div>
    <div className="font-mono text-sm font-semibold tabular-nums">{value}</div>
  </div>
);

const fmtDuration = (sec: number | null): string => {
  if (sec == null) return '—';
  if (sec < 60) return `${sec}s`;
  if (sec < 3600) return `${Math.floor(sec / 60)}m ${sec % 60}s`;
  return `${Math.floor(sec / 3600)}h ${Math.floor((sec % 3600) / 60)}m`;
};

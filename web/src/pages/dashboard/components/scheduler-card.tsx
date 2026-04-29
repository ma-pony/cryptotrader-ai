import { useTranslation } from 'react-i18next';

import { Card, CardContent } from '@/components/ui/card';
import { ErrorState } from '@/components/ui/error-state';
import { Skeleton } from '@/components/ui/skeleton';
import { StatusPill } from '@/components/ui/status-pill';
import { useCountdown } from '@/hooks/use-countdown';
import type { SchedulerStatus } from '@/types/api';

interface SchedulerCardProps {
  data: SchedulerStatus | undefined;
  isLoading: boolean;
  isError: boolean;
}

export const SchedulerCard = ({ data, isLoading, isError }: SchedulerCardProps) => {
  const { t } = useTranslation('dashboard');
  const { formatted: countdown } = useCountdown(data?.next_run_at, 1_000);

  if (isLoading) {
    return (
      <Card>
        <CardContent className="p-4 space-y-2">
          <Skeleton className="h-4 w-24" />
          <Skeleton className="h-10 w-32" />
          <Skeleton className="h-3 w-40" />
        </CardContent>
      </Card>
    );
  }

  if (isError || !data) {
    return (
      <Card>
        <CardContent className="p-4">
          <ErrorState title={t('scheduler.error', { defaultValue: '调度器状态不可用' })} />
        </CardContent>
      </Card>
    );
  }

  const redisDown = !data.redis_available;

  return (
    <Card
      className="overflow-hidden"
      style={{
        background: 'linear-gradient(135deg, color-mix(in oklch, var(--amber-500) 8%, transparent), hsl(var(--card)) 60%)',
        borderColor: 'color-mix(in oklch, var(--amber-500) 35%, transparent)',
      }}
    >
      <CardContent className="flex flex-col gap-2.5 p-4">
        <div className="flex items-center justify-between">
          <div className="inline-flex items-center gap-2">
            <span className="h-2 w-2 rounded-full bg-amber-500 animate-ct-pulse shadow-[0_0_8px_var(--amber-glow)]" />
            <span className="text-[10px] uppercase tracking-wider font-medium text-muted-foreground">
              {t('scheduler.next_run', { defaultValue: '下次触发' })}
            </span>
          </div>
          <StatusPill tone={data.enabled ? 'amber' : 'default'}>
            {data.enabled
              ? t('scheduler.enabled', { defaultValue: '已启用' })
              : t('scheduler.disabled', { defaultValue: '未启用' })}
          </StatusPill>
        </div>

        <div className="font-mono text-[28px] font-semibold tracking-tight text-amber-500 leading-none">
          {countdown}
        </div>

        <div className="flex items-center justify-between text-[11px] text-muted-foreground">
          <span>
            {data.next_pair
              ? `${data.next_pair} · ${t('scheduler.next_pair', { defaultValue: '下一币对' })}`
              : t('scheduler.idle', { defaultValue: '空闲' })}
          </span>
          <span className="font-mono">{redisDown ? 'Redis down' : 'Redis ok'}</span>
        </div>

        {redisDown ? (
          <div className="mt-1 flex items-start gap-2 rounded-md border border-trade-short/35 bg-trade-short-soft px-2 py-1 text-[11px] text-trade-short">
            <span className="font-medium">⚠</span>
            <span>{t('states.redis_down', { defaultValue: 'Redis 不可用（保守模式）' })}</span>
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
};

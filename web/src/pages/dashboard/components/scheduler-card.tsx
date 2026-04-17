import { useTranslation } from 'react-i18next';

import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { ErrorState } from '@/components/ui/error-state';
import { Skeleton } from '@/components/ui/skeleton';
import { formatRelative } from '@/lib/format';
import type { SchedulerStatus } from '@/types/api';

interface SchedulerCardProps {
  data: SchedulerStatus | undefined;
  isLoading: boolean;
  isError: boolean;
}

export const SchedulerCard = ({ data, isLoading, isError }: SchedulerCardProps) => {
  const { t } = useTranslation('dashboard');

  if (isLoading) {
    return (
      <Card>
        <CardContent className="p-4 space-y-2">
          <Skeleton className="h-4 w-24" />
          <Skeleton className="h-6 w-32" />
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
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center gap-2">
          {t('scheduler.title', { defaultValue: '调度器' })}
          <Badge variant={data.enabled ? 'default' : 'secondary'}>
            {data.enabled ? t('scheduler.enabled', { defaultValue: '已启用' }) : t('scheduler.disabled', { defaultValue: '未启用' })}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-1 text-sm">
        {redisDown && (
          <p className="text-warning font-medium" role="alert">
            {t('states.redis_down')}
          </p>
        )}
        {data.next_pair && (
          <p>
            <span className="text-muted-foreground">{t('scheduler.next_pair', { defaultValue: '下一币对' })}:</span>{' '}
            <span className="font-medium">{data.next_pair}</span>
          </p>
        )}
        {data.next_run_at && (
          <p>
            <span className="text-muted-foreground">{t('scheduler.next_run', { defaultValue: '下次触发' })}:</span>{' '}
            <span className="font-medium">{formatRelative(data.next_run_at)}</span>
          </p>
        )}
        {!data.enabled && !data.next_pair && (
          <p className="text-muted-foreground">{t('scheduler.idle', { defaultValue: '调度器当前处于空闲状态' })}</p>
        )}
      </CardContent>
    </Card>
  );
};

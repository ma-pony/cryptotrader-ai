import { AlertTriangle } from 'lucide-react';
import { Suspense } from 'react';
import { useTranslation } from 'react-i18next';

import { ErrorBoundary } from '@/components/error-boundary';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { ErrorState } from '@/components/ui/error-state';
import { Skeleton } from '@/components/ui/skeleton';
import { useRiskStatus } from '@/hooks/use-risk-status';

import { CircuitBreakerCard } from './components/circuit-breaker-card';
import { ThresholdsCard } from './components/thresholds-card';

const RiskContent = () => {
  const { t } = useTranslation('risk');
  const { data, isLoading, isError, refetch } = useRiskStatus();

  if (isLoading) return <Skeleton className="h-64 w-full" />;
  if (isError || !data) return <ErrorState title={t('title')} onRetry={() => void refetch()} />;

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold text-foreground">{t('title')}</h1>

      {!data.redis_available && (
        <div className="flex items-center gap-2 rounded-md border border-warning/40 bg-warning/10 p-3 text-xs text-warning">
          <AlertTriangle className="h-4 w-4 shrink-0" aria-hidden />
          {t('redis_warning')}
        </div>
      )}

      <div className="grid grid-cols-2 gap-4">
        <Card>
          <CardHeader className="p-4 pb-2">
            <CardTitle className="text-sm">{t('trade_count.today')}</CardTitle>
          </CardHeader>
          <CardContent className="p-4 pt-0">
            <span className="text-2xl font-semibold tabular-nums">{data.trade_count_day ?? '—'}</span>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="p-4 pb-2">
            <CardTitle className="text-sm">{t('trade_count.window')}</CardTitle>
          </CardHeader>
          <CardContent className="p-4 pt-0">
            <span className="text-2xl font-semibold tabular-nums">{data.trade_count_hour ?? '—'}</span>
          </CardContent>
        </Card>
      </div>

      <CircuitBreakerCard cb={data.circuit_breaker} />
      <ThresholdsCard thresholds={data.thresholds} />
    </div>
  );
};

const RiskPage = () => (
  <ErrorBoundary>
    <Suspense fallback={<Skeleton className="h-96 w-full" />}>
      <RiskContent />
    </Suspense>
  </ErrorBoundary>
);

export default RiskPage;

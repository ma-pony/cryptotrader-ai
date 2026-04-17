import { Suspense, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { EquityChart } from '@/components/charts/equity-chart';
import { ErrorBoundary } from '@/components/error-boundary';
import { Skeleton } from '@/components/ui/skeleton';
import { useBacktestRun } from '@/hooks/use-backtest';

import { BacktestForm } from './components/backtest-form';
import { BacktestMetricsRow } from './components/backtest-metrics-row';
import { BacktestProgress } from './components/backtest-progress';

const BacktestContent = () => {
  const { t } = useTranslation('backtest');
  const [runId, setRunId] = useState<string>();
  const { data: run } = useBacktestRun(runId);

  const result = run?.result;

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold text-foreground">{t('title')}</h1>
      <BacktestForm onRunStarted={setRunId} />

      {run && <BacktestProgress run={run} />}

      {result && (
        <>
          <BacktestMetricsRow metrics={result.metrics} />
          <section aria-label={t('chart.equity')}>
            <h3 className="text-sm font-medium mb-2">{t('chart.equity')}</h3>
            <EquityChart data={result.equity_curve} height={360} />
          </section>
        </>
      )}
    </div>
  );
};

const BacktestPage = () => (
  <ErrorBoundary>
    <Suspense fallback={<Skeleton className="h-96 w-full" />}>
      <BacktestContent />
    </Suspense>
  </ErrorBoundary>
);

export default BacktestPage;

import { useState } from 'react';
import { useTranslation } from 'react-i18next';

import { EquityChart } from '@/components/charts/equity-chart';
import { PageBoundary } from '@/components/ui/page-boundary';
import { PageHeader } from '@/components/ui/page-header';
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
      <PageHeader title={t('title')} />
      <BacktestForm onRunStarted={setRunId} />

      {run && <BacktestProgress run={run} />}

      {result && (
        <>
          <BacktestMetricsRow metrics={result.metrics} />
          <section aria-label={t('chart.equity')}>
            <h3 className="mb-2 text-sm font-medium">{t('chart.equity')}</h3>
            <EquityChart data={result.equity_curve} height={360} />
          </section>
        </>
      )}
    </div>
  );
};

const BacktestPage = () => (
  <PageBoundary>
    <BacktestContent />
  </PageBoundary>
);

export default BacktestPage;

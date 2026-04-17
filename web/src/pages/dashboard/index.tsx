import { Suspense } from 'react';
import { useTranslation } from 'react-i18next';

import { ErrorBoundary } from '@/components/error-boundary';
import { Skeleton } from '@/components/ui/skeleton';
import { usePortfolioSnapshot } from '@/hooks/use-portfolio-snapshot';
import { useSchedulerStatus } from '@/hooks/use-scheduler-status';

import { EquityChartSection } from './components/equity-chart-section';
import { MetricCardsRow } from './components/metric-cards-row';
import { PositionsTable } from './components/positions-table';
import { SchedulerCard } from './components/scheduler-card';

const DashboardContent = () => {
  const { t } = useTranslation('dashboard');
  const portfolio = usePortfolioSnapshot();
  const scheduler = useSchedulerStatus();

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold text-foreground">{t('title', { defaultValue: '总览' })}</h1>

      <MetricCardsRow data={portfolio.data} isLoading={portfolio.isLoading} />

      <div className="grid grid-cols-3 gap-4">
        <div className="col-span-2">
          <EquityChartSection />
        </div>
        <div className="space-y-4">
          <SchedulerCard data={scheduler.data} isLoading={scheduler.isLoading} isError={scheduler.isError} />
        </div>
      </div>

      <PositionsTable positions={portfolio.data?.positions} isLoading={portfolio.isLoading} />
    </div>
  );
};

const DashboardPage = () => (
  <ErrorBoundary>
    <Suspense fallback={<Skeleton className="h-96 w-full" />}>
      <DashboardContent />
    </Suspense>
  </ErrorBoundary>
);

export default DashboardPage;

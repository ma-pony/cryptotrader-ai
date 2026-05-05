import { useTranslation } from 'react-i18next';

import { PageBoundary } from '@/components/ui/page-boundary';
import { PageHeader } from '@/components/ui/page-header';
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
      <PageHeader title={t('title', { defaultValue: '总览' })} />

      <MetricCardsRow data={portfolio.data} isLoading={portfolio.isLoading} connectionStatus={portfolio.connectionStatus} />

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <div className="lg:col-span-2">
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
  <PageBoundary>
    <DashboardContent />
  </PageBoundary>
);

export default DashboardPage;

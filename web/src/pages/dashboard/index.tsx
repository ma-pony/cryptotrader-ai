import { useTranslation } from 'react-i18next';

import { PageBoundary } from '@/components/ui/page-boundary';
import { PageHeader } from '@/components/ui/page-header';
import { usePortfolioBooks } from '@/hooks/use-portfolio-books';
import { useSchedulerStatus } from '@/hooks/use-scheduler-status';

import { SchedulerCard } from './components/scheduler-card';

const DashboardContent = () => {
  const { t } = useTranslation('dashboard');
  const portfolio = usePortfolioBooks();
  const scheduler = useSchedulerStatus();

  return (
    <div className="space-y-6">
      <PageHeader title={t('title', { defaultValue: '总览' })} />

      <div className="grid gap-4 lg:grid-cols-2">
        {(['simulated', 'real'] as const).map((scope) => (
          <section key={scope} className="rounded-lg border border-border bg-card p-4" aria-label={t(`scopes.${scope}`, { defaultValue: scope })}>
            <h2 className="text-sm font-semibold">{t(`scopes.${scope}`, { defaultValue: scope === 'simulated' ? 'Simulated capital' : 'Real capital' })}</h2>
            {portfolio.isLoading ? <p className="mt-3 text-sm text-muted-foreground">…</p> : portfolio.data?.[scope].books.length ? portfolio.data[scope].books.map((book) => <div key={book.book_id} className="mt-3 rounded border border-border p-3"><div className="font-mono text-xs">{book.book_id}</div><div className="mt-1 text-lg font-semibold">{book.total_equity} USDT</div>{book.connections.map((connection) => <div key={connection.connection_id} className="mt-1 text-xs text-muted-foreground">{connection.connection_id} · {connection.equity} · {connection.position.signed_notional}</div>)}</div>) : <p className="mt-3 text-sm text-muted-foreground">{t('scopes.empty', { defaultValue: 'No execution books in this scope.' })}</p>}
          </section>
        ))}
      </div>
      <SchedulerCard data={scheduler.data} isLoading={scheduler.isLoading} isError={scheduler.isError} />
    </div>
  );
};

const DashboardPage = () => (
  <PageBoundary>
    <DashboardContent />
  </PageBoundary>
);

export default DashboardPage;

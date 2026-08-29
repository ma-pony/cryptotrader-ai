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
      <PageHeader title={t('title')} />
      <div className="grid gap-4 lg:grid-cols-2">
        {(['simulated', 'real'] as const).map((scope) => (
          <section key={scope} className="rounded-lg border border-border bg-card p-4" aria-label={t(`books.${scope}`)}>
            <h2 className="text-sm font-semibold">{t(`books.${scope}`)}</h2>
            {portfolio.isLoading ? <p className="mt-3 text-sm text-muted-foreground">{t('books.loading')}</p> : null}
            {portfolio.isError ? <p className="mt-3 text-sm text-destructive">{t('books.error')}</p> : null}
            {!portfolio.isLoading && !portfolio.isError && portfolio.data?.[scope].books.length === 0 ? (
              <p className="mt-3 text-sm text-muted-foreground">{t('books.empty')}</p>
            ) : null}
            {portfolio.data ? (
              <div className="mt-3 rounded border border-primary/30 bg-primary/5 p-3 text-sm">
                <p className="font-medium">{t('books.total')}</p>
                <p>{t('books.equity')}: {portfolio.data[scope].totals.equity}</p>
                <p>{t('books.notional')}: {portfolio.data[scope].totals.signed_notional}</p>
              </div>
            ) : null}
            {portfolio.data?.[scope].books.map((book) => (
              <article key={book.book_id} className="mt-3 rounded border border-border p-3">
                <h3 className="font-mono text-xs">
                  {t('books.book')}: {book.book_id}
                </h3>
                <p className="mt-1 text-sm">
                  {t('books.equity')}: {book.total_equity}
                </p>
                <p className="text-sm">
                  {t('books.notional')}: {book.total_signed_notional}
                </p>
                {book.connections.map((connection) => (
                  <div
                    key={connection.connection_id}
                    className="mt-2 border-t border-border pt-2 text-xs text-muted-foreground"
                  >
                    <p>
                      {t('books.connection')}: {connection.connection_id} · {t('books.equity')}: {connection.equity}
                    </p>
                    <p>
                      {t('books.balances')}:{' '}
                      {connection.balances.map((balance) => `${balance.asset}: ${balance.amount}`).join(' · ')}
                    </p>
                    <p>
                      {t('books.position')}: {connection.position.pair} · {connection.position.signed_amount} ·{' '}
                      {connection.position.signed_notional} · {t('books.entry')}:{' '}
                      {connection.position.entry_price ?? '—'}
                    </p>
                  </div>
                ))}
              </article>
            ))}
          </section>
        ))}
      </div>
      <SchedulerCard data={scheduler.data} isLoading={scheduler.isLoading} isError={scheduler.isError} />
    </div>
  );
};

export default function DashboardPage() {
  return (
    <PageBoundary>
      <DashboardContent />
    </PageBoundary>
  );
}

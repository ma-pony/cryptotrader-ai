import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Link } from 'react-router';

import { PageBoundary } from '@/components/ui/page-boundary';
import { PageHeader } from '@/components/ui/page-header';
import { useMultiVenueCycles } from '@/hooks/use-multi-venue-cycles';
import { cycleStatusTone, formatCycleStatus } from '@/lib/cycle-status';

const PAGE_SIZE = 20;

const DecisionsContent = () => {
  const { t } = useTranslation(['decisions', 'cycles']);
  const [page, setPage] = useState(1);

  const cycles = useMultiVenueCycles(page, PAGE_SIZE);
  const items = cycles.data?.items ?? [];

  return (
    <div className="space-y-6">
      <PageHeader title={t('title')} />
      {cycles.isLoading ? (
        <p>{t('loading', { ns: 'cycles' })}</p>
      ) : null}
      {cycles.isError ? <p className="text-destructive">{t('loadError', { ns: 'cycles' })}</p> : null}
      {!cycles.isLoading && !cycles.isError && items.length === 0 ? <p>{t('empty', { ns: 'cycles' })}</p> : null}
      {items.length > 0 ? (
        <>
          {items.map((cycle) => (
          <Link
            key={cycle.cycle_id}
            to={`/cycles/${cycle.cycle_id}`}
            className="block rounded border border-border p-3"
          >
            <b>{cycle.cycle_id}</b>
            <span className="ml-2 text-xs text-muted-foreground">
              {cycle.books.map((book) => `${book.book_id} (${book.capital_scope})`).join(', ')}
            </span>
            <span
              className={`ml-2 text-xs ${cycleStatusTone(cycle.cycle_status) === 'danger' ? 'text-destructive' : cycleStatusTone(cycle.cycle_status) === 'warning' ? 'text-amber-600' : 'text-muted-foreground'}`}
            >
              {formatCycleStatus(t, cycle.cycle_status)}
            </span>
            {cycle.requires_attention ? (
              <span className="ml-2 text-amber-600">{t('attention', { ns: 'cycles' })}</span>
            ) : null}
          </Link>
          ))}
          <nav className="flex items-center justify-between" aria-label={t('title')}>
            <button type="button" onClick={() => setPage((current) => Math.max(1, current - 1))} disabled={page === 1}>
              {t('previous', { ns: 'cycles' })}
            </button>
            <span>{t('page', { ns: 'cycles', page })}</span>
            <button type="button" onClick={() => setPage((current) => current + 1)} disabled={!cycles.data?.has_next}>
              {t('next', { ns: 'cycles' })}
            </button>
          </nav>
        </>
      ) : null}
    </div>
  );
};

const DecisionsPage = () => (
  <PageBoundary>
    <DecisionsContent />
  </PageBoundary>
);

export default DecisionsPage;

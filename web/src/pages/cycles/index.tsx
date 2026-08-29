import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Link } from 'react-router';

import { PageBoundary } from '@/components/ui/page-boundary';
import { PageHeader } from '@/components/ui/page-header';
import { useMultiVenueCycles } from '@/hooks/use-multi-venue-cycles';
import { cycleStatusTone, formatCycleStatus } from '@/lib/cycle-status';

const toneClass = {
  neutral: 'text-muted-foreground',
  success: 'text-emerald-600',
  warning: 'text-amber-600',
  danger: 'text-destructive',
};

export default function CyclesPage() {
  const { t } = useTranslation('cycles');
  const [page, setPage] = useState(1);
  const cycles = useMultiVenueCycles(page);
  const items = cycles.data?.items ?? [];

  return (
    <PageBoundary>
      <div className="space-y-5">
        <PageHeader title={t('title')} />
        {cycles.isLoading ? <p>{t('loading')}</p> : null}
        {cycles.isError ? <p className="text-destructive">{t('loadError')}</p> : null}
        {!cycles.isLoading && !cycles.isError && items.length === 0 ? <p>{t('empty')}</p> : null}
        {items.length > 0 ? (
          <>
            <div className="space-y-2">
              {items.map((cycle) => (
                <Link
                  key={cycle.cycle_id}
                  to={`/cycles/${cycle.cycle_id}`}
                  className="block rounded-md border border-border bg-card p-3 hover:bg-muted"
                >
                  <div className="flex justify-between gap-3">
                    <strong>{cycle.cycle_id}</strong>
                    <span className={toneClass[cycleStatusTone(cycle.cycle_status)]}>
                      {formatCycleStatus(t, cycle.cycle_status)}
                    </span>
                  </div>
                  <p className="mt-1 text-sm text-muted-foreground">
                    R{cycle.config_revision} · {t('books', { count: cycle.books.length })}
                    {cycle.requires_attention ? ` · ${t('attention')}` : ''}
                  </p>
                </Link>
              ))}
            </div>
            <nav className="flex items-center justify-between" aria-label={t('title')}>
              <button
                type="button"
                onClick={() => setPage((current) => Math.max(1, current - 1))}
                disabled={page === 1}
              >
                {t('previous')}
              </button>
              <span>{page}</span>
              <button type="button" onClick={() => setPage((current) => current + 1)} disabled={!cycles.data?.has_next}>
                {t('next')}
              </button>
            </nav>
          </>
        ) : null}
      </div>
    </PageBoundary>
  );
}

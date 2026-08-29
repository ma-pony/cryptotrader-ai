import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Link } from 'react-router';

import { PageBoundary } from '@/components/ui/page-boundary';
import { PageHeader } from '@/components/ui/page-header';
import { useDecisions } from '@/hooks/use-decisions';
import { cycleStatusTone, formatCycleStatus } from '@/lib/cycle-status';

const PAGE_SIZE = 20;

const DecisionsContent = () => {
  const { t } = useTranslation(['decisions', 'cycles']);
  const [page] = useState(1);

  const { data, isLoading } = useDecisions({ page, size: PAGE_SIZE });

  return (
    <div className="space-y-6">
      <PageHeader title={t('title')} />
      {isLoading ? (
        <p>{t('loading', { ns: 'cycles' })}</p>
      ) : (
        data?.items.map((cycle) => (
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
        ))
      )}
    </div>
  );
};

const DecisionsPage = () => (
  <PageBoundary>
    <DecisionsContent />
  </PageBoundary>
);

export default DecisionsPage;

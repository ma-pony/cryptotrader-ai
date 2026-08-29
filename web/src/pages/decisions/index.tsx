import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Link } from 'react-router';

import { PageBoundary } from '@/components/ui/page-boundary';
import { PageHeader } from '@/components/ui/page-header';
import { useDecisions } from '@/hooks/use-decisions';

const PAGE_SIZE = 20;

const DecisionsContent = () => {
  const { t } = useTranslation('decisions');
  const [page] = useState(1);

  const { data, isLoading } = useDecisions({ page, size: PAGE_SIZE });

  return (
    <div className="space-y-6">
      <PageHeader title={t('title')} />
      {isLoading ? <p>Loading…</p> : data?.items.map((cycle) => <Link key={cycle.cycle_id} to={`/cycles/${cycle.cycle_id}`} className="block rounded border border-border p-3"><b>{cycle.cycle_id}</b><span className="ml-2 text-xs text-muted-foreground">{cycle.books.map((book) => `${book.book_id} (${book.capital_scope})`).join(', ')}</span>{cycle.requires_attention ? <span className="ml-2 text-amber-600">Requires attention</span> : null}</Link>)}
    </div>
  );
};

const DecisionsPage = () => (
  <PageBoundary>
    <DecisionsContent />
  </PageBoundary>
);

export default DecisionsPage;

import { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useNavigate, useParams } from 'react-router';

import { DecisionDetailPanel } from '@/components/decision-detail/decision-detail-panel';
import { PageBoundary } from '@/components/ui/page-boundary';
import { PageHeader } from '@/components/ui/page-header';
import { useDecisions } from '@/hooks/use-decisions';
import type { DecisionListFilter } from '@/types/api';

import { DecisionsFilterBar } from './components/decisions-filter-bar';
import { DecisionsTable } from './components/decisions-table';

const PAGE_SIZE = 20;

const DecisionsContent = () => {
  const { t } = useTranslation('decisions');
  const navigate = useNavigate();
  const { commitId } = useParams<{ commitId?: string }>();
  const [filter, setFilter] = useState<DecisionListFilter>({ page: 1, size: PAGE_SIZE });

  const { data, isLoading } = useDecisions(filter);

  const pairs = useMemo(() => {
    if (!data) return [];
    return [...new Set(data.items.map((d) => d.pair))].sort();
  }, [data]);

  const handleSelect = useCallback(
    (hash: string) => {
      void navigate(hash === commitId ? '/decisions' : `/decisions/${hash}`, { replace: true });
    },
    [navigate, commitId],
  );

  const handlePageChange = useCallback(
    (page: number) => setFilter((prev) => ({ ...prev, page })),
    [],
  );

  return (
    <div className="space-y-6">
      <PageHeader title={t('title')} />
      <DecisionsFilterBar filter={filter} onFilterChange={setFilter} pairs={pairs} />

      <div className="grid min-h-[600px] grid-cols-1 gap-4 lg:grid-cols-5">
        <div className="overflow-y-auto rounded-md border lg:col-span-2">
          <DecisionsTable
            data={data}
            isLoading={isLoading}
            selectedHash={commitId}
            onSelect={handleSelect}
            onPageChange={handlePageChange}
          />
        </div>
        <div className="overflow-hidden rounded-md border lg:col-span-3">
          <DecisionDetailPanel commitHash={commitId} />
        </div>
      </div>
    </div>
  );
};

const DecisionsPage = () => (
  <PageBoundary>
    <DecisionsContent />
  </PageBoundary>
);

export default DecisionsPage;

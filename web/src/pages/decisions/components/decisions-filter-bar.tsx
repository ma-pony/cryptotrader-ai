import { useTranslation } from 'react-i18next';

import { Button } from '@/components/ui/button';
import type { CycleStatus, DecisionListFilter } from '@/types/api';

interface Props {
  filter: DecisionListFilter;
  onFilterChange: (next: DecisionListFilter) => void;
  pairs: string[];
}

export const DecisionsFilterBar = ({ filter, onFilterChange, pairs }: Props) => {
  const { t } = useTranslation('decisions');
  const statuses: CycleStatus[] = [
    'completed',
    'no_change',
    'awaiting_approval',
    'approval_rejected',
    'component_failed',
    'cycle_failed',
    'risk_rejected',
    'execution_failed',
    'cancelled',
  ];

  return (
    <div className="flex flex-wrap items-end gap-3">
      <label className="space-y-1 text-xs">
        <span className="text-muted-foreground">{t('filter.pair')}</span>
        <select
          className="block h-8 rounded-md border border-input bg-background px-2 text-sm"
          value={filter.pair ?? ''}
          onChange={(e) => {
            const { pair: _, ...rest } = filter;
            onFilterChange(e.target.value ? { ...rest, pair: e.target.value, page: 1 } : { ...rest, page: 1 });
          }}
        >
          <option value="">{t('filter.all_pairs')}</option>
          {pairs.map((p) => (
            <option key={p} value={p}>{p}</option>
          ))}
        </select>
      </label>
      <label className="space-y-1 text-xs">
        <span className="text-muted-foreground">{t('filter.status')}</span>
        <select
          className="block h-8 rounded-md border border-input bg-background px-2 text-sm"
          value={filter.status ?? ''}
          onChange={(e) => {
            const { status: _, ...rest } = filter;
            const status = e.target.value as CycleStatus;
            onFilterChange(status ? { ...rest, status, page: 1 } : { ...rest, page: 1 });
          }}
        >
          <option value="">{t('filter.all_statuses')}</option>
          {statuses.map((status) => (
            <option key={status} value={status}>{t(`status.${status}`)}</option>
          ))}
        </select>
      </label>
      <Button
        variant="ghost"
        size="sm"
        onClick={() => onFilterChange(filter.size ? { page: 1, size: filter.size } : { page: 1 })}
      >
        {t('filter.reset', { defaultValue: '重置' })}
      </Button>
    </div>
  );
};

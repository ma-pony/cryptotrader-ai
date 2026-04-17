import { useTranslation } from 'react-i18next';

import { Button } from '@/components/ui/button';
import type { DecisionListFilter } from '@/types/api';

interface Props {
  filter: DecisionListFilter;
  onFilterChange: (next: DecisionListFilter) => void;
  pairs: string[];
}

export const DecisionsFilterBar = ({ filter, onFilterChange, pairs }: Props) => {
  const { t } = useTranslation('decisions');

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
        <span className="text-muted-foreground">{t('filter.from')}</span>
        <input
          type="date"
          className="block h-8 rounded-md border border-input bg-background px-2 text-sm"
          value={filter.from ?? ''}
          onChange={(e) => {
            const { from: _, ...rest } = filter;
            onFilterChange(e.target.value ? { ...rest, from: e.target.value, page: 1 } : { ...rest, page: 1 });
          }}
        />
      </label>
      <label className="space-y-1 text-xs">
        <span className="text-muted-foreground">{t('filter.to')}</span>
        <input
          type="date"
          className="block h-8 rounded-md border border-input bg-background px-2 text-sm"
          value={filter.to ?? ''}
          onChange={(e) => {
            const { to: _, ...rest } = filter;
            onFilterChange(e.target.value ? { ...rest, to: e.target.value, page: 1 } : { ...rest, page: 1 });
          }}
        />
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

import { ChevronLeft, ChevronRight } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/cn';
import { formatCurrency, formatDateTime } from '@/lib/format';
import type { DecisionListItem, PaginatedDecisions } from '@/types/api';

interface Props {
  data: PaginatedDecisions | undefined;
  isLoading: boolean;
  selectedHash: string | undefined;
  onSelect: (hash: string) => void;
  onPageChange: (page: number) => void;
}

const actionVariant = (action: string): 'success' | 'destructive' | 'secondary' => {
  const a = action.toLowerCase();
  if (a === 'buy' || a === 'long') return 'success';
  if (a === 'sell' || a === 'short') return 'destructive';
  return 'secondary';
};

const Row = ({ item, isSelected, onSelect }: { item: DecisionListItem; isSelected: boolean; onSelect: () => void }) => (
  <tr
    className={cn(
      'cursor-pointer border-b border-border text-xs hover:bg-muted/50 transition-colors',
      isSelected && 'bg-primary/5',
    )}
    onClick={onSelect}
    role="row"
    aria-selected={isSelected}
    tabIndex={0}
    onKeyDown={(e) => { if (e.key === 'Enter') onSelect(); }}
  >
    <td className="px-2 py-1.5 tabular-nums text-muted-foreground">{formatDateTime(item.ts)}</td>
    <td className="px-2 py-1.5 font-medium">{item.pair}</td>
    <td className="px-2 py-1.5 tabular-nums">{formatCurrency(item.price)}</td>
    <td className="px-2 py-1.5">
      <Badge variant={actionVariant(item.verdict.action)}>{item.verdict.action.toUpperCase()}</Badge>
    </td>
    <td className="px-2 py-1.5 tabular-nums">{(item.verdict.size * 100).toFixed(0)}%</td>
    <td className="px-2 py-1.5 tabular-nums">{(item.verdict.confidence * 100).toFixed(0)}%</td>
  </tr>
);

export const DecisionsTable = ({ data, isLoading, selectedHash, onSelect, onPageChange }: Props) => {
  const { t } = useTranslation('decisions');

  if (isLoading) {
    return (
      <div className="space-y-1">
        {Array.from({ length: 8 }).map((_, i) => (
          <div key={i} className="h-8 rounded bg-muted animate-pulse" />
        ))}
      </div>
    );
  }

  if (!data || data.items.length === 0) {
    return <p className="text-sm text-muted-foreground py-8 text-center">{t('list.empty')}</p>;
  }

  return (
    <div className="space-y-2">
      <div className="overflow-x-auto">
        <table className="w-full text-left" role="grid" aria-label={t('title')}>
          <thead>
            <tr className="border-b border-border text-xs text-muted-foreground">
              <th className="px-2 py-1.5 font-medium">{t('list.ts')}</th>
              <th className="px-2 py-1.5 font-medium">{t('list.pair')}</th>
              <th className="px-2 py-1.5 font-medium">{t('list.price')}</th>
              <th className="px-2 py-1.5 font-medium">{t('list.action')}</th>
              <th className="px-2 py-1.5 font-medium">{t('list.size')}</th>
              <th className="px-2 py-1.5 font-medium">{t('list.confidence')}</th>
            </tr>
          </thead>
          <tbody>
            {data.items.map((item) => (
              <Row
                key={item.commit_hash}
                item={item}
                isSelected={item.commit_hash === selectedHash}
                onSelect={() => onSelect(item.commit_hash)}
              />
            ))}
          </tbody>
        </table>
      </div>

      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <span>{t('list.page_info', { defaultValue: '第 {{page}} 页 / 共 {{total}} 条', page: data.page, total: data.total })}</span>
        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="icon"
            disabled={data.page <= 1}
            onClick={() => onPageChange(data.page - 1)}
            aria-label={t('list.prev_page', { defaultValue: '上一页' })}
          >
            <ChevronLeft className="h-4 w-4" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            disabled={!data.has_next}
            onClick={() => onPageChange(data.page + 1)}
            aria-label={t('list.next_page', { defaultValue: '下一页' })}
          >
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </div>
  );
};

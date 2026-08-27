import { ChevronLeft, ChevronRight } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import { Button } from '@/components/ui/button';
import { DirChip } from '@/components/ui/dir-chip';
import { StatusPill } from '@/components/ui/status-pill';
import { cn } from '@/lib/cn';
import { formatCurrency, formatDateTime } from '@/lib/format';
import type { DecisionListItem, PaginatedDecisions } from '@/types/api';

interface Props {
  data: PaginatedDecisions | undefined;
  isLoading: boolean;
  selectedCycleId: string | undefined;
  onSelect: (cycleId: string) => void;
  onPageChange: (page: number) => void;
}

const Row = ({
  item,
  isSelected,
  onSelect,
}: {
  item: DecisionListItem;
  isSelected: boolean;
  onSelect: () => void;
}) => {
  const { t } = useTranslation('decisions');
  const score = item.fused_score;
  const action = item.target_position?.side ?? 'flat';
  const scoreMagnitude = Math.abs(score ?? 0);
  return (
    <button
      type="button"
      onClick={onSelect}
      onKeyDown={(e) => {
        if (e.key === 'Enter') onSelect();
      }}
      aria-pressed={isSelected}
      aria-label={`Cycle ${item.cycle_id.slice(0, 8)} ${item.pair}`}
      className={cn(
        'grid w-full items-center gap-3 border-b border-border px-4 py-3 text-left transition-colors',
        'grid-cols-[minmax(110px,110px)_minmax(92px,92px)_minmax(84px,84px)_minmax(72px,72px)_minmax(96px,96px)_1fr_minmax(96px,96px)]',
        isSelected ? 'bg-amber-500/10 border-l-2 border-l-amber-500 pl-[14px]' : 'hover:bg-muted/60 border-l-2 border-l-transparent',
      )}
    >
      <div className="font-mono text-[11px] text-muted-foreground">{formatDateTime(item.ts)}</div>
      <div className="font-mono text-[11px] text-muted-foreground">{item.cycle_id.slice(0, 8)}</div>
      <div className="font-mono text-xs font-medium">{item.pair}</div>
      <DirChip dir={action} />
      <div className="flex items-center gap-2">
        <div className="h-1 w-10 overflow-hidden rounded bg-muted">
          <div
            className={cn('h-full rounded', score !== null && score > 0 ? 'bg-trade-long' : score !== null && score < 0 ? 'bg-trade-short' : 'bg-muted-foreground')}
            style={{ width: `${scoreMagnitude * 100}%` }}
          />
        </div>
        <span className="font-mono text-[11px] text-muted-foreground">
          {score === null ? '—' : `${score >= 0 ? '+' : ''}${score.toFixed(2)}`}
        </span>
      </div>
      <div className="flex min-w-0 items-center gap-2 text-xs text-muted-foreground">
        <span className="font-mono tabular-nums">{formatCurrency(item.price)}</span>
        <span className="truncate">{item.target_position ? `${(item.target_position.size_ratio * 100).toFixed(0)}%` : '—'}</span>
        <span className="truncate font-mono text-[10px]">R{item.profile_revision}</span>
      </div>
      <div className="flex flex-col items-end gap-0.5">
        <StatusPill tone={item.status === 'completed' || item.status === 'no_change' ? 'success' : item.status === 'awaiting_approval' ? 'warning' : item.status === 'cancelled' ? 'default' : 'danger'}>
          {t(`status.${item.status}`)}
        </StatusPill>
      </div>
    </button>
  );
};

export const DecisionsTable = ({ data, isLoading, selectedCycleId, onSelect, onPageChange }: Props) => {
  const { t } = useTranslation('decisions');

  if (isLoading) {
    return (
      <div className="space-y-1 p-2">
        {Array.from({ length: 8 }).map((_, i) => (
          <div key={i} className="h-10 rounded bg-muted animate-pulse" />
        ))}
      </div>
    );
  }

  if (!data || data.items.length === 0) {
    return <p className="text-sm text-muted-foreground py-8 text-center">{t('list.empty')}</p>;
  }

  return (
    <div className="flex h-full flex-col">
      <div
        className={cn(
          'grid items-center gap-3 border-b border-border bg-muted/30 px-4 py-2 text-[10px] uppercase tracking-wider font-medium text-muted-foreground',
          'grid-cols-[minmax(110px,110px)_minmax(92px,92px)_minmax(84px,84px)_minmax(72px,72px)_minmax(96px,96px)_1fr_minmax(96px,96px)]',
        )}
      >
        <div>{t('list.ts', { defaultValue: '时间' })}</div>
        <div>Cycle</div>
        <div>{t('list.pair', { defaultValue: '交易对' })}</div>
        <div>{t('list.action', { defaultValue: '动作' })}</div>
        <div>{t('list.fused_score')}</div>
        <div>{t('list.price', { defaultValue: '价格 · 仓位' })}</div>
        <div className="text-right">状态</div>
      </div>

      <div className="flex-1 overflow-y-auto">
        {data.items.map((item) => (
          <Row
            key={item.cycle_id}
            item={item}
            isSelected={item.cycle_id === selectedCycleId}
            onSelect={() => onSelect(item.cycle_id)}
          />
        ))}
      </div>

      <div className="flex items-center justify-between border-t border-border px-4 py-2 text-xs text-muted-foreground">
        <span>
          {t('list.page_info', {
            defaultValue: '第 {{page}} 页 / 共 {{total}} 条',
            page: data.page,
            total: data.total,
          })}
        </span>
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

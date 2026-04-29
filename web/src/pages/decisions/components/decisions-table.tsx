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
  selectedHash: string | undefined;
  onSelect: (hash: string) => void;
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
  const conf = item.verdict.confidence;
  const action = item.verdict.action?.toLowerCase() ?? 'hold';
  return (
    <button
      type="button"
      onClick={onSelect}
      onKeyDown={(e) => {
        if (e.key === 'Enter') onSelect();
      }}
      aria-pressed={isSelected}
      aria-label={`Decision ${item.commit_hash.slice(0, 8)} ${item.pair}`}
      className={cn(
        'grid w-full items-center gap-3 border-b border-border px-4 py-3 text-left transition-colors',
        'grid-cols-[minmax(110px,110px)_minmax(92px,92px)_minmax(84px,84px)_minmax(72px,72px)_minmax(96px,96px)_1fr_minmax(96px,96px)]',
        isSelected ? 'bg-amber-500/10 border-l-2 border-l-amber-500 pl-[14px]' : 'hover:bg-muted/60 border-l-2 border-l-transparent',
      )}
    >
      <div className="font-mono text-[11px] text-muted-foreground">{formatDateTime(item.ts)}</div>
      <div className="font-mono text-[11px] text-muted-foreground">{item.commit_hash.slice(0, 8)}</div>
      <div className="font-mono text-xs font-medium">{item.pair}</div>
      <DirChip dir={action} />
      <div className="flex items-center gap-2">
        <div className="h-1 w-10 overflow-hidden rounded bg-muted">
          <div
            className={cn('h-full rounded', conf > 0.6 ? 'bg-amber-500' : 'bg-muted-foreground')}
            style={{ width: `${conf * 100}%` }}
          />
        </div>
        <span className="font-mono text-[11px] text-muted-foreground">
          {(conf * 100).toFixed(0)}%
        </span>
      </div>
      <div className="flex min-w-0 items-center gap-2 text-xs text-muted-foreground">
        <span className="font-mono tabular-nums">{formatCurrency(item.price)}</span>
        <span className="truncate">仓位 {(item.verdict.size * 100).toFixed(0)}%</span>
        {item.debate_status ? (
          <span
            className={cn(
              'truncate text-[10px]',
              item.debate_status.startsWith('skipped') ? 'text-muted-foreground' : 'text-violet-500',
            )}
          >
            {item.debate_status === 'skipped-consensus'
              ? '跳过·强共识'
              : item.debate_status === 'skipped-confusion'
                ? '跳过·共同困惑'
                : item.debate_status}
          </span>
        ) : null}
        {item.reject_reason ? (
          <span className="truncate text-[10px] text-trade-short">{item.reject_reason}</span>
        ) : null}
      </div>
      <div className="flex flex-col items-end gap-0.5">
        {item.pnl != null ? (
          <span
            className={cn(
              'font-mono text-[11px] font-medium tabular-nums',
              item.pnl >= 0 ? 'text-trade-long' : 'text-trade-short',
            )}
          >
            {item.pnl >= 0 ? '+' : '-'}${Math.abs(item.pnl).toLocaleString('en-US', { maximumFractionDigits: 0 })}
          </span>
        ) : action === 'hold' ? (
          <StatusPill tone="default">观望</StatusPill>
        ) : item.reject_reason ? (
          <StatusPill tone="danger">拒</StatusPill>
        ) : item.is_filled ? (
          <StatusPill tone="default">持仓</StatusPill>
        ) : (
          <StatusPill tone={action === 'long' || action === 'buy' ? 'success' : 'danger'}>
            {action.toUpperCase()}
          </StatusPill>
        )}
      </div>
    </button>
  );
};

export const DecisionsTable = ({ data, isLoading, selectedHash, onSelect, onPageChange }: Props) => {
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
        <div>Commit</div>
        <div>{t('list.pair', { defaultValue: '交易对' })}</div>
        <div>{t('list.action', { defaultValue: '动作' })}</div>
        <div>{t('list.confidence', { defaultValue: '置信度' })}</div>
        <div>{t('list.price', { defaultValue: '价格 · 仓位' })}</div>
        <div className="text-right">状态</div>
      </div>

      <div className="flex-1 overflow-y-auto">
        {data.items.map((item) => (
          <Row
            key={item.commit_hash}
            item={item}
            isSelected={item.commit_hash === selectedHash}
            onSelect={() => onSelect(item.commit_hash)}
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

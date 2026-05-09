/**
 * spec 018 — RecentTransitions component
 * FSM 状态转换事件流（rule_id / old_state → new_state / triggered_by / timestamp）
 */

import { useTranslation } from 'react-i18next';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { cn } from '@/lib/cn';
import { formatDateTime } from '@/lib/format';

import type { TransitionItem } from '../queries';
import { useRecentTransitions } from '../queries';

const STATE_COLOR: Record<string, string> = {
  observed: 'text-muted-foreground',
  probationary: 'text-blue-400',
  active: 'text-trade-long',
  deprecated: 'text-amber-400',
  archived: 'text-red-400',
};

const TransitionRow = ({ item }: { item: TransitionItem }) => {
  const oldColor = STATE_COLOR[item.old_state] ?? 'text-muted-foreground';
  const newColor = STATE_COLOR[item.new_state] ?? 'text-muted-foreground';

  return (
    <div className="flex items-start gap-3 border-b border-border py-2 last:border-0">
      <div className="mt-1 h-2 w-2 shrink-0 rounded-full bg-amber-500/60" />
      <div className="min-w-0 flex-1 space-y-0.5">
        <div className="flex items-center gap-1.5 flex-wrap text-xs">
          <span className="font-mono text-foreground/80 truncate">{item.rule_id}</span>
        </div>
        <div className="flex items-center gap-1.5 text-[10px]">
          <span className={cn('font-semibold', oldColor)}>{item.old_state}</span>
          <span className="text-muted-foreground">→</span>
          <span className={cn('font-semibold', newColor)}>{item.new_state}</span>
          <span className="text-muted-foreground/60">via {item.triggered_by}</span>
        </div>
        <div className="text-[10px] text-muted-foreground/50">{formatDateTime(item.timestamp)}</div>
      </div>
    </div>
  );
};

export const RecentTransitions = () => {
  const { t } = useTranslation('memory');
  const { data, isLoading } = useRecentTransitions();
  const items = data?.items ?? [];

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-medium">
          {t('transitions.title', { defaultValue: 'FSM 状态转换' })}
        </CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="text-xs text-muted-foreground">{t('loading', { defaultValue: '加载中…' })}</div>
        ) : items.length === 0 ? (
          <div className="py-6 text-center text-xs text-muted-foreground">
            {t('transitions.empty', { defaultValue: '最近无状态转换' })}
          </div>
        ) : (
          <div className="max-h-72 overflow-y-auto pr-1">
            {items.map((item) => (
              <TransitionRow key={`${item.rule_id}-${item.timestamp}`} item={item} />
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
};

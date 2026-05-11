/**
 * spec 018 — CasesTimeline component
 * 最近 24h IVE classification 时间线（按 timestamp 倒序）
 */

import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { cn } from '@/lib/cn';
import { formatDateTime } from '@/lib/format';

import type { CaseItem } from '../queries';
import { useMemoryCases } from '../queries';

const FAILURE_TYPE_COLOR: Record<string, string> = {
  fundamental: 'text-red-400',
  implementation: 'text-amber-400',
  noise: 'text-muted-foreground',
};

const IVEBadge = ({ type }: { type: string }) => {
  const color = FAILURE_TYPE_COLOR[type] ?? 'text-muted-foreground';
  return (
    <span className={cn('text-[10px] font-semibold uppercase tracking-wide', color)}>
      {type}
    </span>
  );
};

const CaseRow = ({ item }: { item: CaseItem }) => {
  const pnlStr =
    item.final_pnl != null
      ? `${item.final_pnl >= 0 ? '+' : ''}${item.final_pnl.toFixed(2)}`
      : 'N/A';
  const pnlColor =
    item.final_pnl == null
      ? 'text-muted-foreground'
      : item.final_pnl >= 0
        ? 'text-trade-long'
        : 'text-trade-short';

  return (
    <div className="flex items-start gap-3 border-b border-border py-2 last:border-0">
      {/* timeline dot */}
      <div className="mt-1 h-2 w-2 shrink-0 rounded-full bg-border" />
      <div className="min-w-0 flex-1 space-y-0.5">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="font-mono text-xs text-foreground/80">{item.cycle_id}</span>
          <span className="text-[10px] text-muted-foreground">{item.pair}</span>
          <span className="text-[10px] text-muted-foreground">{item.verdict_action}</span>
          <span className={cn('ml-auto font-mono text-xs tabular-nums', pnlColor)}>{pnlStr}</span>
        </div>
        {item.ive_classification ? (
          <div className="flex items-center gap-1.5 flex-wrap">
            <IVEBadge type={item.ive_classification.failure_type} />
            <span className="text-[10px] text-muted-foreground line-clamp-1">
              {item.ive_classification.reasoning}
            </span>
          </div>
        ) : (
          <span className="text-[10px] text-muted-foreground/50">待分类</span>
        )}
        <div className="text-[10px] text-muted-foreground/50">{formatDateTime(item.timestamp)}</div>
      </div>
    </div>
  );
};

// Bucket the "now" reference to a 5-minute grid so the React Query key only
// changes every 5 min, instead of every render — otherwise `new Date()` on each
// render yields a fresh `from` string, the query key churns, and `isLoading`
// stays true forever showing a permanent "加载中…" spinner.
const FIVE_MIN_MS = 5 * 60 * 1000;
const ONE_DAY_MS = 24 * 60 * 60 * 1000;

export const CasesTimeline = () => {
  const { t } = useTranslation('memory');
  // Default: last 24h, snapped to a 5-minute bucket for query-key stability.
  const from = useMemo(() => {
    const bucketedNow = Math.floor(Date.now() / FIVE_MIN_MS) * FIVE_MIN_MS;
    return new Date(bucketedNow - ONE_DAY_MS).toISOString();
  }, []);
  const { data, isLoading } = useMemoryCases({ from });
  const items = data?.items ?? [];

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-medium">
          {t('cases_timeline.title', { defaultValue: 'IVE 分类时间线 (24h)' })}
        </CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="text-xs text-muted-foreground">{t('loading', { defaultValue: '加载中…' })}</div>
        ) : items.length === 0 ? (
          <div className="py-6 text-center text-xs text-muted-foreground">
            {t('cases_timeline.empty', { defaultValue: '最近 24h 无 case 记录' })}
          </div>
        ) : (
          <div className="max-h-80 overflow-y-auto pr-1">
            {items.map((item) => (
              <CaseRow key={item.cycle_id} item={item} />
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
};

import { CheckCircle2, MinusCircle, ShieldAlert, TrendingDown, TrendingUp } from 'lucide-react';
import { useMemo, type ReactNode } from 'react';
import { useTranslation } from 'react-i18next';
import { useNavigate } from 'react-router';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { EmptyState } from '@/components/ui/empty-state';
import { Skeleton } from '@/components/ui/skeleton';
import { useDecisions } from '@/hooks/use-decisions';
import { useRiskStatus } from '@/hooks/use-risk-status';
import { cn } from '@/lib/cn';
import { formatDateTime } from '@/lib/format';

interface FeedItem {
  id: string;
  ts: string;
  kind: 'filled' | 'hold' | 'skipped' | 'block';
  primary: ReactNode;
  secondary?: ReactNode;
  /** Click handler — navigates to the detail view. */
  onClick?: () => void;
}

const ICONS: Record<FeedItem['kind'], { Icon: typeof CheckCircle2; tone: string }> = {
  filled: { Icon: CheckCircle2, tone: 'text-trade-long' },
  hold: { Icon: MinusCircle, tone: 'text-muted-foreground' },
  skipped: { Icon: MinusCircle, tone: 'text-muted-foreground/70' },
  block: { Icon: ShieldAlert, tone: 'text-trade-short' },
};

const ActionBadge = ({
  action,
  pnl,
}: {
  action: string;
  // exactOptionalPropertyTypes: caller may pass an explicit ``undefined``
  // when the field is absent on the API payload.
  pnl?: number | null | undefined;
}) => {
  const lower = action.toLowerCase();
  if (lower === 'long' || lower === 'buy') {
    return (
      <span className="inline-flex items-center gap-1 font-mono text-trade-long">
        <TrendingUp className="h-3 w-3" />
        LONG
        {Number.isFinite(pnl ?? NaN) ? (
          <span className="ml-1 text-[10px] tabular-nums">
            {(pnl ?? 0) >= 0 ? '+' : ''}
            {((pnl ?? 0) * 100).toFixed(2)}%
          </span>
        ) : null}
      </span>
    );
  }
  if (lower === 'short' || lower === 'sell') {
    return (
      <span className="inline-flex items-center gap-1 font-mono text-trade-short">
        <TrendingDown className="h-3 w-3" />
        SHORT
        {Number.isFinite(pnl ?? NaN) ? (
          <span className="ml-1 text-[10px] tabular-nums">
            {(pnl ?? 0) >= 0 ? '+' : ''}
            {((pnl ?? 0) * 100).toFixed(2)}%
          </span>
        ) : null}
      </span>
    );
  }
  if (lower === 'close') {
    return (
      <span className="inline-flex items-center gap-1 font-mono text-amber-500">
        <MinusCircle className="h-3 w-3" />
        CLOSE
        {Number.isFinite(pnl ?? NaN) ? (
          <span className="ml-1 text-[10px] tabular-nums">
            {(pnl ?? 0) >= 0 ? '+' : ''}
            {((pnl ?? 0) * 100).toFixed(2)}%
          </span>
        ) : null}
      </span>
    );
  }
  return <span className="font-mono text-muted-foreground">HOLD</span>;
};

/**
 * Dashboard activity feed — merges the most recent decisions with the most
 * recent risk blocks into a single time-sorted stream so the user can answer
 * "what happened today?" without bouncing between Decisions / Risk pages.
 *
 * Reuses ``useDecisions`` and ``useRiskStatus`` (both already polled
 * elsewhere in the dashboard via React Query, so this is a free subscription
 * via the cache).
 */
export const ActivityFeed = ({ limit = 12 }: { limit?: number }) => {
  const { t } = useTranslation('dashboard');
  const navigate = useNavigate();
  const decisions = useDecisions({ page: 1, size: limit });
  const risk = useRiskStatus();

  const items = useMemo<FeedItem[]>(() => {
    const out: FeedItem[] = [];

    for (const d of decisions.data?.items ?? []) {
      const action = d.verdict.action;
      const isHold = action.toLowerCase() === 'hold' || action.toLowerCase() === 'close';
      const kind: FeedItem['kind'] = d.is_filled
        ? 'filled'
        : isHold
          ? 'hold'
          : 'skipped';
      out.push({
        id: `decision-${d.commit_hash}`,
        ts: d.ts,
        kind,
        primary: (
          <span className="flex items-center gap-2">
            <ActionBadge action={action} pnl={d.pnl} />
            <span className="font-mono text-foreground">{d.pair_display ?? d.pair}</span>
          </span>
        ),
        secondary: d.is_filled ? null : t('activity.not_filled', { defaultValue: '未成交' }),
        onClick: () => void navigate(`/decisions/${d.commit_hash}`),
      });
    }

    for (const b of risk.data?.recent_blocks ?? []) {
      out.push({
        id: `block-${b.commit_hash}-${b.rule}`,
        ts: b.ts,
        kind: 'block',
        primary: (
          <span className="flex items-center gap-2">
            <span className="font-mono text-trade-short">{b.rule}</span>
            <span className="truncate text-muted-foreground">{b.detail}</span>
          </span>
        ),
        onClick: () => void navigate('/risk'),
      });
    }

    out.sort((a, b) => (a.ts < b.ts ? 1 : -1));
    return out.slice(0, limit);
  }, [decisions.data, risk.data, limit, navigate, t]);

  const isLoading = decisions.isLoading || risk.isLoading;

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between p-4 pb-2">
        <CardTitle className="text-sm">
          {t('activity.title', { defaultValue: '今日动向' })}
        </CardTitle>
        <span className="text-[10px] uppercase tracking-wider text-muted-foreground">
          {t('activity.subtitle', { defaultValue: '决策 · 风控' })}
        </span>
      </CardHeader>
      <CardContent className="p-2 pt-0">
        {isLoading ? (
          <div className="space-y-2 px-2 py-2">
            {Array.from({ length: 4 }).map((_, i) => (
              <Skeleton key={i} className="h-7 w-full" />
            ))}
          </div>
        ) : items.length === 0 ? (
          <EmptyState
            size="compact"
            title={t('activity.empty', { defaultValue: '今日无决策记录' })}
            description={t('activity.empty_hint', {
              defaultValue: '调度器运行后，决策与拦截会在这里实时出现',
            })}
          />
        ) : (
          <ul className="divide-y divide-border">
            {items.map((it) => {
              const { Icon, tone } = ICONS[it.kind];
              return (
                <li key={it.id}>
                  <button
                    type="button"
                    onClick={it.onClick}
                    className={cn(
                      'flex w-full items-center gap-3 rounded-md px-2 py-2 text-left text-xs',
                      'transition-colors hover:bg-muted/40 focus-visible:bg-muted/40 focus-visible:outline-none',
                    )}
                  >
                    <span className="w-12 shrink-0 font-mono text-[11px] text-muted-foreground tabular-nums">
                      {formatDateTime(it.ts).slice(-8, -3)}
                    </span>
                    <Icon className={cn('h-3.5 w-3.5 shrink-0', tone)} aria-hidden />
                    <span className="min-w-0 flex-1 truncate">{it.primary}</span>
                    {it.secondary ? (
                      <span className="shrink-0 text-[10px] text-muted-foreground">
                        {it.secondary}
                      </span>
                    ) : null}
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </CardContent>
    </Card>
  );
};

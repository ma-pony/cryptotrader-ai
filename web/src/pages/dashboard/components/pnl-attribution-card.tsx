/**
 * PnL Attribution Card — break down each window's equity change into 4
 * fundamental buckets to resolve the "总权益涨但交易亏" paradox:
 *
 *   delta = realized + funding + fees + unrealized_delta
 *
 *   realized          — close-action commit PnL (from local journal)
 *   funding           — perp funding rate net (from OKX history)
 *   fees              — trading fees (from OKX history)
 *   unrealized_delta  — derived; captures mark-to-market drift on open positions
 *
 * If the live exchange is unreachable (paper mode / network blip), `funding`
 * and `fees` are zero and `exchange_data_available=false`; we then show a
 * fallback display crediting the residual to "未实现 / 资金费 / 手续费 合计".
 */

import { ChevronDown, ChevronRight } from 'lucide-react';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { cn } from '@/lib/cn';
import { formatCurrency } from '@/lib/format';
import type { Portfolio } from '@/types/api';

interface Props {
  data: Portfolio | undefined;
  isLoading?: boolean;
}

const WINDOW_LABEL: Record<string, string> = {
  '24h': '24h',
  '7d': '7 天',
  '30d': '30 天',
};

const SignedCurrency = ({
  value,
  muteZero = true,
}: {
  value: number;
  muteZero?: boolean;
}) => {
  const sign = value > 0 ? '+' : '';
  const tone =
    muteZero && value === 0
      ? 'text-muted-foreground'
      : value > 0
        ? 'text-trade-long'
        : 'text-trade-short';
  return (
    <span className={cn('font-mono tabular-nums', tone)}>
      {sign}
      {formatCurrency(value)}
    </span>
  );
};

const Row = ({
  label,
  value,
  hint,
  emphasis = false,
}: {
  label: string;
  value: number;
  hint?: string;
  emphasis?: boolean;
}) => (
  <div
    className={cn(
      'flex items-baseline justify-between gap-3 py-1',
      emphasis && 'border-t border-border pt-1.5 font-semibold',
    )}
  >
    <span className="text-[11px] uppercase tracking-wider text-muted-foreground">
      {label}
      {hint ? <span className="ml-1 normal-case text-muted-foreground/60">{hint}</span> : null}
    </span>
    <SignedCurrency value={value} />
  </div>
);

export const PnlAttributionCard = ({ data, isLoading }: Props) => {
  const { t } = useTranslation('dashboard');
  const [open, setOpen] = useState(false);
  const breakdowns = data?.pnl_breakdowns ?? [];

  // Compact 1-line summary that's always visible — gives the headline number
  // for the most relevant window (24h) without forcing the user to expand.
  const summary24h = breakdowns.find((b) => b.window === '24h');

  return (
    <Card>
      <CardHeader
        className="cursor-pointer pb-2 transition-colors hover:bg-muted/30"
        onClick={() => setOpen((v) => !v)}
        role="button"
        aria-expanded={open}
      >
        <CardTitle className="flex items-center gap-2 text-sm font-medium">
          {open ? (
            <ChevronDown className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
          ) : (
            <ChevronRight className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
          )}
          {t('pnl_attribution.title', { defaultValue: '收益归因' })}
          {summary24h ? (
            <span className="ml-1 text-[11px] font-normal text-muted-foreground">
              24h{' '}
              <span
                className={cn(
                  'tabular-nums',
                  summary24h.delta > 0
                    ? 'text-trade-long'
                    : summary24h.delta < 0
                      ? 'text-trade-short'
                      : '',
                )}
              >
                {summary24h.delta > 0 ? '+' : ''}
                {formatCurrency(summary24h.delta)}
              </span>
              {' = 平仓 '}
              <SignedCurrency value={summary24h.realized} />
              {' + 资金费 '}
              <SignedCurrency value={summary24h.funding} />
              {' + 其他 '}
              <SignedCurrency value={summary24h.fees + summary24h.unrealized_delta} />
            </span>
          ) : (
            <span className="ml-2 text-[10px] font-normal text-muted-foreground">
              {t('pnl_attribution.subtitle', {
                defaultValue: '总权益变化 = 平仓 PnL + 资金费 + 手续费 + 浮盈变化',
              })}
            </span>
          )}
        </CardTitle>
      </CardHeader>
      {!open ? null : (
      <CardContent className="space-y-4">
        {isLoading ? (
          <Skeleton className="h-32 w-full" />
        ) : breakdowns.length === 0 ? (
          <div className="py-4 text-center text-xs text-muted-foreground">
            {t('pnl_attribution.empty', { defaultValue: '暂无足够 snapshot 数据计算归因' })}
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
            {breakdowns.map((b) => {
              const exchangeOk = b.exchange_data_available;
              return (
                <div key={b.window} className="rounded-md border border-border p-3">
                  <div className="mb-1 flex items-center justify-between">
                    <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                      {WINDOW_LABEL[b.window] ?? b.window}
                    </span>
                    {!exchangeOk ? (
                      <span
                        title="OKX 历史数据不可用，资金费/手续费 合并进未实现"
                        className="text-[9px] uppercase tracking-wider text-amber-500/80"
                      >
                        partial
                      </span>
                    ) : null}
                  </div>
                  <Row label="平仓 PnL" value={b.realized} hint="realized" />
                  {exchangeOk ? (
                    <>
                      <Row label="资金费" value={b.funding} hint="perp funding" />
                      <Row label="手续费" value={b.fees} hint="maker/taker" />
                      <Row label="浮盈变化" value={b.unrealized_delta} hint="mark-to-market" />
                    </>
                  ) : (
                    <Row
                      label="非平仓项"
                      value={b.funding + b.fees + b.unrealized_delta}
                      hint="funding+fees+浮盈"
                    />
                  )}
                  <Row label="净 Δ 权益" value={b.delta} emphasis />
                </div>
              );
            })}
          </div>
        )}
        <p className="border-t border-border/40 pt-2 text-[10px] leading-relaxed text-muted-foreground/70">
          {t('pnl_attribution.footnote', {
            defaultValue:
              '识别策略真实业绩：四桶恒等式 Δ权益 = 平仓 PnL + 资金费 + 手续费 + 浮盈变化。资金费 / 手续费 直接来自 OKX API（缓存 60s）。沙盒环境的资金费率与主网不同，看到异常高的资金费收入属正常。如果显示 "partial"，说明 OKX 历史 API 不可用，剩余三项合并显示。',
          })}
        </p>
      </CardContent>
      )}
    </Card>
  );
};

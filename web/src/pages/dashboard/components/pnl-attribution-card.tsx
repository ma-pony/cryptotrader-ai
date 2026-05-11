/**
 * PnL Attribution Card — break down each window's equity change into:
 *   realized (close-action PnL)
 *   non-realized (浮盈变化 + 资金费 + 手续费 + 充提)
 *
 * Helps the user resolve "总权益涨了但总收益是负的" by showing the actual
 * source of the equity drift (typically funding income on perp positions).
 */

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
  const breakdowns = data?.pnl_breakdowns ?? [];

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium">
          {t('pnl_attribution.title', { defaultValue: '收益归因' })}
          <span className="ml-2 text-[10px] font-normal text-muted-foreground">
            {t('pnl_attribution.subtitle', {
              defaultValue: '总权益变化拆解：交易盈亏 vs 浮盈/资金费/充提',
            })}
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {isLoading ? (
          <Skeleton className="h-24 w-full" />
        ) : breakdowns.length === 0 ? (
          <div className="py-4 text-center text-xs text-muted-foreground">
            {t('pnl_attribution.empty', { defaultValue: '暂无足够 snapshot 数据计算归因' })}
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
            {breakdowns.map((b) => (
              <div key={b.window} className="rounded-md border border-border p-3">
                <div className="mb-1 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                  {WINDOW_LABEL[b.window] ?? b.window}
                </div>
                <Row label="已实现" value={b.realized} hint="平仓 PnL" />
                <Row
                  label="非实现"
                  value={b.non_realized}
                  hint="浮盈 + 资金费 + 手续费"
                />
                {b.external_flow_hint !== 0 ? (
                  <Row
                    label="疑似充提"
                    value={b.external_flow_hint}
                    hint="cash Δ 偏离"
                  />
                ) : null}
                <Row label="净 Δ 权益" value={b.delta} emphasis />
              </div>
            ))}
          </div>
        )}
        <p className="border-t border-border/40 pt-2 text-[10px] leading-relaxed text-muted-foreground/70">
          {t('pnl_attribution.footnote', {
            defaultValue:
              '"非实现" = (净权益变化 − 已实现)。包含未平仓浮盈变化、perp 资金费净收支、手续费。当账户出现"非交易现金流入"（充值 / 内部转账）时，"疑似充提" 列会出现非零提示，避免把它误算成策略业绩。',
          })}
        </p>
      </CardContent>
    </Card>
  );
};

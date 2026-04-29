import { useTranslation } from 'react-i18next';

import { Card, CardContent } from '@/components/ui/card';
import { formatCurrency, formatDateTime } from '@/lib/format';
import type { ConsensusMetrics } from '@/types/api';

import { SectionHeader } from './section-header';

interface Props {
  pair: string;
  price: number;
  ts: string;
  traceId: string | null | undefined;
  consensus: ConsensusMetrics | null | undefined;
}

interface KpiProps {
  label: string;
  value: React.ReactNode;
  sub?: string | undefined;
}

const Kpi = ({ label, value, sub }: KpiProps) => (
  <div className="flex flex-col gap-1 min-w-0">
    <div className="text-[10px] uppercase tracking-wider text-muted-foreground font-medium">{label}</div>
    <div className="text-lg font-semibold tabular-nums truncate">{value}</div>
    {sub ? <div className="text-[10px] text-muted-foreground">{sub}</div> : null}
  </div>
);

export const MarketContextSection = ({ pair, price, ts, traceId, consensus }: Props) => {
  const { t } = useTranslation('decisions');
  const strength = consensus?.strength ?? null;
  const dispersion = consensus?.dispersion ?? null;
  const meanScore = consensus?.mean_score ?? null;

  return (
    <section aria-label={t('detail.context', { defaultValue: '市场上下文' })}>
      <SectionHeader id="sec-context" index={2} title={t('detail.context', { defaultValue: '市场上下文' })} />
      <Card>
        <CardContent className="grid grid-cols-4 gap-4 p-4">
          <Kpi
            label={t('detail.pair', { defaultValue: '交易对' })}
            value={<span className="font-mono">{pair}</span>}
            sub={formatDateTime(ts)}
          />
          <Kpi label={t('detail.price', { defaultValue: '价格' })} value={formatCurrency(price)} />
          <Kpi
            label={t('detail.consensus_strength', { defaultValue: '共识强度' })}
            value={strength != null ? strength.toFixed(2) : '—'}
            sub={meanScore != null ? `均值倾向 ${meanScore.toFixed(2)}` : undefined}
          />
          <Kpi
            label={t('detail.dispersion', { defaultValue: '分歧度' })}
            value={dispersion != null ? dispersion.toFixed(2) : '—'}
            sub={traceId ? `trace ${traceId.slice(0, 10)}…` : undefined}
          />
        </CardContent>
      </Card>
    </section>
  );
};

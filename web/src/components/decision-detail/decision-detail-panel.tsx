import { Activity, BrainCircuit, GitMerge, ShieldCheck, XCircle } from 'lucide-react';
import type { TFunction } from 'i18next';
import { useTranslation } from 'react-i18next';
import { Link } from 'react-router';

import { PairBadge } from '@/components/PairBadge';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { useDecisionDetail } from '@/hooks/use-decision-detail';
import { cn } from '@/lib/cn';
import { formatCurrency, formatDateTime } from '@/lib/format';
import type { ComponentSignal, DecisionDetail, TargetPosition } from '@/types/api';

interface Props {
  cycleId: string | undefined;
}

const componentName = (id: string) => {
  if (id === 'kronos') return 'Kronos';
  if (id === 'llm_committee') return 'LLM 四智能体委员会';
  return id;
};

const targetLabel = (target: TargetPosition | null, t: TFunction<'decisions'>) => {
  if (!target) return t('detail.target.none');
  const ratio = `${(target.size_ratio * 100).toFixed(0)}%`;
  if (target.side === 'long') return t('detail.target.long', { ratio });
  if (target.side === 'short') return t('detail.target.short', { ratio });
  return t('detail.target.flat');
};

const directionLabel = (direction: ComponentSignal['direction']) => {
  if (direction === 'long') return '看多';
  if (direction === 'short') return '看空';
  return '中性';
};

const directionClass = (direction: ComponentSignal['direction']) =>
  direction === 'long'
    ? 'text-trade-long'
    : direction === 'short'
      ? 'text-trade-short'
      : 'text-muted-foreground';

const statusTone = (status: DecisionDetail['status']) => {
  if (status === 'completed' || status === 'no_change') return 'success' as const;
  if (status === 'awaiting_approval') return 'default' as const;
  if (status === 'cancelled') return 'secondary' as const;
  return 'destructive' as const;
};

const Section = ({
  title,
  eyebrow,
  children,
}: {
  title: string;
  eyebrow: string;
  children: React.ReactNode;
}) => (
  <section className="rounded-xl border border-border bg-card p-4">
    <div className="mb-3 text-[10px] font-semibold uppercase tracking-[0.18em] text-muted-foreground">
      {eyebrow}
    </div>
    <h3 className="mb-4 text-sm font-semibold text-foreground">{title}</h3>
    {children}
  </section>
);

const AuditFact = ({ label, value }: { label: string; value: React.ReactNode }) => (
  <div className="min-w-0 rounded-md border border-border bg-background/55 p-3">
    <dt className="text-[10px] uppercase tracking-wider text-muted-foreground">{label}</dt>
    <dd className="mt-1 break-words font-mono text-xs font-semibold text-foreground">{value}</dd>
  </div>
);

const RiskTargetAudit = ({ data }: { data: DecisionDetail }) => {
  const { t } = useTranslation('decisions');
  const result = data.risk_result;
  if (!result) return null;

  return (
    <dl className="mt-3 grid gap-2 sm:grid-cols-3">
      <AuditFact label={t('detail.audit.original_target')} value={targetLabel(data.target_position, t)} />
      <AuditFact label={t('detail.audit.adjusted_target')} value={targetLabel(result.target, t)} />
      <AuditFact
        label={t('detail.audit.cap_source')}
        value={result.cap_source || t('detail.audit.no_cap')}
      />
    </dl>
  );
};

const amountLabel = (amount: number) => amount.toLocaleString(undefined, { maximumFractionDigits: 12 });

const ExecutionAudit = ({ result }: { result: NonNullable<DecisionDetail['execution_result']> }) => {
  const { t } = useTranslation('decisions');
  const hasProtectionOrders = Boolean(result.algo_id) || result.retained_algo_ids.length > 0;
  const hasSafetyFacts = result.orders.length > 0 || hasProtectionOrders || Boolean(result.protection_trigger);
  if (!hasSafetyFacts) return null;

  return (
    <div className="mt-4 space-y-3 border-t border-border pt-4">
      {result.orders.length > 0 ? (
        <div>
          <h4 className="text-xs font-semibold text-foreground">{t('detail.audit.order_execution')}</h4>
          <div className="mt-2 grid gap-2">
            {result.orders.map((order, index) => (
              <article
                key={`${order.intent.pair}-${order.intent.side}-${index}`}
                className="min-w-0 rounded-md border border-border bg-background/55 p-3"
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <span className="text-xs font-semibold">
                    {t('detail.audit.intent')} · {t(`detail.audit.${order.intent.side}`)}
                  </span>
                  <Badge variant="secondary">{order.status}</Badge>
                </div>
                <div className="mt-2 break-all font-mono text-[10px] text-muted-foreground">
                  {order.intent.pair} · {order.intent.reduce_only ? t('detail.audit.reduce_only') : t('detail.audit.opens_exposure')}
                </div>
                <dl className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-3">
                  <AuditFact label={t('detail.audit.intent_amount')} value={amountLabel(order.intent.amount)} />
                  <AuditFact label={t('detail.audit.filled_amount')} value={amountLabel(order.filled_amount)} />
                  {order.exchange_id ? (
                    <AuditFact label={t('detail.audit.order_id')} value={order.exchange_id} />
                  ) : null}
                </dl>
              </article>
            ))}
          </div>
        </div>
      ) : null}

      {hasProtectionOrders ? (
        <div>
          <h4 className="text-xs font-semibold text-foreground">{t('detail.audit.protection_orders')}</h4>
          <dl className="mt-2 grid gap-2 sm:grid-cols-2">
            {result.algo_id ? (
              <AuditFact label={t('detail.audit.new_algo_id')} value={result.algo_id} />
            ) : null}
            {result.retained_algo_ids.length > 0 ? (
              <AuditFact
                label={t('detail.audit.retained_algo_ids')}
                value={result.retained_algo_ids.join(', ')}
              />
            ) : null}
          </dl>
        </div>
      ) : null}

      {result.protection_trigger ? (
        <div>
          <h4 className="text-xs font-semibold text-foreground">{t('detail.audit.protection_trigger')}</h4>
          <dl className="mt-2 grid grid-cols-2 gap-2 sm:grid-cols-4">
            <AuditFact label={t('detail.audit.trigger_reason')} value={result.protection_trigger.trigger_reason} />
            <AuditFact label={t('detail.audit.trigger_price')} value={formatCurrency(result.protection_trigger.trigger_price)} />
            <AuditFact label={t('detail.audit.trigger_order_id')} value={result.protection_trigger.order_id} />
            <AuditFact label={t('detail.audit.trigger_algo_id')} value={result.protection_trigger.algo_id} />
          </dl>
        </div>
      ) : null}
    </div>
  );
};

export const CycleDecisionDetail = ({ data }: { data: DecisionDetail }) => {
  const { t } = useTranslation('decisions');
  const committee = data.components.find((item) => item.component_id === 'llm_committee');
  const analyses = committee?.details.analyses ? Object.values(committee.details.analyses) : [];
  const turns = committee?.details.debate_turns ?? [];

  return (
    <div className="space-y-4 p-4">
      <section
        className="overflow-hidden rounded-xl border border-amber-500/35 bg-card p-5"
        style={{ backgroundImage: 'linear-gradient(135deg, color-mix(in oklch, var(--amber-500) 12%, transparent), transparent 58%)' }}
      >
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <div className="mb-2 flex flex-wrap items-center gap-2">
              <Badge variant={statusTone(data.status)}>{t(`status.${data.status}`)}</Badge>
              <span className="font-mono text-[11px] text-muted-foreground">{data.cycle_id}</span>
              <span className="rounded border border-border px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground">
                Revision {data.profile_revision}
              </span>
            </div>
            <div className="flex flex-wrap items-center gap-2 text-lg font-semibold">
              <PairBadge pair={data.pair} pairDisplay={data.pair_display} marketType={data.market_type} />
              <span className={cn('font-mono', data.target_position?.side === 'long' ? 'text-trade-long' : data.target_position?.side === 'short' ? 'text-trade-short' : 'text-muted-foreground')}>
                {targetLabel(data.target_position, t)}
              </span>
            </div>
            <p className="mt-2 text-xs text-muted-foreground">{formatDateTime(data.ts)} · {data.context.mode.toUpperCase()}</p>
          </div>
          {data.context.available ? (
            <div className="text-right">
              <div className="font-mono text-xl font-semibold">{formatCurrency(data.context.current_price)}</div>
              <div className="mt-1 text-[10px] uppercase tracking-wider text-muted-foreground">ATR {formatCurrency(data.context.atr)}</div>
            </div>
          ) : (
            <div className="text-right text-xs text-muted-foreground">{t('detail.context_unavailable')}</div>
          )}
        </div>
      </section>

      {data.component_error ? (
        <section className="rounded-xl border border-destructive/40 bg-destructive/5 p-4">
          <div className="flex items-center gap-2 text-sm font-semibold text-destructive"><XCircle className="h-4 w-4" />组件执行失败</div>
          {Object.entries(data.component_error).map(([id, error]) => (
            <p key={id} className="mt-2 font-mono text-xs text-muted-foreground">{componentName(id)} · {error}</p>
          ))}
        </section>
      ) : null}

      {data.error && !data.component_error ? (
        <section className="rounded-xl border border-destructive/40 bg-destructive/5 p-4">
          <div className="flex items-center gap-2 text-sm font-semibold text-destructive">
            <XCircle className="h-4 w-4" />{t('detail.cycle_error')}
          </div>
          <p className="mt-2 break-words font-mono text-xs text-muted-foreground">{data.error}</p>
        </section>
      ) : null}

      <Section title="独立组件信号" eyebrow="01 · Components">
        <div className="grid gap-3 xl:grid-cols-2">
          {data.components.map((signal) => (
            <article key={signal.component_id} className="rounded-lg border border-border bg-background/60 p-4">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <h4 className="text-sm font-semibold">{componentName(signal.component_id)}</h4>
                  <code className="text-[10px] text-muted-foreground">{signal.component_id}</code>
                </div>
                <div className={cn('text-right font-mono', directionClass(signal.direction))}>
                  <div className="text-sm font-semibold">{directionLabel(signal.direction)}</div>
                  <div className="text-[11px]">{(signal.confidence * 100).toFixed(0)}%</div>
                </div>
              </div>
              <p className="mt-3 text-xs leading-5 text-muted-foreground">{signal.reasoning}</p>
            </article>
          ))}
          {data.components.length === 0 ? <p className="text-xs text-muted-foreground">本周期没有成功的组件输出。</p> : null}
        </div>
      </Section>

      <Section title="确定性加权融合" eyebrow="02 · Fusion">
        {data.fusion ? (
          <div className="space-y-3">
            <div className="flex items-end justify-between gap-4">
              <div>
                <div className="text-[10px] uppercase tracking-wider text-muted-foreground">融合分数</div>
                <div className={cn('font-mono text-3xl font-semibold', data.fusion.score > 0 ? 'text-trade-long' : data.fusion.score < 0 ? 'text-trade-short' : 'text-muted-foreground')}>
                  {data.fusion.score >= 0 ? '+' : ''}{data.fusion.score.toFixed(2)}
                </div>
              </div>
              <GitMerge className="h-7 w-7 text-amber-500" />
            </div>
            <div className="space-y-2">
              {data.fusion.contributions.map((item) => (
                <div key={item.component_id} className="grid grid-cols-[1fr_auto_auto] items-center gap-4 rounded-md border border-border px-3 py-2 text-xs">
                  <span className="font-medium">{componentName(item.component_id)}</span>
                  <span className="font-mono text-muted-foreground">{(item.weight * 100).toFixed(0)}% × {item.signed_score >= 0 ? '+' : ''}{item.signed_score.toFixed(2)}</span>
                  <span className={cn('w-14 text-right font-mono font-semibold', item.weighted_score > 0 ? 'text-trade-long' : item.weighted_score < 0 ? 'text-trade-short' : 'text-muted-foreground')}>
                    {item.weighted_score >= 0 ? '+' : ''}{item.weighted_score.toFixed(2)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        ) : <p className="text-xs text-muted-foreground">组件未全部成功，因此未进入融合。</p>}
      </Section>

      {committee ? (
        <Section title="内部辩论" eyebrow="03 · LLM Committee">
          <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <BrainCircuit className="h-4 w-4 text-violet-500" />
              {analyses.length} 个智能体 · {turns.length} 个辩论回合记录
            </div>
            <Link to={`/debate/${data.cycle_id}`} className="text-xs font-medium text-amber-500 hover:underline">查看完整辩论 →</Link>
          </div>
          <div className="grid gap-2 sm:grid-cols-2">
            {analyses.map((analysis) => (
              <div key={analysis.agent_id} className="rounded-md border border-border bg-background/60 p-3">
                <div className="flex items-center justify-between gap-2 text-xs">
                  <span className="font-semibold">{analysis.agent_id}</span>
                  <span className="font-mono text-muted-foreground">{analysis.direction} {(analysis.confidence * 100).toFixed(0)}%</span>
                </div>
                <p className="mt-2 line-clamp-3 text-xs leading-5 text-muted-foreground">{analysis.reasoning}</p>
              </div>
            ))}
          </div>
          {turns.length > 0 ? (
            <div className="mt-3 space-y-2">
              {turns.slice(0, 3).map((turn, index) => (
                <div key={`${turn.round}-${turn.from}-${index}`} className="rounded-md border-l-2 border-l-violet-500 bg-muted/35 px-3 py-2 text-xs">
                  <div className="mb-1 flex flex-wrap items-center gap-2 font-mono text-[10px] text-muted-foreground">
                    <span>R{turn.round}</span><span>{turn.from}</span><span>→</span><span>{turn.to ?? 'committee'}</span><span>{turn.move}</span>
                  </div>
                  <p className="leading-5 text-muted-foreground">{turn.reasoning}</p>
                </div>
              ))}
            </div>
          ) : (
            <p className="mt-3 rounded-md bg-muted/35 p-3 text-xs text-muted-foreground">{committee.details.debate_skip_reason || '委员会一致性足够，本轮无需交叉挑战。'}</p>
          )}
        </Section>
      ) : null}

      <div className="grid gap-4 xl:grid-cols-2">
        <Section title="目标仓位与退出保护" eyebrow="04 · Target Plan">
          <div className="flex items-center gap-3">
            <Activity className="h-5 w-5 text-amber-500" />
            <span className="text-base font-semibold">{targetLabel(data.target_position, t)}</span>
          </div>
          {data.trade_plan ? (
            <dl className="mt-4 grid grid-cols-2 gap-3 text-xs">
              <div className="rounded-md bg-muted/40 p-3"><dt className="text-muted-foreground">ATR 止损</dt><dd className="mt-1 font-mono font-semibold">{data.trade_plan.stop_loss === null ? '—' : formatCurrency(data.trade_plan.stop_loss)}</dd></div>
              <div className="rounded-md bg-muted/40 p-3"><dt className="text-muted-foreground">目标止盈</dt><dd className="mt-1 font-mono font-semibold">{data.trade_plan.take_profit === null ? '—' : formatCurrency(data.trade_plan.take_profit)}</dd></div>
            </dl>
          ) : null}
        </Section>

        <Section title="审批、风控与执行" eyebrow="05 · Controls">
          <div className="space-y-2 text-xs">
            <div className="flex items-center justify-between rounded-md border border-border px-3 py-2"><span>HITL</span><Badge variant="secondary">{data.hitl_result?.status ?? 'not_required'}</Badge></div>
            <div className="flex items-center justify-between rounded-md border border-border px-3 py-2"><span className="flex items-center gap-2"><ShieldCheck className="h-4 w-4" />风控</span><Badge variant={data.risk_result?.passed ? 'success' : data.risk_result ? 'destructive' : 'secondary'}>{data.risk_result ? (data.risk_result.passed ? 'PASS' : 'REJECT') : 'NOT_RUN'}</Badge></div>
            <div className="flex items-center justify-between rounded-md border border-border px-3 py-2"><span>执行</span><Badge variant={data.execution_result?.succeeded ? 'success' : data.execution_result ? 'destructive' : 'secondary'}>{data.execution_result ? (data.execution_result.succeeded ? 'SUCCEEDED' : 'FAILED') : 'NOT_RUN'}</Badge></div>
          </div>
          <RiskTargetAudit data={data} />
          {data.risk_result?.reason ? <p className="mt-3 text-xs text-destructive">{data.risk_result.reason}</p> : null}
          {data.execution_result?.error ? <p className="mt-3 text-xs text-destructive">{data.execution_result.error}</p> : null}
          {data.execution_result ? <ExecutionAudit result={data.execution_result} /> : null}
        </Section>
      </div>
    </div>
  );
};

export const DecisionDetailPanel = ({ cycleId }: Props) => {
  const { t } = useTranslation('decisions');
  const { data, isLoading, isError } = useDecisionDetail(cycleId);

  if (!cycleId) return <div className="flex h-full items-center justify-center text-sm text-muted-foreground">{t('detail.select_hint')}</div>;
  if (isLoading) return <div className="space-y-3 p-4">{Array.from({ length: 5 }).map((_, index) => <Skeleton key={index} className="h-24 w-full" />)}</div>;
  if (isError || !data) return <div className="p-4 text-sm text-destructive">{t('detail.load_error')}</div>;

  return <div className="h-full overflow-y-auto"><CycleDecisionDetail data={data} /></div>;
};

import { useTranslation } from 'react-i18next';
import { useParams } from 'react-router';
import type { z } from 'zod';

import { PageBoundary } from '@/components/ui/page-boundary';
import { PageHeader } from '@/components/ui/page-header';
import { useMultiVenueCycle } from '@/hooks/use-multi-venue-cycles';
import { formatCycleStatus } from '@/lib/cycle-status';
import type { CycleSchema } from '@/types/api.schema';

type Cycle = z.infer<typeof CycleSchema>;
type Book = Cycle['books'][number];
type Connection = Book['connections'][number];

const scalar = (item: unknown) => {
  if (item === null || item === undefined) return '—';
  if (typeof item === 'string' || typeof item === 'number' || typeof item === 'boolean') return `${item}`;
  return '—';
};

const Field = ({ label, children }: { label: string; children: unknown }) => (
  <div className="grid grid-cols-[minmax(9rem,auto)_1fr] gap-x-3 text-sm">
    <dt className="text-muted-foreground">{label}</dt>
    <dd className="break-all font-mono text-foreground">{scalar(children)}</dd>
  </div>
);

const Section = ({ title, children, open = false }: { title: string; children: React.ReactNode; open?: boolean }) => (
  <details className="rounded-md border border-border bg-card p-3" open={open}>
    <summary className="cursor-pointer font-medium text-foreground">{title}</summary>
    <dl className="mt-3 space-y-2">{children}</dl>
  </details>
);

const ConnectionPortfolio = ({ portfolio, title }: { portfolio: Connection['portfolio_before']; title: string }) => {
  const { t } = useTranslation('cycles');
  if (!portfolio) return null;
  return (
    <Section title={title}>
      <Field label={t('equity')}>{portfolio.equity}</Field>
      <Field label={t('balances')}>
        {portfolio.balances.map((item) => `${item.asset}: ${item.amount}`).join(', ')}
      </Field>
      <Field
        label={t('position')}
      >{`${portfolio.position.pair} ${portfolio.position.signed_amount} / ${portfolio.position.signed_notional}`}</Field>
    </Section>
  );
};

const BookPortfolio = ({ portfolio, title }: { portfolio: Book['portfolio_before']; title: string }) => {
  const { t } = useTranslation('cycles');
  if (!portfolio) return null;
  return (
    <Section title={title}>
      <Field label={t('equity')}>{portfolio.total_equity}</Field>
      <Field label={t('targetExposure')}>{portfolio.total_signed_notional}</Field>
      {portfolio.connections.map((connection) => (
        <div key={connection.connection_id} className="rounded border border-border p-2">
          <Field label={t('connection')}>{connection.connection_id}</Field>
          <Field label={t('balances')}>
            {connection.balances.map((balance) => `${balance.asset}: ${balance.amount}`).join(', ')}
          </Field>
          <Field label={t('position')}>{`${connection.position.pair} ${connection.position.signed_amount}`}</Field>
        </div>
      ))}
    </Section>
  );
};

const ConnectionAudit = ({ connection }: { connection: Connection }) => {
  const { t } = useTranslation('cycles');
  const { plan, execution } = connection;
  return (
    <article className="space-y-3 rounded-md border border-border p-3">
      <h4 className="font-medium">
        {t('connection')}: {connection.connection_id}
      </h4>
      {connection.unavailable ? (
        <p className="rounded bg-destructive/10 p-2 text-sm text-destructive">{t('unavailable')}</p>
      ) : null}
      {execution?.requires_attention ? (
        <p className="rounded bg-amber-500/10 p-2 text-sm text-amber-700">{t('attention')}</p>
      ) : null}
      {execution?.error_operation ? (
        <p className="text-sm text-destructive">
          {t('errorOperation')}: {execution.error_operation}
        </p>
      ) : null}
      <ConnectionPortfolio title={t('portfolioBefore')} portfolio={connection.portfolio_before} />
      <ConnectionPortfolio title={t('portfolioAfter')} portfolio={connection.portfolio_after} />
      {connection.risk ? (
        <Section title={t('risk')}>
          <Field label={t('passed')}>{connection.risk.passed}</Field>
          <Field label={t('increase')}>{connection.risk.risk_increase}</Field>
          <Field label={t('reasoning')}>{connection.risk.reason}</Field>
          <Field label={t('operation')}>{connection.risk.operation}</Field>
        </Section>
      ) : null}
      {plan ? (
        <Section title={t('plan')}>
          <Field label={t('pair')}>{plan.pair.symbol}</Field>
          <Field
            label={t('notional')}
          >{`${plan.current_signed_notional} / ${plan.target_signed_notional} / ${plan.delta_signed_notional}`}</Field>
          <Field
            label={t('amount')}
          >{`${plan.current_signed_amount} / ${plan.target_signed_amount} / ${plan.delta_signed_amount}`}</Field>
          <Field label={t('side')}>{`${plan.side}; ${plan.reduce_only}; ${plan.market_type}`}</Field>
          <Field
            label={t('quote')}
          >{`${plan.quote.pair.symbol} ${plan.quote.bid}/${plan.quote.ask}/${plan.quote.last}`}</Field>
          <Field
            label={t('capabilities')}
          >{`${plan.capabilities.market_types.join(', ')}; ${plan.capabilities.supported_order_types.join(', ')}`}</Field>
          <Field
            label={t('protection')}
          >{`${plan.stop_loss} / ${plan.take_profit}; ${plan.old_protection_ids.join(', ')}`}</Field>
        </Section>
      ) : null}
      {execution ? (
        <Section title={t('execution')} open>
          <Field label={t('statusLabel')}>{formatCycleStatus(t, execution.status)}</Field>
          <Field
            label={t('targetLabel')}
          >{`${execution.target_signed_notional} / ${execution.target_signed_amount}`}</Field>
          <Field label={t('orders')}>
            {execution.orders.map((order) => `${order.id} ${order.side} ${order.amount} ${order.status}`).join('; ')}
          </Field>
          <Field label={t('protection')}>
            {execution.protection
              ? `${execution.protection.protection_ids.join(', ')} ${execution.protection.active}`
              : '—'}
          </Field>
          <Field
            label={t('compensation')}
          >{`${execution.compensation.operation}; ${execution.compensation.attempted}/${execution.compensation.succeeded}; ${execution.compensation.safe_signed_amount}`}</Field>
          <Field label={t('finalPosition')}>
            {execution.final_position
              ? `${execution.final_position.position.pair.symbol} ${execution.final_position.position.signed_amount}; ${execution.final_position.protection_ids.join(', ')}`
              : '—'}
          </Field>
          <Field label={t('trace')}>{execution.trace.join(' → ')}</Field>
          <Field label={t('executionQuote')}>
            {execution.execution_quote
              ? `${execution.execution_quote.bid}/${execution.execution_quote.ask}/${execution.execution_quote.last}`
              : '—'}
          </Field>
        </Section>
      ) : null}
    </article>
  );
};

const BookAudit = ({ book }: { book: Book }) => {
  const { t } = useTranslation('cycles');
  return (
    <article className="space-y-3 rounded-lg border border-border bg-card p-4">
      <h3 className="font-semibold">
        {t('book')}: {book.book_id}
      </h3>
      <dl className="grid gap-1 sm:grid-cols-2">
        <Field label={t('scope')}>{book.capital_scope}</Field>
        <Field label={t('revision')}>{book.config_revision}</Field>
        <Field label={t('statusLabel')}>{formatCycleStatus(t, book.status)}</Field>
        <Field label={t('pair')}>{book.pair}</Field>
        <Field label={t('marketType')}>{book.market_type}</Field>
        <Field
          label={t('hitl')}
        >{`${book.hitl.status} ${book.hitl.approval_id ?? ''} R${book.hitl.config_revision}`}</Field>
      </dl>
      {book.failure ? (
        <p className="text-sm text-destructive">
          {t('failure')}: {book.failure.stage}
        </p>
      ) : null}
      {book.errors.map((error) => (
        <p key={error} className="text-sm text-destructive">
          {t('errors')}: {error}
        </p>
      ))}
      {book.execution?.requires_attention ? (
        <p className="rounded bg-amber-500/10 p-2 text-sm text-amber-700">{t('attention')}</p>
      ) : null}
      <Section title={t('risk')}>
        <Field label={t('passed')}>{book.risk?.passed}</Field>
        <Field label={t('requestedExposure')}>{book.requested_target_exposure}</Field>
        <Field label={t('targetExposure')}>{book.target_exposure}</Field>
        <Field label={t('ready')}>{book.ready}</Field>
        <Field label={t('execution')}>
          {book.execution
            ? `${formatCycleStatus(t, book.execution.status)}; ${t('reallocated')}: ${book.execution.reallocated}`
            : '—'}
        </Field>
        <Field label={t('reason')}>{book.risk?.reason}</Field>
        <Field label={t('cap')}>{book.risk?.cap_source}</Field>
      </Section>
      <BookPortfolio title={t('portfolioBefore')} portfolio={book.portfolio_before} />
      <BookPortfolio title={t('portfolioAfter')} portfolio={book.portfolio_after} />
      <Field label={t('afterAvailable')}>{book.portfolio_after_available}</Field>
      <Section title={t('connections')} open>
        {book.connections.map((connection) => (
          <ConnectionAudit key={connection.connection_id} connection={connection} />
        ))}
      </Section>
    </article>
  );
};

export default function CycleDetailPage() {
  const { t } = useTranslation('cycles');
  const { cycleId } = useParams<{ cycleId: string }>();
  const cycle = useMultiVenueCycle(cycleId);
  if (cycle.isError)
    return (
      <PageBoundary>
        <p className="text-destructive">{t('loadError')}</p>
      </PageBoundary>
    );
  if (!cycle.data)
    return (
      <PageBoundary>
        <p>{t('loading')}</p>
      </PageBoundary>
    );
  const item = cycle.data;
  return (
    <PageBoundary>
      <div className="space-y-5">
        <PageHeader
          title={t('detailTitle', { id: item.cycle_id })}
          subtitle={`${t('source')}: ${item.market_data_source_id}`}
        />
        {item.requires_attention ? (
          <p className="rounded border border-amber-500 bg-amber-500/10 p-3 font-medium">{t('attention')}</p>
        ) : null}
        <Section title={t('shared')} open>
          {item.shared_signals.components.map((component) => (
            <article key={component.component_id} className="rounded border border-border p-3">
              <h3>{component.component_id}</h3>
              <Field label={t('confidence')}>{component.confidence}</Field>
              <Field label={t('reasoning')}>{component.reasoning}</Field>
              <Field label={t('details')}>
                {component.details
                  .map(
                    (detail) =>
                      `${detail.key}: ${detail.value.string_value ?? detail.value.number_value ?? detail.value.boolean_value ?? detail.value.kind}`,
                  )
                  .join(', ')}
              </Field>
            </article>
          ))}
          {item.shared_signals.fused ? (
            <Section title={t('fused')}>
              <Field label={t('score')}>{item.shared_signals.fused.score}</Field>
              <Field label={t('reasoning')}>{item.shared_signals.fused.reasoning}</Field>
              <Field label={t('contributions')}>
                {item.shared_signals.fused.contributions
                  .map(
                    (entry) => `${entry.component_id}: ${entry.weight}/${entry.signed_score}/${entry.weighted_score}`,
                  )
                  .join(', ')}
              </Field>
            </Section>
          ) : null}
          {item.shared_signals.target_position ? (
            <Field
              label={t('target')}
            >{`${item.shared_signals.target_position.side} ${item.shared_signals.target_position.size_ratio}`}</Field>
          ) : null}
        </Section>
        <div className="space-y-4">
          {item.books.map((book) => (
            <BookAudit key={book.book_id} book={book} />
          ))}
        </div>
      </div>
    </PageBoundary>
  );
}

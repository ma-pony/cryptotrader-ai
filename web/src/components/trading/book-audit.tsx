import { useTranslation } from 'react-i18next';
import type { z } from 'zod';

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

const Field = ({ label, children }: { label: string; children: React.ReactNode }) => (
  <div className="grid grid-cols-[minmax(9rem,auto)_1fr] gap-x-3 text-sm">
    <dt className="text-muted-foreground">{label}</dt>
    <dd className="break-all font-mono text-foreground">
      {typeof children === 'string' || typeof children === 'number' || typeof children === 'boolean'
        ? scalar(children)
        : children}
    </dd>
  </div>
);

const Section = ({ title, children, open = false }: { title: string; children: React.ReactNode; open?: boolean }) => (
  <details className="rounded-md border border-border bg-card p-3" open={open}>
    <summary className="cursor-pointer font-medium text-foreground">{title}</summary>
    <dl className="mt-3 space-y-2">{children}</dl>
  </details>
);

const ProtectionDetails = ({
  protection,
  title,
}: {
  protection: NonNullable<Connection['execution']>['protection'];
  title: string;
}) => {
  const { t } = useTranslation('cycles');
  if (!protection) return <Field label={title}>—</Field>;
  return (
    <Section title={title}>
      <Field label={t('protectionIds')}>{protection.protection_ids.join(', ')}</Field>
      <Field label={t('pair')}>{protection.pair.symbol}</Field>
      <Field label={t('positionSide')}>{protection.position_side}</Field>
      <Field label={t('amount')}>{protection.amount}</Field>
      <Field label={t('stopLoss')}>{protection.stop_loss ?? '—'}</Field>
      <Field label={t('takeProfit')}>{protection.take_profit ?? '—'}</Field>
      <Field label={t('active')}>{protection.active}</Field>
      <Field label={t('triggered')}>{protection.triggered}</Field>
    </Section>
  );
};

const OrderDetails = ({
  order,
  title,
}: {
  order: NonNullable<Connection['execution']>['orders'][number] | null;
  title: string;
}) => {
  const { t } = useTranslation('cycles');
  if (!order) return <Field label={title}>—</Field>;
  return (
    <Section title={title}>
      <Field label={t('orderId')}>{order.id}</Field>
      <Field label={t('pair')}>{order.pair.symbol}</Field>
      <Field label={t('side')}>{order.side}</Field>
      <Field label={t('orderType')}>{order.order_type}</Field>
      <Field label={t('amount')}>{order.amount}</Field>
      <Field label={t('filledAmount')}>{order.filled_amount}</Field>
      <Field label={t('averagePrice')}>{order.average_price ?? '—'}</Field>
      <Field label={t('statusLabel')}>{formatCycleStatus(t, order.status)}</Field>
      <Field label={t('reduceOnly')}>{order.reduce_only}</Field>
    </Section>
  );
};

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
          {plan.capabilities.unknown_fields.length > 0 ? (
            <p className="rounded bg-amber-500/10 p-2 text-sm text-amber-700">历史能力证据不完整</p>
          ) : null}
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
          <Section title={t('orders')}>
            {execution.orders.map((order) => (
              <OrderDetails key={order.id} order={order} title={order.id} />
            ))}
          </Section>
          <ProtectionDetails protection={execution.protection} title={t('protection')} />
          <Section title={t('compensation')}>
            <Field label={t('attempted')}>{execution.compensation.attempted}</Field>
            <Field label={t('succeeded')}>{execution.compensation.succeeded}</Field>
            <Field label={t('operation')}>{execution.compensation.operation}</Field>
            <Field label={t('safeSignedAmount')}>{execution.compensation.safe_signed_amount ?? '—'}</Field>
            <OrderDetails order={execution.compensation.order} title={t('compensationOrder')} />
            <ProtectionDetails
              protection={execution.compensation.required_protection}
              title={t('requiredProtection')}
            />
          </Section>
          {execution.final_position ? (
            <Section title={t('finalPosition')}>
              <Field label={t('pair')}>{execution.final_position.position.pair.symbol}</Field>
              <Field label={t('signedAmount')}>{execution.final_position.position.signed_amount}</Field>
              <Field label={t('signedNotional')}>{execution.final_position.position.signed_notional}</Field>
              <Field label={t('entryPrice')}>{execution.final_position.position.entry_price ?? '—'}</Field>
              <Field label={t('protected')}>{execution.final_position.protected}</Field>
              <Field label={t('protectionIds')}>{execution.final_position.protection_ids.join(', ')}</Field>
              {execution.final_position.protections.map((protection) => (
                <ProtectionDetails
                  key={protection.protection_ids.join('-')}
                  protection={protection}
                  title={t('nestedProtection')}
                />
              ))}
            </Section>
          ) : null}
          <Field label={t('trace')}>{execution.trace.join(' → ')}</Field>
          {execution.quantity_frozen === null ? (
            <p className="rounded bg-amber-500/10 p-2 text-sm text-amber-700">历史数量冻结状态未知</p>
          ) : null}
          {execution.execution_quote ? (
            <Section title={t('executionQuote')}>
              <Field label={t('pair')}>{execution.execution_quote.pair.symbol}</Field>
              <Field label={t('bid')}>{execution.execution_quote.bid}</Field>
              <Field label={t('ask')}>{execution.execution_quote.ask}</Field>
              <Field label={t('last')}>{execution.execution_quote.last}</Field>
            </Section>
          ) : null}
        </Section>
      ) : null}
    </article>
  );
};

export const BookAudit = ({ book }: { book: Book }) => {
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
        <Field label={t('riskRequestedExposure')}>{book.risk?.requested_target_exposure}</Field>
        <Field label={t('cappedExposure')}>{book.risk?.capped_target_exposure}</Field>
        <Field label={t('connectionWeights')}>{book.risk?.connection_weights.join(', ')}</Field>
        {book.risk?.connection_targets.map((target) => (
          <Section key={target.connection_id} title={`${t('connectionTarget')}: ${target.connection_id}`}>
            <Field label={t('book')}>{target.book_id}</Field>
            <Field label={t('connection')}>{target.connection_id}</Field>
            <Field label={t('weight')}>{target.weight}</Field>
            <Field label={t('bookEquity')}>{target.book_equity}</Field>
            <Field label={t('targetExposure')}>{target.target_exposure}</Field>
            <Field label={t('signedNotional')}>{target.target_signed_notional}</Field>
          </Section>
        ))}
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
      {book.reconciliation_required === null ? (
        <p role="status">历史对账状态未知，请勿重复下单</p>
      ) : book.reconciliation_required ? (
        <p role="status">执行已有回包，账户或风险快照需刷新核对。请勿重复下单。</p>
      ) : null}
      <Section title={t('connections')} open>
        {book.connections.map((connection) => (
          <ConnectionAudit key={connection.connection_id} connection={connection} />
        ))}
      </Section>
    </article>
  );
};

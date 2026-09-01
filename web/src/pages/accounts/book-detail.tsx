import { useRef } from 'react';
import { Link, useParams } from 'react-router';
import { usePortfolioBook } from '@/hooks/use-portfolio-books';
import { ConfigurationEditorGate, useConfiguration } from '@/pages/settings/configuration-context';
import { ConfigurationSaveBar } from '@/pages/settings';
import { BookForm } from './book-form';
import { MoneyValues } from './money';
import { ExitDialog } from './exit-dialog';
import { ApprovalQueue } from '@/components/trading/approval-queue';

function BookConfiguration({ bookId }: { bookId: string }) {
  const runtime = useConfiguration();
  const form = useRef<HTMLFormElement>(null);
  const execution = runtime.document!.execution;
  const index = execution.books.findIndex((item) => item.id === bookId);
  const draft = execution.books[index];
  return (
    <form
      ref={form}
      noValidate
      onSubmit={(event) => {
        event.preventDefault();
        void runtime.save('books', form.current ?? undefined);
      }}
    >
      <h2 className="mb-4 text-base font-semibold">资金池配置</h2>
      {draft ? (
        <fieldset disabled={runtime.isSaving}>
          <BookForm
            book={draft}
            index={index}
            saved
            connections={runtime.baseline?.execution.connections ?? []}
            catalog={runtime.catalog.data}
            errors={runtime.errors}
            onChange={(next) => runtime.update('execution', { ...execution, books: execution.books.map((item, i) => (i === index ? next : item)) })}
            onRemove={() => {
              runtime.setBookRows(runtime.bookRows.filter((_, i) => i !== index));
              runtime.update('execution', { ...execution, books: execution.books.filter((_, i) => i !== index) });
            }}
          />
        </fieldset>
      ) : (
        <p role="status" className="configuration-help">
          {runtime.baseline?.execution.books.some((item) => item.id === bookId) ? '待删除，保存配置后生效；放弃修改可恢复。' : '资金池已从配置移除。'}
        </p>
      )}
      <ConfigurationSaveBar section="books" form={form.current} />
    </form>
  );
}

export default function BookDetail() {
  const { bookId = '' } = useParams();
  const query = usePortfolioBook(bookId);
  if (query.isPending) return <p role="status">正在读取资金池…</p>;
  if (!query.data)
    return (
      <p role="alert">
        资金池无法读取。<Link to="/accounts">返回账户列表</Link>
      </p>
    );
  const book = query.data;
  return (
    <section className="space-y-6 text-sm">
      <Link to="/accounts" className="text-primary underline">
        返回账户列表
      </Link>
      <header>
        <h1 className="text-xl font-semibold">{book.label}</h1>
        <p className="configuration-help">
          {book.capital_scope === 'simulated' ? '模拟资金池' : '真实资金池'} · {book.enabled ? '已启用' : '已停用'} ·
          所有已分配账户的最近事实
        </p>
      </header>
      <div className="grid gap-4 md:grid-cols-2">
        <section className="configuration-section">
          <h2>账户权益</h2>
          <MoneyValues values={book.total_equity} />
        </section>
        <section className="configuration-section">
          <h2>全品种净名义金额</h2>
          <MoneyValues values={book.total_signed_notional} />
        </section>
      </div>
      <p className="configuration-help">未进行币种换算。此处是账户事实，不是资金池可用风险预算。</p>
      <section className="configuration-section space-y-3" aria-labelledby="book-risk-heading">
        <h2 id="book-risk-heading" className="font-semibold">
          整池风险
        </h2>
        {book.risk_state ? (
          <>
            <p className="configuration-help">
              风控观察时间：<time dateTime={book.risk_state.observed_at}>{book.risk_state.observed_at}</time>
            </p>
            <dl className="grid grid-cols-2 gap-4 md:grid-cols-3">
              {(
                [
                  ['风控权益', book.risk_state.equity],
                  ['持久化权益峰值', book.risk_state.peak_equity],
                  ['当前总持仓占用', book.risk_state.gross_notional],
                  ['待成交增仓占用', book.risk_state.pending_increase_notional],
                  ['实际已用保证金', book.risk_state.used_margin],
                  ['实际可用保证金', book.risk_state.available_margin],
                ] as const
              ).map(([label, value]) => (
                <div key={label}>
                  <dt className="text-muted-foreground">{label}</dt>
                  <dd className="mt-1 tabular-nums">
                    {value === null ? '未知' : `${value} ${book.risk_state!.valuation_currency}`}
                  </dd>
                </div>
              ))}
            </dl>
            {book.risk_state.completeness.length ? (
              <div role="status">
                <p>风险事实不完整：暂停增仓，仍保留可证明的安全减仓。</p>
                <ul className="mt-2 list-disc pl-5">
                  {book.risk_state.completeness.map((reason) => (
                    <li key={reason}>{reason}</li>
                  ))}
                </ul>
              </div>
            ) : null}
            <p className="configuration-help">
              减仓单成交前不释放占用。原始目标、调整目标与原因可在决策记录和审批计划核对。
            </p>
            <Link className="text-primary underline" to="/decisions">
              查看决策记录
            </Link>
          </>
        ) : (
          <p role="status">尚无资金池风控快照。账户刷新和仅分析不触发交易；交易周期准备完成后在此查看风控依据。</p>
        )}
      </section>
      {book.connections.map((connection) => (
        <article key={connection.connection_id} className="configuration-section">
          <Link
            className="font-medium text-primary underline"
            to={`/accounts/connections/${encodeURIComponent(connection.connection_id)}`}
          >
            {connection.label}
          </Link>
          <MoneyValues values={connection.snapshot ? [connection.snapshot.equity] : []} />
          <ExitDialog account={connection} />
          <p>
            {connection.snapshot?.positions.length ?? '未知'} 个持仓品种 ·{' '}
            {connection.failure_reason ?? '查看账户核对同步时间'}
          </p>
        </article>
      ))}
      <section className="configuration-section" aria-label="资金池待审批">
        <ApprovalQueue bookId={book.book_id} />
      </section>
      <ConfigurationEditorGate>
        <BookConfiguration bookId={bookId} />
      </ConfigurationEditorGate>
    </section>
  );
}

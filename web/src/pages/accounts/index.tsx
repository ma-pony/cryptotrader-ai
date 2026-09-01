import { Link } from 'react-router';
import { useAccounts } from '@/hooks/use-accounts';
import { usePortfolioBooks } from '@/hooks/use-portfolio-books';
import { useConfigurationCatalog } from '@/hooks/use-configuration-catalog';
import { MoneyValues } from './money';

export default function AccountsPage() {
  const accounts = useAccounts();
  const books = usePortfolioBooks();
  const catalog = useConfigurationCatalog();
  return (
    <section className="space-y-6 text-sm">
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold">账户</h1>
          <p className="configuration-help">完整账户事实与成交核对。刷新只读取账户，不启用交易。</p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Link className="configuration-button" to="/accounts/connections">
            管理连接
          </Link>
          <Link className="configuration-button" to="/accounts/books">
            管理资金池
          </Link>
        </div>
      </header>
      {accounts.isPending ? <p role="status">正在读取账户账本…</p> : null}
      {accounts.isError ? (
        <p role="alert" className="configuration-error">
          账户账本读取失败，请重试。
        </p>
      ) : null}
      {accounts.data?.items.length === 0 ? <p>尚未配置账户。先添加本地模拟连接。</p> : null}
      <div className="grid gap-6 lg:grid-cols-2">
        {(['simulated', 'real'] as const).map((scope) => (
          <section key={scope} className="configuration-section">
            <h2 className="text-base font-semibold">{scope === 'simulated' ? '模拟账户' : '真实账户'}</h2>
            <MoneyValues values={accounts.data?.[scope] ?? []} />
            <p className="configuration-help">按原币种列示，不跨币种相加</p>
            {accounts.data?.items
              .filter((item) => item.capital_scope === scope)
              .map((item) => (
                <article key={item.connection_id} className="border-t border-border py-4">
                  <div className="flex items-center justify-between gap-3">
                    <Link
                      className="font-medium text-primary underline underline-offset-4"
                      to={`/accounts/connections/${encodeURIComponent(item.connection_id)}`}
                    >
                      {item.label}
                    </Link>
                    <span>{item.enabled ? '已启用' : '已停用'}</span>
                  </div>
                  <p className="configuration-help">
                    {catalog.data?.venues.find((venue) => venue.id === item.adapter_id)?.label.zh_CN ??
                      item.adapter_id}{' '}
                    ·{' '}
                    {catalog.data?.venues
                      .find((venue) => venue.id === item.adapter_id)
                      ?.environments.find((environment) => environment.id === item.environment)?.label.zh_CN ??
                      item.environment}{' '}
                    · {item.book_ids.length ? `资金池：${item.book_ids.join('、')}` : '未分配资金池'}
                  </p>
                  <MoneyValues values={item.snapshot ? [item.snapshot.equity] : []} />
                  {item.failure_reason ? <p className="configuration-error">{item.failure_reason}</p> : null}
                </article>
              ))}
            {books.data?.[scope].books.map((book) => (
              <Link
                className="configuration-button mr-2"
                key={book.book_id}
                to={`/accounts/books/${encodeURIComponent(book.book_id)}`}
              >
                资金池 · {book.label}
              </Link>
            ))}
          </section>
        ))}
      </div>
    </section>
  );
}

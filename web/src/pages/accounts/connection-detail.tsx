import { useState } from 'react';
import { Link, useParams } from 'react-router';
import { PageHeader } from '@/components/ui/page-header';
import { Button } from '@/components/ui/button';
import { useTranslation } from 'react-i18next';
import { useAccount, useAccountHistory, useSyncAccount } from '@/hooks/use-accounts';
import { useConfigurationCatalog } from '@/hooks/use-configuration-catalog';
import { formatDateTime } from '@/lib/format';
import { ConnectionForm } from './connection-form';
import { FactReason, MoneyValues } from './money';
import { ExitDialog } from './exit-dialog';
import { useRemoveAccount } from '@/hooks/use-account-operations';
import { ConfigurationEditorGate, useConfiguration } from '@/pages/settings/configuration-context';
import { AttentionList } from '@/components/alerts/attention-list';

const tabs = ['概览', '持仓与订单', '成交与收益', '连接配置'] as const;
function fieldText(values: FormData, key: string) {
  const value = values.get(key);
  return typeof value === 'string' ? value : '';
}
export default function ConnectionDetail() {
  const { t } = useTranslation('configuration');
  const catalog = useConfigurationCatalog();
  const { connectionId = '' } = useParams();
  const account = useAccount(connectionId);
  const configuration = useConfiguration();
  const remove = useRemoveAccount(connectionId);
  const sync = useSyncAccount(connectionId);
  const [tab, setTab] = useState<(typeof tabs)[number]>('概览');
  const [filters, setFilters] = useState({ symbol: '', start: '', end: '', offset: 0 });
  const history = useAccountHistory(connectionId, filters, tab === '成交与收益');
  if (account.isPending) return <p role="status">正在读取账户账本…</p>;
  if (!account.data)
    return (
      <p role="alert">
        账户无法读取。<Link to="/accounts">返回账户列表</Link>
      </p>
    );
  const item = account.data;
  const venue = catalog.data?.venues.find((entry) => entry.id === item.adapter_id);
  const environment = venue?.environments.find((entry) => entry.id === item.environment)?.label.zh_CN;
  return (
    <section className="space-y-6 text-sm">
      <Link className="text-primary underline" to="/accounts">
        返回账户列表
      </Link>
      <PageHeader
        title={item.label}
        subtitle={
          <>
            {item.capital_scope === 'simulated' ? '模拟账户' : '真实账户'} · {venue?.label.zh_CN ?? item.adapter_id} ·{' '}
            {environment ?? item.environment} · <span>{item.enabled ? '已启用' : '已停用'}</span>
          </>
        }
        actions={
          <>
            <ExitDialog account={item} />
            {!item.archived ? (
              <div>
                <Button
                  variant="outline"
                  disabled={remove.isPending || !configuration.revision}
                  onClick={() => {
                    if (
                      window.confirm(
                        '安全移除连接前必须停用所属池（或无归属连接），并由平台最新同步证明无持仓、无挂单。历史记录仍可读取。确认移除？',
                      )
                    )
                      remove.mutate(configuration.revision!);
                  }}
                >
                  安全移除连接
                </Button>
                {!configuration.revision ? (
                  <p className="configuration-help">配置版本不可用，无法安全移除；请先解锁并重新读取配置。</p>
                ) : null}
              </div>
            ) : (
              <p role="status">已归档 · 历史只读</p>
            )}
            <Button variant="outline" disabled={sync.isPending || item.archived} onClick={() => sync.mutate()}>
              {sync.isPending ? '正在刷新…' : '刷新账户'}
            </Button>
          </>
        }
      />
      <AttentionList connectionId={connectionId} />
      {remove.isError ? (
        <p role="alert" className="configuration-error">
          无法安全移除。请先停用并刷新账户，处理剩余持仓和挂单后再试。
        </p>
      ) : null}
      <div className="rounded-lg border border-border bg-card p-4">
        <p>最近成功同步：{item.last_success_at ? formatDateTime(item.last_success_at) : '尚未同步'}</p>
        {item.last_failure_at ? (
          <p className="configuration-help">最近失败：{formatDateTime(item.last_failure_at)}</p>
        ) : null}
        {item.failure_reason ? (
          <p role="alert" className="configuration-error">
            {item.failure_reason}
          </p>
        ) : null}
        {sync.isError ? (
          <p role="alert" className="configuration-error">
            刷新未完成，保留最近成功事实。请检查连接后重试。
          </p>
        ) : null}
        <p className="configuration-help">刷新不改变连接启用、真实资金授权或人工审批。</p>
      </div>
      <div role="tablist" aria-label="账户详情" className="flex flex-wrap gap-2 border-b border-border pb-3">
        {tabs.map((name) => (
          <button
            key={name}
            role="tab"
            aria-selected={tab === name}
            aria-controls={`account-${name}`}
            id={`tab-${name}`}
            className={`configuration-button ${tab === name ? 'configuration-primary' : ''}`}
            onClick={() => setTab(name)}
          >
            {name}
          </button>
        ))}
      </div>
      <div role="tabpanel" id={`account-${tab}`} aria-labelledby={`tab-${tab}`} className="space-y-5">
        {tab === '概览' ? (
          <>
            {!item.snapshot ? (
              <p>尚无账户快照，点击“刷新账户”读取当前事实。</p>
            ) : (
              <>
                <div className="grid gap-4 md:grid-cols-3">
                  {[
                    { label: '账户权益', value: item.snapshot.equity },
                    { label: '已用保证金', value: item.snapshot.used_margin },
                    { label: '可用保证金', value: item.snapshot.available_margin },
                  ].map(({ label, value }) => (
                    <section className="configuration-section" key={label}>
                      <h2 className="font-medium">{label}</h2>
                      <MoneyValues values={[value]} />
                    </section>
                  ))}
                </div>
                <h2 className="font-medium">原币余额</h2>
                <MoneyValues values={item.snapshot.balances} />
                {item.snapshot.completeness.map((reason) => (
                  <p key={reason} className="configuration-help">
                    <FactReason reason={reason} />
                  </p>
                ))}
              </>
            )}
            <h2 className="font-medium">历史覆盖</h2>
            {Object.values(item.coverage).some(Boolean) ? (
              Object.entries(item.coverage).map(([kind, window]) =>
                window ? (
                  <p key={kind}>
                    {kind === 'fills' ? '成交' : '资金费'}：{formatDateTime(window.coverage_start)} 至{' '}
                    {formatDateTime(window.coverage_end)} ·{' '}
                    {window.from_inception ? '从本地账户起点记录' : '不代表账户全部历史'}
                  </p>
                ) : null,
              )
            ) : (
              <p className="configuration-help">尚无可核对的历史覆盖区间</p>
            )}
          </>
        ) : null}
        {tab === '持仓与订单' ? (
          <>
            <h2 className="font-medium">全部持仓</h2>
            {item.snapshot?.positions.length ? (
              item.snapshot.positions.map((position, index) => (
                <article className="configuration-section" key={`${position.instrument.venue_symbol}-${index}`}>
                  <h3>{position.instrument.venue_symbol}</h3>
                  <p>
                    净持仓 {position.signed_amount} · 可用 {position.available_amount ?? '未知'}
                  </p>
                  <MoneyValues values={[position.signed_notional, position.unrealized_pnl]} />
                  {position.instrument.reason ? <p>{t('accountFacts.unknownInstrument')}</p> : null}
                </article>
              ))
            ) : (
              <p>最近快照无持仓记录</p>
            )}
            <h2 className="font-medium">普通、保护及已结束订单</h2>
            {item.orders.length ? (
              item.orders.map((order) => (
                <article className="configuration-section" key={order.venue_order_id}>
                  <h3>
                    {order.protection ? '保护订单' : '普通订单'} · {order.instrument.venue_symbol}
                  </h3>
                  <p>
                    {order.venue_order_id} ·{' '}
                    {t(`accountFacts.orderStatus.${order.status}`, { defaultValue: t('accountFacts.unknownOrder') })} ·
                    已成交 {order.filled_amount} / {order.amount}
                  </p>
                  <p className="configuration-help">
                    最后确认：{formatDateTime(order.observed_at)}
                    {!order.currently_open && !['filled', 'canceled', 'rejected', 'closed'].includes(order.status)
                      ? ' · 已不在当前挂单中，结束状态未知'
                      : ''}
                  </p>
                  {order.attribution.decision_id ? (
                    <Link
                      className="text-primary underline"
                      to={`/decisions/${encodeURIComponent(order.attribution.decision_id)}`}
                    >
                      查看来源决策
                    </Link>
                  ) : (
                    <p>来源：{order.attribution.source === 'external' ? '外部订单' : '策略订单'}</p>
                  )}
                </article>
              ))
            ) : (
              <p>暂无已记录订单</p>
            )}
          </>
        ) : null}
        {tab === '成交与收益' ? (
          <>
            <form
              className="flex flex-wrap items-end gap-3"
              onSubmit={(event) => {
                event.preventDefault();
                const values = new FormData(event.currentTarget);
                setFilters({
                  symbol: fieldText(values, 'symbol'),
                  start: fieldText(values, 'start') ? new Date(fieldText(values, 'start')).toISOString() : '',
                  end: fieldText(values, 'end') ? new Date(fieldText(values, 'end')).toISOString() : '',
                  offset: 0,
                });
              }}
            >
              <label>
                品种
                <input className="configuration-input block" name="symbol" placeholder="全部品种" />
              </label>
              <label>
                开始时间
                <input className="configuration-input block" type="datetime-local" name="start" />
              </label>
              <label>
                结束时间
                <input className="configuration-input block" type="datetime-local" name="end" />
              </label>
              <button className="configuration-button">查询</button>
            </form>
            {history.income.isPending || history.fills.isPending ? <p role="status">正在读取成交与收益…</p> : null}
            {history.income.isError || history.fills.isError ? (
              <p role="alert">查询失败，请检查时间区间后重试。</p>
            ) : null}
            {history.income.data ? (
              <>
                <p className="configuration-help">
                  查询区间：{formatDateTime(history.income.data.start)} 至 {formatDateTime(history.income.data.end)}。
                  未指定结束时间时截至最近已存账户快照，之后的交易尚未计入。
                </p>
                <div className="grid gap-4 md:grid-cols-3">
                  {(['realized_gross', 'fees', 'funding', 'net_trading'] as const).map((key) => (
                    <section className="configuration-section" key={key}>
                      <h2 className="font-medium">
                        {
                          {
                            realized_gross: '已实现毛利',
                            fees: '手续费（正值为支出）',
                            funding: '资金费（正值为收入）',
                            net_trading: '净交易收益',
                          }[key]
                        }
                      </h2>
                      <MoneyValues values={history.income.data![key]} />
                    </section>
                  ))}
                </div>
                <section className="configuration-section">
                  <h2 className="font-medium">当前未实现盈亏</h2>
                  <p className="configuration-help">
                    账户估值时间：
                    {history.income.data.unrealized_as_of
                      ? formatDateTime(history.income.data.unrealized_as_of)
                      : '未知'}
                    。不计入所选历史区间的净交易收益。
                  </p>
                  <MoneyValues values={history.income.data.unrealized} />
                </section>
                <p className="configuration-help">{history.income.data.methodology}</p>
                {history.income.data.completeness.map((reason) => (
                  <p key={reason} className="configuration-help">
                    {reason}
                  </p>
                ))}
              </>
            ) : null}
            {history.fills.data?.items.length === 0 ? <p>运行模拟交易后可在这里核对成交</p> : null}
            {history.fills.data?.items.map((fill) => (
              <article className="configuration-section" key={fill.venue_fill_id}>
                <h3>
                  {fill.instrument.venue_symbol} · {fill.side === 'buy' ? '买入' : '卖出'}
                </h3>
                <p>
                  {fill.amount} @ {fill.price} · {formatDateTime(fill.occurred_at)}
                </p>
                <MoneyValues values={[fill.fee, fill.realized_pnl]} />
                <p>归属资金池：{fill.attribution.book_id ?? '未知'}</p>
                {fill.attribution.decision_id ? (
                  <Link
                    className="text-primary underline"
                    to={`/decisions/${encodeURIComponent(fill.attribution.decision_id)}`}
                  >
                    查看来源决策
                  </Link>
                ) : (
                  <p>外部或无来源决策记录</p>
                )}
              </article>
            ))}
            <div className="flex gap-3">
              <button
                className="configuration-button"
                disabled={filters.offset === 0}
                onClick={() => setFilters((current) => ({ ...current, offset: Math.max(0, current.offset - 50) }))}
              >
                上一页
              </button>
              <button
                className="configuration-button"
                disabled={!history.fills.data || filters.offset + 50 >= history.fills.data.total}
                onClick={() => setFilters((current) => ({ ...current, offset: current.offset + 50 }))}
              >
                下一页
              </button>
            </div>
          </>
        ) : null}
        {tab === '连接配置' ? (
          item.archived ? (
            <p>账户已归档，仅保留历史读取。</p>
          ) : (
            <ConfigurationEditorGate>
              <ConnectionForm connectionId={connectionId} onSaved={() => undefined} />
            </ConfigurationEditorGate>
          )
        ) : null}
      </div>
    </section>
  );
}

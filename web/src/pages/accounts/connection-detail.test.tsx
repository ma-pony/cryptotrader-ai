import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { expect, it } from 'vitest';
import { workflowHarness } from '@/test/configuration-workflow';
import { formatDateTime } from '@/lib/format';
import type { AccountIncome } from '@/types/api';

const unknown = { amount: null, currency: 'USDT', unavailable_reason: '缺少当前估值' };
const account = {
  connection_id: 'paper',
  label: 'Paper',
  adapter_id: 'paper',
  environment: 'paper',
  capital_scope: 'simulated',
  enabled: false,
  book_ids: [],
  snapshot: null,
  last_success_at: '2026-08-31T10:00:00Z',
  last_failure_at: '2026-08-31T11:00:00Z',
  failure_reason: '账户同步失败，请检查连接权限与历史覆盖范围',
  coverage: { fills: null, funding: null },
  orders: [
    {
      connection_id: 'paper',
      venue_order_id: 'real-order-1',
      instrument: { venue_symbol: 'BTC/USDT', pair: 'BTC/USDT', market_type: 'spot', tradable: true, reason: null },
      side: 'buy',
      order_type: 'limit',
      amount: '2',
      filled_amount: '0',
      average_price: null,
      remaining_notional: unknown,
      status: 'open',
      reduce_only: false,
      protection: false,
      client_order_id: 'real-client-1',
      observed_at: '2026-08-31T10:00:00Z',
      currently_open: false,
      attribution: { source: 'strategy', book_id: 'simulation', decision_id: 'decision-exact', operation_id: null },
    },
  ],
};
const income: AccountIncome = {
  start: '2026-08-24T10:00:00Z',
  end: '2026-08-31T10:00:00Z',
  realized_gross: [unknown],
  fees: [{ amount: '0.01', currency: 'BTC', unavailable_reason: null }],
  funding: [unknown],
  unrealized: [unknown],
  unrealized_as_of: null,
  net_trading: [unknown],
  completeness: ['成交历史未覆盖查询区间'],
  methodology: '移动平均成本，净交易收益不含未实现盈亏',
  items: [],
  total: 0,
  offset: 0,
  limit: 50,
};

function harness(incomeResponse = income, requiredAccess?: string) {
  return workflowHarness('/accounts/connections/paper', undefined, requiredAccess, undefined, (url) => {
    if (url.includes('/api/accounts/paper/income'))
      return Promise.resolve(new Response(JSON.stringify(incomeResponse)));
    if (url.includes('/api/accounts/paper/fills'))
      return Promise.resolve(new Response(JSON.stringify({ items: [], total: 0, offset: 0, limit: 50 })));
    if (url.includes('/api/accounts/paper')) return Promise.resolve(new Response(JSON.stringify(account)));
    return undefined;
  });
}

it('keeps account facts readable while revision-dependent actions and the editor stay locked', async () => {
  harness(income, 'missing-config-access');
  expect(await screen.findByRole('heading', { name: 'Paper', level: 1 })).toBeInTheDocument();
  expect(screen.getByText(account.failure_reason)).toBeVisible();
  expect(screen.getByRole('button', { name: '安全移除连接' })).toBeDisabled();
  expect(screen.getByRole('button', { name: '人工退出' })).toBeDisabled();
  expect(screen.getAllByText(/配置版本不可用/).length).toBeGreaterThanOrEqual(2);
  await userEvent.click(screen.getByRole('tab', { name: '连接配置' }));
  expect(await screen.findByRole('heading', { name: '解锁配置中心' })).toBeInTheDocument();
});

it('shows stale success and failure independently and refreshes only on explicit click', async () => {
  const result = harness();
  expect(await screen.findByRole('heading', { name: 'Paper', level: 1 })).toBeInTheDocument();
  expect(screen.getByText('已停用')).toBeInTheDocument();
  expect(screen.getByText(account.failure_reason)).toBeInTheDocument();
  expect(screen.getByText(/最近成功同步/)).toBeInTheDocument();
  expect(result.fetchMock.mock.calls.filter(([, init]) => init?.method === 'POST')).toHaveLength(0);
  await userEvent.click(screen.getByRole('button', { name: '刷新账户' }));
  await waitFor(() =>
    expect(
      result.fetchMock.mock.calls.filter(
        ([url, init]) => url.endsWith('/api/accounts/paper/sync') && init?.method === 'POST',
      ),
    ).toHaveLength(1),
  );
  expect(
    result.fetchMock.mock.calls.filter(([url, init]) => init?.method === 'PUT' || url.includes('/trading-runs')),
  ).toHaveLength(0);
});

it('keeps unknown income and original fee currency and reuses configuration form', async () => {
  harness();
  await screen.findByRole('heading', { name: 'Paper', level: 1 });
  await userEvent.click(screen.getByRole('tab', { name: '成交与收益' }));
  expect(await screen.findByText('运行模拟交易后可在这里核对成交')).toBeInTheDocument();
  expect(screen.getByText('0.01 BTC')).toBeInTheDocument();
  expect(screen.getAllByText(/缺少当前估值/).length).toBeGreaterThan(0);
  expect(screen.getByRole('heading', { name: '当前未实现盈亏' })).toBeInTheDocument();
  expect(screen.getByText(/账户估值时间：/)).toHaveTextContent('未知');
  expect(screen.getByText(/不计入所选历史区间的净交易收益/)).toBeInTheDocument();
  await userEvent.click(screen.getByRole('tab', { name: '连接配置' }));
  expect(await screen.findByRole('button', { name: '保存并检查' })).toBeInTheDocument();
});

it('labels a newer current valuation independently of the historical query range', async () => {
  const asOf = '2026-08-31T12:00:00Z';
  const end = '2026-08-03T00:00:00Z';
  harness({
    ...income,
    start: '2026-08-01T00:00:00Z',
    end,
    unrealized_as_of: asOf,
    unrealized: [{ amount: '999', currency: 'USDT', unavailable_reason: null }],
  });
  await screen.findByRole('heading', { name: 'Paper', level: 1 });
  await userEvent.click(screen.getByRole('tab', { name: '成交与收益' }));
  expect(await screen.findByText('999 USDT')).toBeInTheDocument();
  expect(screen.getByText(/查询区间：/)).toHaveTextContent(formatDateTime(end));
  expect(screen.getByText(/账户估值时间：/)).toHaveTextContent(formatDateTime(asOf));
  expect(screen.getByText(/不计入所选历史区间的净交易收益/)).toBeInTheDocument();
});

it('renders persisted order history with Chinese state and its exact decision', async () => {
  harness();
  await screen.findByRole('heading', { name: 'Paper', level: 1 });
  await userEvent.click(screen.getByRole('tab', { name: '持仓与订单' }));
  expect(screen.getByText(/real-order-1 · 挂单中/)).toBeInTheDocument();
  expect(screen.getByText(/结束状态未知/)).toBeInTheDocument();
  expect(screen.getByRole('link', { name: '查看来源决策' })).toHaveAttribute('href', '/decisions/decision-exact');
});

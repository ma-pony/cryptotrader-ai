import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { expect, it } from 'vitest';
import { workflowHarness } from '@/test/configuration-workflow';

const result = {
  canceled_order_ids: [],
  canceled_protection_ids: [],
  orders: [],
  remaining_position: null,
  remaining_order_ids: [],
  remaining_protection_ids: [],
  observed_at: null,
  failure_reason: null,
  reconciliation_required: false,
};
function harness(failure = false) {
  let executing = false;
  return workflowHarness('/accounts/connections/paper', undefined, undefined, undefined, (url) => {
    if (url.includes('/operations/prepare'))
      return Promise.resolve(
        new Response(JSON.stringify({ operation_id: 'manual-1', status: 'preparing' }), { status: 202 }),
      );
    if (url.includes('/account-operations/manual-1/execute')) {
      executing = true;
      return Promise.resolve(
        new Response(JSON.stringify({ operation_id: 'manual-1', status: 'executing' }), { status: 202 }),
      );
    }
    if (url.includes('/account-operations/manual-1'))
      return Promise.resolve(
        new Response(
          JSON.stringify({
            operation_id: 'manual-1',
            connection_id: 'paper',
            pair: 'BTC/USDT',
            kind: 'flatten',
            status: executing ? (failure ? 'failed' : 'completed') : 'awaiting_confirmation',
            created_at: '2026-08-31T10:00:00Z',
            updated_at: '2026-08-31T10:00:00Z',
            plan: {
              operation_id: 'manual-1',
              version: 1,
              connection_id: 'paper',
              book_id: null,
              capital_scope: 'simulated',
              pair: 'BTC/USDT',
              kind: 'flatten',
              stopped_scope: ['connection:paper'],
              ordinary_order_ids: ['ordinary-1'],
              position_amount: '2',
              close_amount: '2',
              protection_ids: ['protection-1'],
              snapshot_time: '2026-08-31T10:00:00Z',
            },
            result: executing
              ? {
                  ...result,
                  remaining_position: failure ? '1' : '0',
                  failure_reason: failure ? '退出未完成，保护已保留' : null,
                }
              : result,
          }),
        ),
      );
    if (url.endsWith('/api/accounts/paper'))
      return Promise.resolve(
        new Response(
          JSON.stringify({
            connection_id: 'paper',
            label: 'Paper',
            adapter_id: 'paper',
            environment: 'paper',
            capital_scope: 'simulated',
            enabled: true,
            book_ids: [],
            snapshot: null,
            last_success_at: null,
            last_failure_at: null,
            failure_reason: null,
            coverage: { fills: null, funding: null },
            orders: [],
          }),
        ),
      );
    return undefined;
  });
}

it('requires scope and frozen-plan confirmations before executing, with keyboard focus', async () => {
  const h = harness();
  await screen.findByRole('heading', { name: 'Paper', level: 1 });
  await userEvent.click(screen.getByRole('button', { name: '人工退出' }));
  const dialog = screen.getByRole('dialog');
  expect(dialog).toContainElement(document.activeElement as HTMLElement);
  expect(within(dialog).getByText(/其他资金池不受影响/)).toBeInTheDocument();
  expect(h.fetchMock.mock.calls.filter(([, init]) => init?.method === 'POST')).toHaveLength(0);
  await userEvent.click(within(dialog).getByRole('button', { name: '确认停用并读取计划' }));
  await screen.findByText(/预计平仓数量/);
  expect(h.fetchMock.mock.calls.filter(([url]) => url.endsWith('/execute'))).toHaveLength(0);
  await userEvent.click(screen.getByRole('button', { name: '确认执行此计划' }));
  expect(await screen.findByText('退出已完成，账户保持停用')).toBeInTheDocument();
  await waitFor(() => expect(h.fetchMock.mock.calls.filter(([url]) => url.endsWith('/execute'))).toHaveLength(1));
  expect(h.fetchMock.mock.calls.some(([url]) => url.includes('/trading-runs') && url.includes('start'))).toBe(false);
});

it('shows remaining facts on failure without offering automatic resubmission', async () => {
  harness(true);
  await screen.findByRole('heading', { name: 'Paper', level: 1 });
  await userEvent.click(screen.getByRole('button', { name: '人工退出' }));
  await userEvent.click(screen.getByRole('button', { name: '确认停用并读取计划' }));
  await screen.findByRole('button', { name: '确认执行此计划' });
  await userEvent.click(screen.getByRole('button', { name: '确认执行此计划' }));
  expect(await screen.findByText('退出未完成，保护已保留')).toBeInTheDocument();
  expect(screen.getByText(/剩余持仓：1/)).toBeInTheDocument();
  expect(screen.getByRole('button', { name: '重新读取状态' })).toBeInTheDocument();
  expect(screen.queryByRole('button', { name: '确认执行此计划' })).not.toBeInTheDocument();
});

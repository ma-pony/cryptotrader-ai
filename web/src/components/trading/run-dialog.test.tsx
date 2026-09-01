import { screen, within, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { expect, it } from 'vitest';
import { workflowHarness } from '@/test/configuration-workflow';

it('offers an explicit trading scope confirmation from the engine', async () => {
  const h = workflowHarness('/engine');
  const base = h.fetchMock.getMockImplementation()!;
  h.fetchMock.mockImplementation((url, init) =>
    url.includes('/api/trading-runs/scope')
      ? Promise.resolve(
          new Response(
            JSON.stringify({
              pair: 'BTC/USDT',
              saved_revision: 1,
              ready: true,
              reasons: [],
              books: [
                {
                  book_id: 'sim',
                  label: '模拟主账户',
                  capital_scope: 'simulated',
                  enabled: true,
                  eligible: true,
                  reasons: [],
                  hitl_required: true,
                  connections: [{ connection_id: 'paper', label: 'Paper', environment: 'paper', enabled: true }],
                },
                {
                  book_id: 'sim-two',
                  label: '模拟第二池',
                  capital_scope: 'simulated',
                  enabled: true,
                  eligible: true,
                  reasons: [],
                  hitl_required: false,
                  connections: [
                    { connection_id: 'paper-two', label: 'Paper Two', environment: 'lab-v2', enabled: true },
                  ],
                },
                {
                  book_id: 'real',
                  label: '真实账户',
                  capital_scope: 'real',
                  enabled: true,
                  eligible: false,
                  reasons: [
                    {
                      code: 'real_authorization_missing',
                      message: '尚未授权真实资金下单。',
                      path: 'execution.live_order_execution_enabled',
                    },
                  ],
                  hitl_required: false,
                  connections: [],
                },
              ],
            }),
          ),
        )
      : url.endsWith('/api/trading-runs') && init?.method === 'POST'
        ? Promise.resolve(new Response(JSON.stringify({ detail: '配置已变化' }), { status: 409 }))
        : base(url, init),
  );
  expect(await screen.findByRole('heading', { name: '引擎' })).toBeInTheDocument();
  expect(h.fetchMock.mock.calls.some(([url]) => url.endsWith('/api/runtime/status'))).toBe(true);
  await userEvent.click(screen.getByRole('button', { name: '发起交易' }));
  const dialogElement = screen.getByRole('dialog', { name: '确认交易范围' });
  const dialog = within(dialogElement);
  expect(dialogElement).toHaveClass('w-[calc(100%-2rem)]', 'overscroll-contain');
  expect(dialog.getByRole('button', { name: '关闭' })).toHaveClass('min-h-10', 'min-w-10');
  const confirm = await screen.findByRole('button', { name: '确认全部范围并发起交易' });
  expect(confirm).toBeDisabled();
  expect(screen.getByText('本次跳过')).toBeInTheDocument();
  expect(screen.getByText('尚未授权真实资金下单。')).toBeInTheDocument();
  await userEvent.click(dialog.getByRole('checkbox', { name: /模拟主账户/ }));
  expect(confirm).toBeDisabled();
  await userEvent.click(dialog.getByRole('checkbox', { name: /模拟第二池/ }));
  expect(confirm).toBeEnabled();
  await userEvent.click(confirm);
  expect(await dialog.findByRole('alert')).toHaveTextContent('交易未启动');
  await waitFor(() => expect(confirm).toBeDisabled());
  const call = h.fetchMock.mock.calls.find(
    ([url, init]) => url.endsWith('/api/trading-runs') && init?.method === 'POST',
  );
  expect(JSON.parse(call![1]!.body as string)).toEqual({
    pair: 'BTC/USDT',
    expected_revision: 1,
    confirmed_book_ids: ['sim', 'sim-two'],
  });
  expect(h.saved().document.execution.live_order_execution_enabled).toBe(false);
});

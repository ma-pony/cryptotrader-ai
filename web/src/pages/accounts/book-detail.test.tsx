import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { expect, it, vi } from 'vitest';
import { workflowHarness } from '@/test/configuration-workflow';

function harness(riskState: Record<string, unknown> | null = null, requiredAccess?: string) {
  vi.spyOn(window, 'confirm').mockReturnValue(true);
  return workflowHarness('/accounts/books/sim', undefined, requiredAccess, undefined, (url) => {
    if (url.includes('/api/portfolio/books/sim'))
      return Promise.resolve(
        new Response(
          JSON.stringify({
            book_id: 'sim',
            label: '模拟主账户',
            capital_scope: 'simulated',
            enabled: true,
            total_equity: [],
            total_signed_notional: [],
            connections: [],
            risk_state: riskState,
          }),
        ),
      );
    return undefined;
  });
}

it('keeps saved book facts visible while its configuration editor is locked', async () => {
  harness(null, 'missing-config-access');
  expect(await screen.findByRole('heading', { name: '模拟主账户', level: 1 })).toBeInTheDocument();
  expect(screen.getByRole('heading', { name: '整池风险' })).toBeVisible();
  expect(await screen.findByRole('heading', { name: '解锁配置中心' })).toBeInTheDocument();
});

it('keeps a removed book draft visible until the user saves its deletion', async () => {
  const result = harness();
  await screen.findByRole('heading', { name: '模拟主账户', level: 1 });
  await userEvent.click(screen.getByRole('button', { name: '移除' }));
  expect(screen.getByText(/待删除/)).toBeInTheDocument();
  expect(result.writes).toHaveLength(0);
  await userEvent.click(screen.getByRole('button', { name: '保存配置' }));
  await waitFor(() => expect(result.saved().document.execution.books).toEqual([]));
  expect(result.writes).toHaveLength(1);
});

it('shows persisted whole-pool risk with its currency, observation time and pending occupancy', async () => {
  harness({
    book_id: 'sim',
    capital_scope: 'simulated',
    valuation_currency: 'USDT',
    observed_at: '2026-08-31T10:00:00Z',
    equity: '80',
    peak_equity: '100',
    positions_by_instrument: [{ instrument: 'ETH/USDT:USDT', signed_notional: '70' }],
    pending_increase_notional: '5',
    gross_notional: '70',
    net_notional: '70',
    used_margin: '12',
    available_margin: '68',
    completeness: [],
  });
  await screen.findByRole('heading', { name: '整池风险' });
  expect(screen.getByText('100 USDT')).toBeInTheDocument();
  expect(screen.getByText('5 USDT')).toBeInTheDocument();
  expect(screen.getByText('12 USDT')).toBeInTheDocument();
  expect(screen.getByText('2026-08-31T10:00:00Z')).toBeInTheDocument();
});

it('can discard removal from the same detail page without sending a write', async () => {
  const result = harness();
  await screen.findByRole('heading', { name: '模拟主账户', level: 1 });
  await userEvent.click(screen.getByRole('button', { name: '移除' }));
  expect(screen.getByText(/待删除/)).toBeInTheDocument();
  await userEvent.click(screen.getByRole('button', { name: '放弃修改' }));
  expect(screen.getByRole('button', { name: '移除' })).toBeInTheDocument();
  expect(screen.queryByText(/待删除/)).not.toBeInTheDocument();
  expect(result.writes).toHaveLength(0);
  expect(result.saved().document.execution.books.map((book) => book.id)).toEqual(['sim']);
});

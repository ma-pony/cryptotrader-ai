// Test HTTP boundary doubles resolve immediately; no external fetch.
/* eslint-disable @typescript-eslint/require-await */
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { fireEvent, render, screen } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router';
import { expect, it, vi } from 'vitest';
import '@/lib/i18n';
import { researchRun } from '@/test/research-fixture';
import BacktestDetail from './backtest-detail';

vi.mock('@/components/charts/equity-chart', () => ({ EquityChart: () => <div aria-label="历史权益图" /> }));

function harness(status = 'completed', empty = false, legacy = false) {
  let run = researchRun({ status: status as ReturnType<typeof researchRun>['status'] });
  if (legacy) {
    run.config_snapshot = null;
    run.incomplete_fields = ['config_snapshot: 旧文件没有完整安全配置，不能复用'];
  }
  if (status !== 'completed') run.result = null;
  if (empty && run.result)
    run.result = {
      ...run.result,
      metrics: {
        total_return_pct: 0,
        sharpe: 0,
        max_drawdown_pct: 0,
        win_rate: null,
        fill_count: 0,
        closed_trade_count: 0,
      },
      fills: [],
      closed_trades: [],
      fees: '0',
      equity_curve: [],
    };
  const requests: string[] = [];
  vi.stubGlobal(
    'fetch',
    vi.fn(async (_url: string, init?: RequestInit) => {
      requests.push(init?.method ?? 'GET');
      if (init?.method === 'DELETE') {
        run = { ...run, status: 'canceled' };
        return new Response(JSON.stringify({ canceled: true }));
      }
      return new Response(JSON.stringify(run));
    }),
  );
  const mount = () =>
    render(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <MemoryRouter initialEntries={['/research/backtests/run-first']}>
          <Routes>
            <Route path="/research/backtests/:runId" element={<BacktestDetail />} />
          </Routes>
        </MemoryRouter>
      </QueryClientProvider>,
    );
  return { mount, requests };
}

it('reopens saved history with original decision, sample state, model identity and historical equity', async () => {
  const h = harness();
  const first = h.mount();
  expect(await screen.findByRole('link', { name: /查看原决策/ })).toHaveAttribute(
    'href',
    '/decisions/decision-original',
  );
  expect(screen.getByText('100.0%')).toBeInTheDocument();
  expect(screen.getByText(/1009.79/)).toBeInTheDocument();
  expect(screen.getByText('夏普比率')).toBeInTheDocument();
  expect(screen.getByText(/实际模型未知/)).toBeInTheDocument();
  first.unmount();
  h.mount();
  await screen.findByRole('link', { name: /查看原决策/ });
  expect(h.requests.every((method) => method === 'GET')).toBe(true);
});

it('does not turn an empty closed-trade sample into zero percent wins', async () => {
  harness('completed', true).mount();
  expect(await screen.findByText('尚无平仓样本')).toBeInTheDocument();
});

it('keeps an incomplete imported record readable but prevents snapshot reuse', async () => {
  harness('completed', true, true).mount();
  expect(await screen.findByRole('button', { name: '配置不完整，无法复用' })).toBeDisabled();
  expect(screen.getByText(/旧文件没有完整安全配置/)).toBeInTheDocument();
  expect(screen.queryByRole('link', { name: '复用此配置' })).not.toBeInTheDocument();
});

it('cancels an active run and keeps the terminal detail available', async () => {
  harness('running').mount();
  fireEvent.click(await screen.findByRole('button', { name: '取消回测' }));
  expect(await screen.findByText('已取消')).toBeInTheDocument();
  expect(screen.queryByRole('button', { name: '取消回测' })).not.toBeInTheDocument();
});

it.each(['interrupted', 'failed'])('shows %s with a route to the next experiment', async (status) => {
  harness(status).mount();
  expect(await screen.findByRole('link', { name: '复用此配置' })).toHaveAttribute('href', '/research?reuse=run-first');
  expect(screen.getByText(status === 'interrupted' ? '运行已中断' : '运行失败')).toBeInTheDocument();
  expect(screen.getByText(/实际模型未知/)).toBeInTheDocument();
});

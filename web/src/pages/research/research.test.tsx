// Test HTTP boundary doubles resolve immediately; no external fetch.
/* eslint-disable @typescript-eslint/require-await */
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router';
import { expect, it, vi } from 'vitest';
import '@/lib/i18n';
import { runtimeConfigFixture } from '@/test/runtime-config-fixture';
import { researchRun } from '@/test/research-fixture';
import ResearchPage from './index';

function harness(fail = false, populated = false) {
  const writes: Record<string, unknown>[] = [];
  const run = researchRun();
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string, init?: RequestInit) => {
      if (init?.method === 'POST') {
        writes.push(JSON.parse(init.body as string) as Record<string, unknown>);
        return new Response(JSON.stringify(fail ? { detail: 'failure' } : { run_id: 'run-new', status: 'queued' }), {
          status: fail ? 503 : 202,
        });
      }
      if (url.includes('/api/config')) return new Response(JSON.stringify(runtimeConfigFixture()));
      if (url.includes('/runs?'))
        return new Response(JSON.stringify({ items: populated ? [run] : [], limit: 20, offset: 0, has_next: false }));
      return new Response(JSON.stringify(run));
    }),
  );
  render(
    <QueryClientProvider
      client={new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })}
    >
      <MemoryRouter initialEntries={['/research']}>
        <Routes>
          <Route path="/research" element={<ResearchPage />} />
          <Route path="/research/backtests/run-new" element={<p>新运行详情</p>} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
  return writes;
}

it('explains missing inputs and fees before starting, converts percent to fee rate and keeps failed input', async () => {
  const writes = harness(true);
  await waitFor(() => expect(screen.getByRole('button', { name: '运行回测' })).toBeEnabled());
  expect(screen.getByText(/模型费用：/)).toBeInTheDocument();
  expect(screen.getByText(/缺项：/)).toBeInTheDocument();
  fireEvent.change(screen.getByLabelText('起始日期'), { target: { value: '2024-01-01' } });
  fireEvent.change(screen.getByLabelText('结束日期'), { target: { value: '2024-01-02' } });
  fireEvent.change(screen.getByLabelText('手续费（%）'), { target: { value: '0.1' } });
  fireEvent.change(screen.getByLabelText('滑点（bps）'), { target: { value: '5' } });
  fireEvent.click(screen.getByRole('button', { name: '运行回测' }));
  expect(await screen.findByText(/回测启动失败/)).toBeInTheDocument();
  expect(writes).toHaveLength(1);
  expect(writes[0]).toMatchObject({
    fee_rate: '0.001',
    slippage_bps: '5',
    initial_equity: '10000',
    snapshot_run_id: null,
  });
  expect(screen.getByLabelText('起始日期')).toHaveValue('2024-01-01');
});

it('reuses a historical snapshot via GET and starts only after explicit submit', async () => {
  const writes = harness(false, true);
  await screen.findByRole('option', { name: /run-first/ });
  fireEvent.change(screen.getByLabelText('配置来源与历史复用'), { target: { value: 'run-first' } });
  await waitFor(() => expect(screen.getByLabelText('初始资金（USDT）')).toHaveValue(1000));
  expect(writes).toHaveLength(0);
  fireEvent.click(screen.getByRole('button', { name: '运行回测' }));
  expect(await screen.findByText('新运行详情')).toBeInTheDocument();
  expect(writes[0]).toMatchObject({ snapshot_run_id: 'run-first', fee_rate: '0.001' });
});

it('offers a usable first-run state and preserves the market and analysis entrances', async () => {
  const writes = harness();
  expect(await screen.findByText(/还没有回测记录/)).toBeInTheDocument();
  expect(screen.getByRole('link', { name: '市场观察' })).toHaveAttribute('href', '/research/market');
  expect(screen.getByRole('link', { name: '仅分析' })).toHaveAttribute('href', '/research/analysis');
  expect(writes).toHaveLength(0);
});

// Test HTTP boundary doubles resolve immediately; no external fetch.
/* eslint-disable @typescript-eslint/require-await */
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router';
import { expect, it, vi } from 'vitest';
import { researchRun } from '@/test/research-fixture';
import BacktestCompare from './backtest-compare';
import '@/lib/i18n';

it('leads with fee and context differences and never ranks incompatible experiments', async () => {
  const left = researchRun();
  const right = researchRun({ run_id: 'run-right', params: { ...left.params, fee_rate: '0.002' } });
  vi.stubGlobal(
    'fetch',
    vi.fn(
      async () =>
        new Response(
          JSON.stringify({
            comparable: false,
            condition_differences: { fee_rate: { left: '0.001', right: '0.002' } },
            configuration_differences: {
              data_coverage: { left: { historical_news: 'unavailable' }, right: { historical_news: 'available' } },
            },
            left,
            right,
          }),
        ),
    ),
  );
  render(
    <QueryClientProvider client={new QueryClient()}>
      <MemoryRouter initialEntries={['/research/compare?left=run-first&right=run-right']}>
        <BacktestCompare />
      </MemoryRouter>
    </QueryClientProvider>,
  );
  expect(await screen.findByText('实验条件不同，不作排名')).toBeInTheDocument();
  expect(screen.getByRole('heading', { name: '手续费率' })).toBeInTheDocument();
  expect(screen.getByText('数据覆盖')).toBeInTheDocument();
  expect(screen.getAllByText('100.0%')).toHaveLength(2);
  expect(screen.queryByRole('heading', { name: /最佳|胜出|排名第/ })).not.toBeInTheDocument();
});

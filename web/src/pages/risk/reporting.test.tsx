import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router';
import { beforeEach, expect, it, vi } from 'vitest';
import i18n from '@/lib/i18n';
import RiskPage from './index';

beforeEach(() => i18n.changeLanguage('zh-CN'));
it('shows four real limits and reporting metrics without invented enforcement claims', async () => {
  vi.stubGlobal(
    'fetch',
    vi.fn((url: string) =>
      Promise.resolve(
        new Response(
          JSON.stringify(
            url.endsWith('/status')
              ? {
                  trade_count_hour: 7,
                  trade_count_day: 13,
                  redis_available: true,
                  circuit_breaker: { state: 'inactive' },
                  thresholds: {
                    max_single_pct: 0.45,
                    max_total_exposure_pct: 0.8,
                    max_margin_used_pct: 0.35,
                    max_drawdown_pct: 0.12,
                  },
                  daily_loss_pct: 3,
                  drawdown_pct: 4,
                  total_exposure_pct: 50,
                  cvar_95: 6,
                  correlation_groups: [{ name: 'BTC-correlated', open: 3, pairs: ['BTC/USDT'] }],
                  cooldowns: [],
                  recent_blocks: [],
                }
              : [],
          ),
        ),
      ),
    ),
  );
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <RiskPage />
      </MemoryRouter>
    </QueryClientProvider>,
  );
  expect(await screen.findByText('45%')).toBeInTheDocument();
  for (const value of ['80%', '35%', '12%']) expect(screen.getByText(value)).toBeInTheDocument();
  expect(screen.getByRole('link', { name: '修改风控配置' })).toHaveAttribute('href', '/settings/risk');
  expect(screen.getByText('连接集中度上限')).toBeInTheDocument();
  expect(screen.getAllByText('仅供观察').length).toBeGreaterThan(0);
  expect(screen.queryByRole('meter')).not.toBeInTheDocument();
  expect(screen.queryByText(/项检查在线|每组最多|所有闸门开放|所有交易对均可交易/)).not.toBeInTheDocument();
  expect(screen.getByText('BTC-correlated')).toBeInTheDocument();
});

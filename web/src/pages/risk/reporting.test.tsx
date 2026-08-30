import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { fireEvent, render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router';
import { beforeEach, expect, it, vi } from 'vitest';
import i18n from '@/lib/i18n';
import RiskPage from './index';

beforeEach(() => i18n.changeLanguage('zh-CN'));
function renderRisk(state: 'active' | 'inactive' = 'inactive') {
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
                  circuit_breaker: { state },
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
}
it('shows four real limits and reporting metrics without invented enforcement claims', async () => {
  renderRisk();
  expect(await screen.findByText('45%')).toBeInTheDocument();
  for (const value of ['80%', '35%', '12%']) expect(screen.getByText(value)).toBeInTheDocument();
  expect(screen.getByRole('link', { name: '修改风控配置' })).toHaveAttribute('href', '/settings/risk');
  expect(screen.getByText('连接集中度上限')).toBeInTheDocument();
  expect(screen.getAllByText('仅供观察').length).toBeGreaterThan(0);
  expect(screen.queryByRole('meter')).not.toBeInTheDocument();
  expect(screen.queryByText(/项检查在线|每组最多|所有闸门开放|所有交易对均可交易/)).not.toBeInTheDocument();
  expect(screen.getByText('BTC-correlated')).toBeInTheDocument();
});

it.each([
  ['zh-CN', '熔断触发记录', '重置断路器'],
  ['en-US', 'Circuit breaker trigger recorded', 'Reset Circuit Breaker'],
])(
  'reports an active breaker record without claiming trading is paused in %s',
  async (locale, stateLabel, resetLabel) => {
    await i18n.changeLanguage(locale);
    renderRisk('active');
    expect(await screen.findByRole('button', { name: resetLabel })).toBeInTheDocument();
    expect(screen.queryByText(/交易暂停|暂停交易|trading paused|trading stopped/i)).not.toBeInTheDocument();
    expect(screen.getByText(stateLabel)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: resetLabel }));
    expect(screen.getByText(/清除此断路器记录|clears this circuit breaker record only/)).toBeInTheDocument();
    expect(screen.queryByText(/恢复交易|resume trading|trading will resume/i)).not.toBeInTheDocument();
  },
);

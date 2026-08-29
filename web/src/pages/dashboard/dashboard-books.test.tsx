import { render, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import i18n from '@/lib/i18n';
import DashboardPage from '@/pages/dashboard';
import { PortfolioBooksSchema } from '@/types/api.schema';

const portfolio = PortfolioBooksSchema.parse({
  pair: 'BTC/USDT',
  simulated: {
    totals: { equity: '100000', signed_notional: '1000' },
    books: [
      {
        book_id: 'sim-book',
        capital_scope: 'simulated',
        pair: 'BTC/USDT',
        total_equity: '100000',
        total_signed_notional: '1000',
        connections: [
          {
            connection_id: 'sim-venue',
            equity: '100000',
            balances: [{ asset: 'USDT', amount: '90000' }],
            position: { pair: 'BTC/USDT', signed_amount: '1', signed_notional: '1000', entry_price: '99000' },
          },
        ],
      },
    ],
  },
  real: {
    totals: { equity: '12000', signed_notional: '200' },
    books: [
      {
        book_id: 'real-book',
        capital_scope: 'real',
        pair: 'BTC/USDT',
        total_equity: '12000',
        total_signed_notional: '200',
        connections: [
          {
            connection_id: 'real-venue',
            equity: '12000',
            balances: [{ asset: 'USDT', amount: '10000' }],
            position: { pair: 'BTC/USDT', signed_amount: '0.2', signed_notional: '200', entry_price: '100000' },
          },
        ],
      },
    ],
  },
});
vi.mock('@/hooks/use-portfolio-books', () => ({
  usePortfolioBooks: () => ({ data: portfolio, isLoading: false, isError: false }),
}));
vi.mock('@/hooks/use-scheduler-status', () => ({
  useSchedulerStatus: () => ({ data: undefined, isLoading: false, isError: true }),
}));

describe('dashboard execution books', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('en-US');
  });
  it('keeps simulated and real scope totals isolated', () => {
    render(<DashboardPage />);
    for (const text of [
      'Simulated capital',
      'Real capital',
      '100000',
      '12000',
      'sim-book',
      'real-book',
      'sim-venue',
      'real-venue',
      'USDT: 90000',
      'Entry',
    ])
      expect(screen.getAllByText(text, { exact: false }).length).toBeGreaterThan(0);
    expect(screen.queryByText('112000')).not.toBeInTheDocument();
  });
});

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it } from 'vitest';

import i18n from '@/lib/i18n';
import type { ApprovalRequest } from '@/types/api';

import { ApprovalItem } from './approval-item';

const approval = {
  approval_id: 'approval-1',
  cycle_id: 'cycle-1',
  book_id: 'book-real', pair: 'ETH/USDT', config_revision: 8,
  proposal: { version: 1, book_id: 'book-real', capital_scope: 'real', config_revision: 8, pair: { symbol: 'ETH/USDT' }, requested_target_exposure: '0.4', target_exposure: '0.4', risk: { passed: true, requested_target_exposure: '0.4', capped_target_exposure: '0.4', connection_weights: ['1'], connection_targets: [], rejected_by: '', reason: '', cap_source: '' }, connection_risks: [], connection_plans: [{ book_id: 'book-real', connection_id: 'OKX Live', pair: { symbol: 'ETH/USDT' }, current_signed_notional: '0', target_signed_notional: '12000', delta_signed_notional: '12000', current_signed_amount: '0', target_signed_amount: '3', delta_signed_amount: '3', post_fill_signed_amount: '3', quote: { pair: { symbol: 'ETH/USDT' }, bid: '4000', ask: '4001', last: '4000' }, execution_price: '4001', amount: '3', side: 'buy', reduce_only: false, market_type: 'swap', stop_loss: null, take_profit: null, old_protection_ids: [], capabilities: { market_types: ['swap'], native_protection: true, hedge_mode: false, reduce_only: true, supported_order_types: ['market'] } }], unavailable_connections: [], errors: [], ready: true },
  status: 'pending',
  created_at: '2026-08-28T00:00:00Z',
  decided_at: null,
} as unknown as ApprovalRequest;

describe('target plan approval', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('zh-CN');
  });

  it('renders one immutable execution-book proposal and its connection target', () => {
    const client = new QueryClient({ defaultOptions: { mutations: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <ApprovalItem approval={approval} />
      </QueryClientProvider>,
    );

    expect(screen.getByText('book-real · ETH/USDT')).toBeInTheDocument();
    expect(screen.getByText(/配置版本 8/)).toBeInTheDocument();
    expect(screen.getByText('OKX Live · ETH/USDT')).toBeInTheDocument();
    expect(screen.queryByRole('spinbutton')).not.toBeInTheDocument();
  });
});

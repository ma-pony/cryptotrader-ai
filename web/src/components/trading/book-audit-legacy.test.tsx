import { render, screen } from '@testing-library/react';
import { expect, it } from 'vitest';

import { BookAudit } from './book-audit';

it('keeps unknown legacy execution facts visibly unsafe', () => {
  render(
    <BookAudit
      book={
        {
          book_id: 'legacy-book', capital_scope: 'real', config_revision: 1, pair: 'BTC/USDT:USDT',
          market_type: 'swap', status: 'partial',
          hitl: { approval_id: null, status: 'not_required', config_revision: 1 }, failure: null,
          requested_target_exposure: null, target_exposure: null, risk: null, ready: null, errors: [], execution: null,
          portfolio_before: null, portfolio_after: null, portfolio_after_available: null, reconciliation_required: null,
          connections: [{
            connection_id: 'legacy-connection', portfolio_before: null, portfolio_after: null, risk: null,
            unavailable: false,
            plan: {
              book_id: 'legacy-book', connection_id: 'legacy-connection', pair: { symbol: 'BTC/USDT:USDT' },
              current_signed_notional: '0', target_signed_notional: '100', delta_signed_notional: '100',
              current_signed_amount: '0', target_signed_amount: '1', delta_signed_amount: '1', post_fill_signed_amount: '1',
              quote: { pair: { symbol: 'BTC/USDT:USDT' }, bid: '99', ask: '101', last: '100' },
              execution_price: '101', amount: '1', side: 'buy', reduce_only: false, market_type: 'swap',
              stop_loss: null, take_profit: null, old_protection_ids: [],
              capabilities: {
                market_types: ['swap'], native_protection: false, hedge_mode: false, reduce_only: true,
                supported_order_types: ['market'], account_reads: [], exit_operations: [], history_initial_days: null,
                unknown_fields: ['account_reads', 'exit_operations', 'history_initial_days'],
              },
            },
            execution: {
              book_id: 'legacy-book', connection_id: 'legacy-connection', pair: { symbol: 'BTC/USDT:USDT' },
              target_signed_notional: '100', target_signed_amount: '1', status: 'completed', orders: [], protection: null,
              compensation: { attempted: false, succeeded: false, order: null, operation: '', safe_signed_amount: null,
                required_protection: null },
              final_position: null, error_operation: '', requires_attention: true, trace: [], execution_quote: null,
              quantity_frozen: null,
            },
          }],
        } as never
      }
    />,
  );
  expect(screen.getByText('历史对账状态未知，请勿重复下单')).toBeInTheDocument();
  expect(screen.getByText('历史能力证据不完整')).toBeInTheDocument();
  expect(screen.getByText('历史数量冻结状态未知')).toBeInTheDocument();
});

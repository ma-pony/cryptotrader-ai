import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router';
import { afterEach, describe, expect, it, vi } from 'vitest';

import i18n from '@/lib/i18n';
import { formatCycleStatus } from '@/lib/cycle-status';
import CycleDetailPage from '@/pages/cycles/cycle-detail';
import CyclesPage from '@/pages/cycles';
import { CycleSchema, type JsonValueOut } from '@/types/api.schema';

const cycleQuery = vi.fn();
const cyclesQuery = vi.fn();

vi.mock('@/hooks/use-multi-venue-cycles', () => ({
  useMultiVenueCycle: () => cycleQuery() as never,
  useMultiVenueCycles: () => cyclesQuery() as never,
}));

const jsonString = (string_value: string): JsonValueOut => ({
  kind: 'string',
  boolean_value: null,
  number_value: null,
  string_value,
  datetime_value: null,
  pair_value: null,
  items: [],
  entries: [],
});
const jsonNumber = (number_value: string): JsonValueOut => ({
  kind: 'number',
  boolean_value: null,
  number_value,
  string_value: null,
  datetime_value: null,
  pair_value: null,
  items: [],
  entries: [],
});
const jsonArray = (items: JsonValueOut[]): JsonValueOut => ({
  kind: 'array',
  boolean_value: null,
  number_value: null,
  string_value: null,
  datetime_value: null,
  pair_value: null,
  items,
  entries: [],
});
const jsonObject = (entries: Array<{ key: string; value: JsonValueOut }>): JsonValueOut => ({
  kind: 'object',
  boolean_value: null,
  number_value: null,
  string_value: null,
  datetime_value: null,
  pair_value: null,
  items: [],
  entries,
});
const quote = { pair: { symbol: 'BTC/USDT' }, bid: '99', ask: '101', last: '100' };
const position = { pair: 'BTC/USDT', signed_amount: '1', signed_notional: '100', entry_price: '98' };
const plan = {
  book_id: 'real-book',
  connection_id: 'venue-down',
  pair: { symbol: 'BTC/USDT' },
  current_signed_notional: '0',
  target_signed_notional: '100',
  delta_signed_notional: '100',
  current_signed_amount: '0',
  target_signed_amount: '1',
  delta_signed_amount: '1',
  post_fill_signed_amount: '1',
  quote,
  execution_price: '101',
  amount: '1',
  side: 'buy',
  reduce_only: false,
  market_type: 'spot',
  stop_loss: '90',
  take_profit: '120',
  old_protection_ids: ['old-protection'],
  capabilities: {
    market_types: ['spot'],
    native_protection: true,
    hedge_mode: false,
    reduce_only: true,
    supported_order_types: ['market'],
  },
};
const order = {
  id: 'order-marker',
  pair: { symbol: 'BTC/USDT' },
  side: 'order-side-marker',
  order_type: 'order-type-marker',
  amount: 'order-amount-marker',
  filled_amount: 'order-filled-marker',
  average_price: 'order-price-marker',
  status: 'filled',
  reduce_only: false,
};
const protection = {
  protection_ids: ['protection-marker'],
  pair: { symbol: 'BTC/USDT' },
  position_side: 'protection-side-marker',
  amount: 'protection-amount-marker',
  stop_loss: 'protection-stop-marker',
  take_profit: 'protection-take-marker',
  active: true,
  triggered: false,
};
const connectionPortfolio = {
  connection_id: 'venue-down',
  equity: '1000',
  balances: [{ asset: 'USDT', amount: '900' }],
  position,
};

const partialCycle = CycleSchema.parse({
  cycle_id: 'cycle-partial',
  config_revision: 7,
  market_data_source_id: 'kraken-feed',
  cycle_status: 'partial',
  execution_status: 'partial',
  requires_attention: true,
  created_at: '2026-08-29T00:00:00Z',
  shared_signals: {
    components: [
      {
        component_id: 'committee-1',
        direction: 'long',
        confidence: 0.91,
        reasoning: 'committee reasoning',
        details: [
          {
            key: 'plugin-evidence',
            value: jsonObject([
              { key: 'nested-marker', value: jsonString('nested plugin value') },
              { key: 'confidence-marker', value: jsonNumber('0.73') },
              { key: 'array-marker', value: jsonArray([jsonString('nested array value')]) },
            ]),
          },
        ],
      },
    ],
    fused: {
      score: 0.8,
      reasoning: 'fused reasoning',
      contributions: [{ component_id: 'committee-1', weight: 1, signed_score: 0.8, weighted_score: 0.8 }],
    },
    target_position: { side: 'long', size_ratio: 0.5 },
  },
  books: [
    {
      book_id: 'real-book',
      capital_scope: 'real',
      config_revision: 7,
      pair: 'BTC/USDT',
      market_type: 'spot',
      status: 'partial',
      hitl: { approval_id: 'approval-1', status: 'approved', config_revision: 7 },
      failure: { stage: 'connection' },
      requested_target_exposure: '0.5',
      target_exposure: '0.4',
      risk: {
        passed: false,
        requested_target_exposure: '0.5',
        capped_target_exposure: '0.4',
        connection_weights: ['connection-weight-marker'],
        connection_targets: [
          {
            book_id: 'real-book',
            connection_id: 'venue-down',
            weight: 'target-weight-marker',
            book_equity: 'target-equity-marker',
            target_exposure: 'target-exposure-marker',
            target_signed_notional: 'target-notional-marker',
          },
        ],
        rejected_by: 'cap',
        reason: 'risk reason',
        cap_source: 'book-cap',
      },
      ready: false,
      errors: ['connection failure'],
      execution: { status: 'partial', requires_attention: true, reallocated: true },
      portfolio_before: {
        book_id: 'real-book',
        capital_scope: 'real',
        pair: 'BTC/USDT',
        total_equity: '1000',
        total_signed_notional: '0',
        connections: [connectionPortfolio],
      },
      portfolio_after: {
        book_id: 'real-book',
        capital_scope: 'real',
        pair: 'BTC/USDT',
        total_equity: '990',
        total_signed_notional: '100',
        connections: [connectionPortfolio],
      },
      portfolio_after_available: true,
      connections: [
        {
          connection_id: 'venue-down',
          portfolio_before: connectionPortfolio,
          portfolio_after: connectionPortfolio,
          risk: {
            connection_id: 'venue-down',
            passed: false,
            risk_increase: true,
            reason: 'connection risk',
            operation: 'risk-check',
          },
          plan,
          unavailable: true,
          execution: {
            book_id: 'real-book',
            connection_id: 'venue-down',
            pair: { symbol: 'BTC/USDT' },
            target_signed_notional: '100',
            target_signed_amount: '1',
            status: 'partial',
            orders: [order],
            protection,
            compensation: {
              attempted: true,
              succeeded: false,
              order,
              operation: 'compensate',
              safe_signed_amount: '0',
              required_protection: protection,
            },
            final_position: {
              position: {
                pair: { symbol: 'BTC/USDT' },
                signed_amount: '1',
                signed_notional: '100',
                entry_price: '101',
              },
              protected: true,
              protection_ids: ['protection-1'],
              protections: [protection],
            },
            error_operation: 'place-order',
            requires_attention: true,
            trace: ['planned', 'failed'],
            execution_quote: quote,
          },
        },
      ],
    },
  ],
});

const renderPage = (page: React.ReactNode) =>
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <MemoryRouter initialEntries={['/cycles/cycle-partial']}>{page}</MemoryRouter>
    </QueryClientProvider>,
  );

afterEach(async () => {
  vi.clearAllMocks();
  await i18n.changeLanguage('zh-CN');
});

describe('cycles pages', () => {
  it('renders the strict partial audit hierarchy exactly once for shared committee evidence', async () => {
    cycleQuery.mockReturnValue({ data: partialCycle, isError: false });
    await i18n.changeLanguage('en-US');
    renderPage(
      <Routes>
        <Route path="/cycles/:cycleId" element={<CycleDetailPage />} />
      </Routes>,
    );
    expect(await screen.findByText('Shared committee evidence')).toBeInTheDocument();
    expect(screen.getAllByText('committee-1')).toHaveLength(1);
    for (const text of [
      'fused reasoning',
      'Target position',
      'Execution book: real-book',
      'R7',
      'approval-1',
      'connection failure',
      'risk reason',
      '0.5',
      '0.4',
      'book-cap',
      'venue-down',
      'connection-weight-marker',
      'target-weight-marker',
      'target-equity-marker',
      'target-exposure-marker',
      'target-notional-marker',
      'Portfolio before',
      'Connection unavailable',
      'Immutable plan',
      'order-marker',
      'order-side-marker',
      'order-type-marker',
      'order-amount-marker',
      'order-filled-marker',
      'filled',
      'order-price-marker',
      'protection-marker',
      'protection-side-marker',
      'protection-amount-marker',
      'protection-stop-marker',
      'protection-take-marker',
      'compensate',
      'false',
      'Final position',
      'place-order',
      'planned → failed',
      'Requires attention',
      'nested plugin value',
      '0.73',
      'nested array value',
    ])
      expect(screen.getAllByText(text, { exact: false }).length).toBeGreaterThan(0);
  });

  it('localizes an unknown backend status while retaining its raw code', async () => {
    await i18n.changeLanguage('en-US');
    expect(formatCycleStatus(i18n.t, 'plugin_waiting')).toContain('Unknown status');
    expect(formatCycleStatus(i18n.t, 'plugin_waiting')).toContain('plugin_waiting');
    await i18n.changeLanguage('zh-CN');
    expect(formatCycleStatus(i18n.t, 'plugin_waiting')).toContain('未知状态');
    expect(formatCycleStatus(i18n.t, 'plugin_waiting')).toContain('plugin_waiting');
  });

  it('renders localized list states, named cycle link, pagination, and status', async () => {
    await i18n.changeLanguage('en-US');
    cyclesQuery.mockReturnValue({
      data: { items: [partialCycle], page: 1, size: 20, total: 2, has_next: true },
      isLoading: false,
      isError: false,
    });
    const view = renderPage(<CyclesPage />);
    expect(await screen.findByRole('link', { name: /cycle-partial/i })).toHaveAttribute(
      'href',
      '/cycles/cycle-partial',
    );
    expect(screen.getByText('Partial')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Next' })).toBeEnabled();
    view.unmount();
    cyclesQuery.mockReturnValue({ data: undefined, isLoading: true, isError: false });
    renderPage(<CyclesPage />);
    expect(screen.getByText('Loading cycles…')).toBeInTheDocument();
  });

  it('renders localized empty and error list states', async () => {
    await i18n.changeLanguage('zh-CN');
    cyclesQuery.mockReturnValue({
      data: { items: [], page: 1, size: 20, total: 0, has_next: false },
      isLoading: false,
      isError: false,
    });
    const view = renderPage(<CyclesPage />);
    expect(screen.getByText('暂无周期记录。')).toBeInTheDocument();
    view.unmount();
    cyclesQuery.mockReturnValue({ data: undefined, isLoading: false, isError: true });
    renderPage(<CyclesPage />);
    expect(screen.getByText('无法加载周期记录。')).toBeInTheDocument();
  });
});

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router';
import { afterEach, describe, expect, it, vi } from 'vitest';

import i18n from '@/lib/i18n';
import { CycleSchema } from '@/types/api.schema';

import { ActivityFeed } from './activity-feed';

const cyclesQuery = vi.fn();
const riskQuery = vi.fn();

vi.mock('@/hooks/use-multi-venue-cycles', () => ({
  useMultiVenueCycles: () => cyclesQuery() as never,
}));
vi.mock('@/hooks/use-risk-status', () => ({
  useRiskStatus: () => riskQuery() as never,
}));

const cycle = CycleSchema.parse({
  cycle_id: 'activity-cycle',
  config_revision: 5,
  market_data_source_id: 'fixture',
  cycle_status: 'completed',
  execution_status: 'completed',
  requires_attention: false,
  created_at: '2026-08-29T00:00:00Z',
  shared_signals: { components: [], fused: null, target_position: { side: 'long', size_ratio: 0.4 } },
  books: [{
    book_id: 'paper', capital_scope: 'simulated', config_revision: 5, pair: 'BTC/USDT', market_type: 'spot', status: 'completed',
    hitl: { approval_id: null, status: 'executed', config_revision: 5 }, failure: null, requested_target_exposure: '0.4', target_exposure: '0.4', risk: null,
    ready: true, errors: [], execution: null, portfolio_before: null, portfolio_after: null, portfolio_after_available: null, connections: [],
  }],
});

const renderFeed = () => render(
  <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
    <MemoryRouter><ActivityFeed /></MemoryRouter>
  </QueryClientProvider>,
);

afterEach(async () => {
  vi.clearAllMocks();
  await i18n.changeLanguage('zh-CN');
});

describe('activity feed localization', () => {
  it('renders canonical cycle direction and status in English without Chinese fallbacks', async () => {
    await i18n.changeLanguage('en-US');
    cyclesQuery.mockReturnValue({ data: { items: [cycle] }, isLoading: false, isError: false });
    riskQuery.mockReturnValue({ data: { recent_blocks: [] }, isLoading: false, isError: false });
    renderFeed();
    expect(await screen.findByText('Today’s activity')).toBeInTheDocument();
    expect(screen.getByText('Long')).toBeInTheDocument();
    expect(screen.getByText('Status: Completed')).toBeInTheDocument();
    expect(screen.queryByText('看多')).not.toBeInTheDocument();
  });

  it('localizes unknown direction and activity error states while preserving raw data', async () => {
    await i18n.changeLanguage('en-US');
    const unknown = CycleSchema.parse({ ...cycle, shared_signals: { ...cycle.shared_signals, target_position: { side: 'plugin_side', size_ratio: 0.4 } } });
    cyclesQuery.mockReturnValue({ data: { items: [unknown] }, isLoading: false, isError: false });
    riskQuery.mockReturnValue({ data: { recent_blocks: [] }, isLoading: false, isError: false });
    const view = renderFeed();
    expect(await screen.findByText('Direction: plugin_side')).toBeInTheDocument();
    view.unmount();
    cyclesQuery.mockReturnValue({ data: undefined, isLoading: false, isError: true });
    riskQuery.mockReturnValue({ data: undefined, isLoading: false, isError: false });
    renderFeed();
    expect(await screen.findByText('Activity is unavailable.')).toBeInTheDocument();
  });
});

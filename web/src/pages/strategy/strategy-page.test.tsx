import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import StrategyPage from './index';

describe('StrategyPage', () => {
  it('loads components from RuntimeConfig instead of a profile endpoint', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({ revision: 3, updated_at: '2026-08-28T00:00:00Z', setup_required: false, document: { system: { active: true }, market_data: { source_id: 'binance', parameters: [] }, llm: { models: {} }, signals: { components: [{ component_id: 'kronos', enabled: true, weight: .6, parameters: [] }], neutral_threshold: .2, max_target_ratio: 1, atr_stop_multiplier: 2, reward_ratio: 2, hitl_required: false }, risk: {}, execution: { connections: [], books: [], allocation_policy: 'weighted' }, hitl: {}, scheduler: {}, triggers: {}, notifications: {}, infrastructure: {} } }), { status: 200 })));
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<QueryClientProvider client={client}><StrategyPage/></QueryClientProvider>);
    expect(await screen.findByText('kronos')).toBeInTheDocument();
    expect(fetch).toHaveBeenCalledWith(expect.stringContaining('/api/config'), expect.anything());
  });
});

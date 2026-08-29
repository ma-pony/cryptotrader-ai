import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import DebatePage from '@/pages/debate';
import i18n from '@/lib/i18n';
import { CycleSchema, type JsonValueOut } from '@/types/api.schema';

const jsonString = (string_value: string) => ({
  kind: 'string' as const, boolean_value: null, number_value: null, string_value, datetime_value: null, pair_value: null, items: [], entries: [],
});
const jsonNumber = (number_value: string) => ({
  kind: 'number' as const, boolean_value: null, number_value, string_value: null, datetime_value: null, pair_value: null, items: [], entries: [],
});
const jsonBoolean = (boolean_value: boolean) => ({
  kind: 'boolean' as const, boolean_value, number_value: null, string_value: null, datetime_value: null, pair_value: null, items: [], entries: [],
});
const jsonObject = (entries: Array<{ key: string; value: JsonValueOut }>) => ({
  kind: 'object' as const, boolean_value: null, number_value: null, string_value: null, datetime_value: null, pair_value: null, items: [], entries,
});
const jsonArray = (items: JsonValueOut[]) => ({
  kind: 'array' as const, boolean_value: null, number_value: null, string_value: null, datetime_value: null, pair_value: null, items, entries: [],
});

const agent = (agent_id: string, direction: string, confidence: string) => jsonObject([
  { key: 'agent_id', value: jsonString(agent_id) },
  { key: 'direction', value: jsonString(direction) },
  { key: 'confidence', value: jsonNumber(confidence) },
  { key: 'reasoning', value: jsonString(`${agent_id} initial rationale`) },
]);
const turn = (from: string, to: string, direction: string, confidence: string) => jsonObject([
  { key: 'round', value: jsonNumber('1') },
  { key: 'from', value: jsonString(from) },
  { key: 'to', value: jsonString(to) },
  { key: 'before', value: jsonObject([{ key: 'direction', value: jsonString('bearish') }, { key: 'confidence', value: jsonNumber('0.62') }]) },
  { key: 'after', value: jsonObject([{ key: 'direction', value: jsonString(direction) }, { key: 'confidence', value: jsonNumber(confidence) }]) },
  { key: 'move', value: jsonString('强化') },
  { key: 'reasoning', value: jsonString('round one critique') },
  { key: 'new_findings', value: jsonString('new on-chain finding') },
]);

const cycle = CycleSchema.parse({
  cycle_id: 'cycle-debate', config_revision: 12, market_data_source_id: 'market-feed', cycle_status: 'completed', execution_status: 'completed', requires_attention: false, created_at: '2026-08-29T00:00:00Z', books: [],
  shared_signals: {
    components: [{
      component_id: 'llm_committee', direction: 'long', confidence: 0.83, reasoning: 'final committee signal',
      details: [
        { key: 'analyses', value: jsonObject([
          { key: 'technical', value: agent('technical', 'bullish', '0.71') },
          { key: 'chain', value: agent('chain', 'bearish', '0.62') },
          { key: 'news', value: agent('news', 'neutral', '0.50') },
          { key: 'macro', value: agent('macro', 'bullish', '0.68') },
        ]) },
        { key: 'debate_turns', value: jsonArray([turn('technical', 'chain', 'bullish', '0.78'), turn('chain', 'technical', 'bearish', '0.59')]) },
        { key: 'consensus_metrics', value: jsonObject([{ key: 'dispersion', value: jsonNumber('0.42') }, { key: 'strength', value: jsonNumber('0.83') }, { key: 'mean_score', value: jsonNumber('0.31') }]) },
        { key: 'debate_skipped', value: jsonBoolean(false) },
        { key: 'debate_skip_reason', value: jsonString('dispersion crossed gate') },
      ],
    }],
    fused: { score: 0.72, reasoning: 'fused', contributions: [{ component_id: 'llm_committee', weight: 1, signed_score: 0.72, weighted_score: 0.72 }] },
    target_position: { side: 'long', size_ratio: 0.4 },
  },
});

const response = (body: unknown) => new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } });

describe('debate canonical cycle details', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('zh-CN');
  });

  afterEach(() => vi.unstubAllGlobals());

  it('loads the canonical cycle record and renders the four-agent internal debate', async () => {
    const fetchMock = vi.fn((url: string, init?: RequestInit) => {
      if (String(url) === '/api/cycles/cycle-debate' && init?.method === 'GET') return Promise.resolve(response(cycle));
      if (String(url) === '/api/cycles?page=1&size=20' && init?.method === 'GET') return Promise.resolve(response({ items: [cycle], total: 1, page: 1, size: 20, has_next: false }));
      throw new Error(`unexpected request ${String(url)} ${init?.method}`);
    });
    vi.stubGlobal('fetch', fetchMock);
    render(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <MemoryRouter initialEntries={['/debate/cycle-debate']}>
          <Routes><Route path="/debate/:cycleId" element={<DebatePage />} /></Routes>
        </MemoryRouter>
      </QueryClientProvider>,
    );

    await screen.findByText('四智能体委员会 · 辩论后总结');
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith('/api/cycles/cycle-debate', expect.objectContaining({ method: 'GET' })));
    expect(fetchMock).not.toHaveBeenCalledWith(expect.stringContaining('/api/decisions'), expect.anything());
    for (const text of ['技术面', '链上', '新闻', '宏观', '第 1 轮 · 交叉挑战', 'round one critique', 'new on-chain finding', '看多', 'final committee signal', '0.42']) {
      expect(screen.getAllByText(text, { exact: false }).length).toBeGreaterThan(0);
    }
    expect(screen.getByText(/触发辩论.*dispersion crossed gate/)).toBeInTheDocument();
    await i18n.changeLanguage('en-US');
    for (const text of ['Four-agent committee · Post-debate summary', 'Round 1 · Cross challenge', 'Debate triggered', 'Bullish']) {
      expect((await screen.findAllByText(text, { exact: false })).length).toBeGreaterThan(0);
    }
    expect(screen.queryByText('四智能体委员会 · 辩论后总结')).not.toBeInTheDocument();
  });

  it('renders controlled unavailable state for schema-valid but malformed committee details', async () => {
    await i18n.changeLanguage('en-US');
    const malformed = CycleSchema.parse({
      ...cycle,
      shared_signals: { ...cycle.shared_signals, components: [{ ...cycle.shared_signals.components[0]!, details: [{ key: 'analyses', value: jsonString('not a committee object') }] }] },
    });
    vi.stubGlobal('fetch', vi.fn((url: string) => {
      if (String(url) === '/api/cycles/cycle-debate') return Promise.resolve(response(malformed));
      if (String(url) === '/api/cycles?page=1&size=20') return Promise.resolve(response({ items: [malformed], total: 1, page: 1, size: 20, has_next: false }));
      throw new Error(`unexpected request ${String(url)}`);
    }));
    render(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <MemoryRouter initialEntries={['/debate/cycle-debate']}><Routes><Route path="/debate/:cycleId" element={<DebatePage />} /></Routes></MemoryRouter>
      </QueryClientProvider>,
    );
    expect(await screen.findByText('Unable to load debate details for cycle cycle-debate.')).toBeInTheDocument();
  });

  it('contains invalid committee number envelopes inside the unavailable state', async () => {
    await i18n.changeLanguage('en-US');
    const malformedNumber = CycleSchema.parse({
      ...cycle,
      shared_signals: {
        ...cycle.shared_signals,
        components: [
          {
            ...cycle.shared_signals.components[0]!,
            details: [
              {
                key: 'analyses',
                value: jsonObject([
                  {
                    key: 'technical',
                    value: jsonObject([
                      { key: 'agent_id', value: jsonString('technical') },
                      { key: 'direction', value: jsonString('bullish') },
                      { key: 'confidence', value: jsonNumber('NaN') },
                      { key: 'reasoning', value: jsonString('valid shape, invalid decoded number') },
                    ]),
                  },
                ]),
              },
            ],
          },
        ],
      },
    });
    vi.stubGlobal('fetch', vi.fn((url: string) => {
      if (String(url) === '/api/cycles/cycle-debate') return Promise.resolve(response(malformedNumber));
      if (String(url) === '/api/cycles?page=1&size=20') return Promise.resolve(response({ items: [malformedNumber], total: 1, page: 1, size: 20, has_next: false }));
      throw new Error(`unexpected request ${String(url)}`);
    }));
    render(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <MemoryRouter initialEntries={['/debate/cycle-debate']}><Routes><Route path="/debate/:cycleId" element={<DebatePage />} /></Routes></MemoryRouter>
      </QueryClientProvider>,
    );
    expect(await screen.findByText('Unable to load debate details for cycle cycle-debate.')).toBeInTheDocument();
    expect(screen.queryByText('Four-agent committee · Post-debate summary')).not.toBeInTheDocument();
  });
});

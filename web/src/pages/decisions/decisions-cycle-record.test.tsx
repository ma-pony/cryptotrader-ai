import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import i18n from '@/lib/i18n';
import type { DecisionDetail } from '@/types/api';

import { DecisionDetailPanel } from '@/components/decision-detail/decision-detail-panel';

const cycleDecision = {
  cycle_id: 'cycle-001',
  ts: '2026-08-28T00:00:00Z',
  pair: 'BTC/USDT',
  pair_display: 'BTC/USDT',
  market_type: 'spot',
  status: 'completed',
  profile_revision: 7,
  context: {
    pair: 'BTC/USDT',
    as_of: '2026-08-28T00:00:00Z',
    mode: 'paper',
    exchange_id: 'binance',
    market_type: 'spot',
    equity: 10_000,
    current_price: 64_000,
    atr: 1_000,
    current_position: { side: 'flat', amount: 0, size_ratio: 0, avg_price: null, unrealized_pnl: 0 },
    portfolio: {},
  },
  components: [
    {
      component_id: 'kronos',
      direction: 'long',
      confidence: 0.8,
      reasoning: '趋势与动量同向。',
      details: {},
    },
    {
      component_id: 'llm_committee',
      direction: 'neutral',
      confidence: 0.6,
      reasoning: '四智能体完成内部辩论后维持中性。',
      details: {
        analyses: {
          tech_agent: { agent_id: 'tech_agent', direction: 'bullish', confidence: 0.7, reasoning: '趋势偏强。' },
        },
        debate_turns: [
          {
            round: 1,
            from: 'tech_agent',
            to: 'macro_agent',
            before: { direction: 'bullish', confidence: 0.7 },
            after: { direction: 'bullish', confidence: 0.72 },
            move: '保持',
            reasoning: '宏观证据不足以推翻趋势。',
            new_findings: '',
            errored: false,
          },
        ],
        consensus_metrics: { strength: 0.45, mean_score: 0.2, dispersion: 0.3 },
        debate_skipped: false,
        debate_skip_reason: '',
      },
    },
  ],
  component_error: null,
  fusion: {
    score: 0.48,
    reasoning: 'kronos: +0.48; llm_committee: +0.00',
    contributions: [
      { component_id: 'kronos', weight: 0.6, signed_score: 0.8, weighted_score: 0.48 },
      { component_id: 'llm_committee', weight: 0.4, signed_score: 0, weighted_score: 0 },
    ],
  },
  target_position: { side: 'long', size_ratio: 0.3 },
  trade_plan: {
    target: { side: 'long', size_ratio: 0.3 },
    stop_loss: 62_000,
    take_profit: 68_000,
    component_signals: [],
    fused_signal: {
      score: 0.48,
      reasoning: 'weighted',
      contributions: [],
    },
  },
  hitl_result: null,
  risk_result: { passed: true, rejected_by: '', reason: '', target: { side: 'long', size_ratio: 0.3 } },
  execution_result: { succeeded: true, algo_id: 'oco-1', error: null, orders: [] },
} as unknown as DecisionDetail;

vi.mock('@/hooks/use-decision-detail', () => ({
  useDecisionDetail: () => ({ data: cycleDecision, isLoading: false, isError: false }),
}));

describe('cycle decision detail', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('zh-CN');
  });

  it('renders component contributions, target position, and internal debate', () => {
    render(
      <MemoryRouter>
        <DecisionDetailPanel cycleId="cycle-001" />
      </MemoryRouter>,
    );

    expect(screen.getAllByText('Kronos').length).toBeGreaterThan(0);
    expect(screen.getAllByText('+0.48').length).toBeGreaterThan(0);
    expect(screen.getAllByText('目标多仓 30%').length).toBeGreaterThan(0);
    expect(screen.getByText('内部辩论')).toBeInTheDocument();
    expect(screen.queryByText('AI 裁决')).not.toBeInTheDocument();
  });
});

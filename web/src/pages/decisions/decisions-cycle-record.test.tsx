import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import i18n from '@/lib/i18n';
import type { DecisionDetail } from '@/types/api';
import { CycleRiskResultSchema } from '@/types/api.schema';

import {
  CycleDecisionDetail,
  DecisionDetailPanel,
} from '@/components/decision-detail/decision-detail-panel';

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
  error: null,
  fusion: {
    score: 0.48,
    reasoning: 'kronos: +0.48; llm_committee: +0.00',
    contributions: [
      { component_id: 'kronos', weight: 0.6, signed_score: 0.8, weighted_score: 0.48 },
      { component_id: 'llm_committee', weight: 0.4, signed_score: 0, weighted_score: 0 },
    ],
  },
  target_position: { side: 'long', size_ratio: 0.8 },
  trade_plan: {
    target: { side: 'long', size_ratio: 0.8 },
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
  risk_result: {
    passed: true,
    rejected_by: '',
    reason: 'capped by max single position',
    cap_source: 'max_single_position',
    target: { side: 'long', size_ratio: 0.2 },
  },
  execution_result: {
    succeeded: true,
    algo_id: 'oco-active-21',
    error: null,
    retained_algo_ids: ['oco-retained-08'],
    protection_trigger: {
      algo_id: 'paper-oco-13',
      trigger_reason: 'stop_loss',
      trigger_price: 63_500,
      order_id: 'paper-close-34',
    },
    orders: [
      {
        intent: {
          pair: 'BTC/USDT',
          side: 'buy',
          amount: 0.125,
          reduce_only: false,
        },
        status: 'partially_filled',
        exchange_id: 'exchange-order-55',
        raw: {},
        filled_amount: 0.08,
      },
    ],
  },
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
    expect(screen.getAllByText('目标多仓 80%').length).toBeGreaterThan(0);
    expect(screen.getByText('内部辩论')).toBeInTheDocument();
    expect(screen.queryByText('AI 裁决')).not.toBeInTheDocument();
  });

  it('preserves structured risk cap provenance in the API schema', () => {
    const parsed = CycleRiskResultSchema.parse(cycleDecision.risk_result);

    expect(parsed).toEqual(cycleDecision.risk_result);
  });

  it('renders original and adjusted targets with non-empty execution safety facts', () => {
    render(
      <MemoryRouter>
        <DecisionDetailPanel cycleId="cycle-001" />
      </MemoryRouter>,
    );

    expect(screen.getByText('原始目标')).toBeInTheDocument();
    expect(screen.getByText('风控后目标')).toBeInTheDocument();
    expect(screen.getByText('仓位上限来源')).toBeInTheDocument();
    expect(screen.getByText('max_single_position')).toBeInTheDocument();
    expect(screen.getAllByText('目标多仓 20%').length).toBeGreaterThan(0);
    expect(screen.getByText('订单执行')).toBeInTheDocument();
    expect(screen.getByText('partially_filled')).toBeInTheDocument();
    expect(screen.getByText('0.125')).toBeInTheDocument();
    expect(screen.getByText('0.08')).toBeInTheDocument();
    expect(screen.getByText('oco-active-21')).toBeInTheDocument();
    expect(screen.getByText('oco-retained-08')).toBeInTheDocument();
    expect(screen.getByText('stop_loss')).toBeInTheDocument();
    expect(screen.getByText('$63,500.00 USDT')).toBeInTheDocument();
    expect(screen.getByText('paper-close-34')).toBeInTheDocument();
  });

  it('keeps an empty execution result compact', () => {
    const emptyExecution = {
      ...cycleDecision,
      execution_result: {
        succeeded: true,
        algo_id: null,
        error: null,
        retained_algo_ids: [],
        protection_trigger: null,
        orders: [],
      },
    } as DecisionDetail;

    render(
      <MemoryRouter>
        <CycleDecisionDetail data={emptyExecution} />
      </MemoryRouter>,
    );

    expect(screen.queryByText('订单执行')).not.toBeInTheDocument();
    expect(screen.queryByText('保护单')).not.toBeInTheDocument();
    expect(screen.queryByText('保护触发')).not.toBeInTheDocument();
  });

  it('renders the new audit labels consistently in English', async () => {
    await i18n.changeLanguage('en-US');

    render(
      <MemoryRouter>
        <DecisionDetailPanel cycleId="cycle-001" />
      </MemoryRouter>,
    );

    expect(screen.getByText('Original target')).toBeInTheDocument();
    expect(screen.getByText('Risk-adjusted target')).toBeInTheDocument();
    expect(screen.getByText('Cap source')).toBeInTheDocument();
    expect(screen.getByText('Order execution')).toBeInTheDocument();
    expect(screen.getByText('Protection orders')).toBeInTheDocument();
    expect(screen.getByText('Protection trigger')).toBeInTheDocument();
    expect(screen.getAllByText('Target long 20%').length).toBeGreaterThan(0);
  });
});

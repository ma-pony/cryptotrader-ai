/**
 * Contract tests: verify frontend zod schemas parse representative backend responses.
 *
 * These guard against schema drift between:
 *   - backend pydantic models (src/api/routes/*.py)
 *   - frontend zod schemas (src/types/api.schema.ts)
 *
 * Each payload in this file mirrors the exact shape produced by a route in a
 * realistic success scenario. If a backend field is renamed or re-typed without
 * a matching frontend zod update, the corresponding test fails.
 */

import { describe, expect, it } from 'vitest';

import {
  DecisionDetailSchema,
  DecisionListItemSchema,
  MetricsSummarySchema,
  PaginatedDecisionsSchema,
  PortfolioSchema,
  RiskStatusSchema,
} from '@/types/api.schema';

describe('Portfolio schema contract', () => {
  it('parses full snapshot with all Phase-1 extras', () => {
    const backend = {
      equity: 128450.33,
      cash: 42180.0,
      positions: [
        {
          pair: 'BTC/USDT',
          pair_display: 'BTC/USDT',
          side: 'long',
          size: 0.42,
          avg_price: 89120,
          unrealized_pnl: 1831.2,
          unrealized_pnl_pct: 0.0489,
          opened_at: '2026-04-22T10:32:00+00:00',
        },
      ],
      pnl_24h: 2394.16,
      pnl_24h_pct: 0.0187,
      drawdown: 0.083,
      updated_at: '2026-04-24T14:32:08+00:00',
      sharpe_90d: 2.14,
      win_rate: 0.627,
      total_trades: 142,
      realized_pnl_30d: 14620.18,
    };

    const parsed = PortfolioSchema.parse(backend);
    expect(parsed.sharpe_90d).toBe(2.14);
    expect(parsed.win_rate).toBe(0.627);
    expect(parsed.total_trades).toBe(142);
    expect(parsed.realized_pnl_30d).toBe(14620.18);
  });

  it('accepts null Phase-1 extras (under-30-samples path)', () => {
    const backend = {
      equity: 10000,
      cash: 10000,
      positions: [],
      pnl_24h: 0,
      pnl_24h_pct: 0,
      drawdown: 0,
      updated_at: '2026-04-24T00:00:00+00:00',
      sharpe_90d: null,
      win_rate: null,
      total_trades: 0,
      realized_pnl_30d: 0,
    };
    expect(() => PortfolioSchema.parse(backend)).not.toThrow();
  });

  it('tolerates backend without new fields (old deployment)', () => {
    const old = {
      equity: 10000,
      cash: 10000,
      positions: [],
      pnl_24h: 0,
      pnl_24h_pct: 0,
      drawdown: 0,
      updated_at: '2026-04-24T00:00:00+00:00',
    };
    const parsed = PortfolioSchema.parse(old);
    expect(parsed.total_trades).toBe(0);
    expect(parsed.sharpe_90d).toBeUndefined();
  });
});

describe('Decision list + detail schemas', () => {
  it('parses a cycle list item', () => {
    const item = {
      cycle_id: 'cycle-1',
      ts: '2026-08-28T10:32:08+00:00',
      pair: 'BTC/USDT',
      pair_display: 'BTC/USDT',
      market_type: 'spot',
      status: 'completed',
      profile_revision: 7,
      price: 92810.22,
      fused_score: 0.48,
      target_position: { side: 'long', size_ratio: 0.3 },
      component_error: null,
      risk_result: { passed: true, rejected_by: '', reason: '', target: { side: 'long', size_ratio: 0.3 } },
      execution_result: { succeeded: true, algo_id: 'oco-1', error: null, orders: [] },
    };
    const parsed = DecisionListItemSchema.parse(item);
    expect(parsed.cycle_id).toBe('cycle-1');
    expect(parsed.fused_score).toBe(0.48);
  });

  it('parses paginated list envelope', () => {
    expect(() =>
      PaginatedDecisionsSchema.parse({
        items: [],
        total: 0,
        page: 1,
        size: 20,
        has_next: false,
      }),
    ).not.toThrow();
  });

  it('parses component, debate, fusion, target, risk, and execution groups', () => {
    const detail = {
      cycle_id: 'cycle-1',
      ts: '2026-08-28T10:32:08+00:00',
      pair: 'BTC/USDT',
      pair_display: 'BTC/USDT',
      market_type: 'spot',
      status: 'completed',
      profile_revision: 7,
      context: {
        pair: 'BTC/USDT',
        as_of: '2026-08-28T10:32:08+00:00',
        mode: 'paper',
        exchange_id: 'binance',
        market_type: 'spot',
        equity: 10_000,
        current_price: 92810.22,
        atr: 1_200,
        current_position: { side: 'flat', amount: 0, size_ratio: 0, avg_price: null, unrealized_pnl: 0 },
        portfolio: {},
      },
      components: [{
        component_id: 'llm_committee',
        direction: 'long',
        confidence: 0.75,
        reasoning: 'committee summary',
        details: {
          analyses: {
            tech_agent: { agent_id: 'tech_agent', direction: 'bullish', confidence: 0.82, reasoning: 'bull case' },
          },
          debate_turns: [{
            round: 1,
            from: 'tech_agent',
            to: 'chain_agent',
            before: { direction: 'bullish', confidence: 0.6 },
            after: { direction: 'bullish', confidence: 0.85 },
            move: '强化',
            reasoning: 'new finding',
            new_findings: 'whale alert',
            errored: false,
          }],
          consensus_metrics: { strength: 0.7, mean_score: 0.5, dispersion: 0.15 },
          debate_skipped: false,
          debate_skip_reason: '',
        },
      }],
      component_error: null,
      fusion: {
        score: 0.48,
        reasoning: 'weighted',
        contributions: [
          { component_id: 'llm_committee', weight: 0.4, signed_score: 0.75, weighted_score: 0.3 },
        ],
      },
      target_position: { side: 'long', size_ratio: 0.3 },
      trade_plan: {
        target: { side: 'long', size_ratio: 0.3 },
        stop_loss: 90_000,
        take_profit: 98_000,
        component_signals: [],
        fused_signal: { score: 0.48, reasoning: 'weighted', contributions: [] },
      },
      hitl_result: null,
      risk_result: { passed: true, rejected_by: '', reason: '', target: { side: 'long', size_ratio: 0.3 } },
      execution_result: {
        succeeded: true,
        algo_id: 'oco-1',
        error: null,
        orders: [
        {
          intent: { pair: 'BTC/USDT', side: 'buy', amount: 0.1, reduce_only: false },
          status: 'filled',
          exchange_id: 'order-1',
          raw: {},
        },
        ],
      },
    };
    const parsed = DecisionDetailSchema.parse(detail);
    expect(parsed.components[0]?.details.debate_turns).toHaveLength(1);
    expect(parsed.fusion?.score).toBe(0.48);
    expect(parsed.target_position?.side).toBe('long');
  });

  it('rejects the removed legacy verdict contract', () => {
    const legacy = {
      commit_hash: 'abc',
      ts: '2026-01-01T00:00:00+00:00',
      pair: 'BTC/USDT',
      price: 50000,
      verdict: { action: 'hold', size: 0, confidence: 0, reasoning: '', source: 'ai' },
    };
    expect(DecisionDetailSchema.safeParse(legacy).success).toBe(false);
  });
});

describe('Risk schema contract', () => {
  it('parses full response with 4 meters + groups + cooldowns + blocks', () => {
    const backend = {
      trade_count_hour: 2,
      trade_count_day: 7,
      circuit_breaker: {
        state: 'inactive',
        triggered_at: null,
        expires_at: null,
        reason: null,
      },
      thresholds: {
        max_position_pct: 0.1,
        max_daily_loss_pct: 0.03,
        max_stop_loss_pct: 0.05,
        max_trades_per_hour: 6,
        max_trades_per_day: 20,
        post_loss_cooldown_seconds: 7200,
      },
      redis_available: true,
      daily_loss_pct: 0.8,
      drawdown_pct: 2.1,
      total_exposure_pct: 42.0,
      cvar_95: 3.4,
      correlation_groups: [
        { name: 'BTC-correlated', open: 1, max: 2, pairs: ['BTC/USDT', 'BTC/USD'] },
      ],
      cooldowns: [
        { pair: 'BTC/USDT', until_seconds: 1680, kind: 'same-pair' },
        { pair: '*', until_seconds: 3600, kind: 'post-loss' },
      ],
      recent_blocks: [
        {
          ts: '2026-04-24T04:32:00+00:00',
          cycle_id: '9f2c8e1',
          rule: 'CooldownCheck',
          detail: 'same-pair cooldown active',
        },
      ],
    };
    const parsed = RiskStatusSchema.parse(backend);
    expect(parsed.cvar_95).toBe(3.4);
    expect(parsed.correlation_groups).toHaveLength(1);
    expect(parsed.cooldowns[0]!.kind).toBe('same-pair');
    expect(parsed.recent_blocks[0]!.rule).toBe('CooldownCheck');
  });

  it('accepts all-null meters (no data available)', () => {
    const backend = {
      trade_count_hour: null,
      trade_count_day: null,
      circuit_breaker: { state: 'inactive' },
      thresholds: {
        max_position_pct: 0.1,
        max_daily_loss_pct: 0.03,
        max_stop_loss_pct: 0.05,
        max_trades_per_hour: 6,
        max_trades_per_day: 20,
        post_loss_cooldown_seconds: 7200,
      },
      redis_available: false,
      daily_loss_pct: null,
      drawdown_pct: null,
      total_exposure_pct: null,
      cvar_95: null,
    };
    const parsed = RiskStatusSchema.parse(backend);
    expect(parsed.cvar_95).toBeNull();
    expect(parsed.correlation_groups).toEqual([]);
    expect(parsed.cooldowns).toEqual([]);
  });
});

describe('Metrics schema contract', () => {
  it('parses full summary with histogram + cost series', () => {
    const backend = {
      counters: {
        trades_total: 142,
        orders_placed: 142,
        orders_failed: 0,
        risk_rejections: 12,
        debate_skipped_total: 23,
      },
      percentiles: {
        pipeline_p50_ms: 1250,
        pipeline_p95_ms: 4800,
        execution_p50_ms: 320,
        execution_p95_ms: 880,
      },
      collected_at: '2026-04-24T14:32:08+00:00',
      llm_calls_24h: 612,
      llm_cost_24h: 8.42,
      cache_hit_rate: 0.72,
      decisions_per_day: 6.0,
      latency_histogram: [
        { upper_bound_s: 1.0, count: 10 },
        { upper_bound_s: 5.0, count: 50 },
        { upper_bound_s: 1e12, count: 200 },
      ],
      cost_14d: [
        { ts: '2026-04-11', cost_usd: 3.2 },
        { ts: '2026-04-24', cost_usd: 8.42 },
      ],
    };
    const parsed = MetricsSummarySchema.parse(backend);
    expect(parsed.llm_calls_24h).toBe(612);
    expect(parsed.cache_hit_rate).toBeLessThanOrEqual(1.0);  // I-C3 invariant
    expect(parsed.latency_histogram).toHaveLength(3);
    expect(parsed.cost_14d).toHaveLength(2);
  });

  it('fills defaults when old server omits Phase-1 fields', () => {
    const old = {
      counters: {
        trades_total: 0,
        orders_placed: 0,
        orders_failed: 0,
        risk_rejections: 0,
        debate_skipped_total: 0,
      },
      percentiles: {
        pipeline_p50_ms: 0,
        pipeline_p95_ms: 0,
        execution_p50_ms: 0,
        execution_p95_ms: 0,
      },
      collected_at: '2026-04-24T00:00:00+00:00',
    };
    const parsed = MetricsSummarySchema.parse(old);
    expect(parsed.llm_calls_24h).toBe(0);
    expect(parsed.latency_histogram).toEqual([]);
    expect(parsed.cost_14d).toEqual([]);
  });
});

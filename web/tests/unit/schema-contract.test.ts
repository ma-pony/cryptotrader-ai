import { describe, expect, it } from 'vitest';

import { MetricsSummarySchema, RiskStatusSchema } from '@/types/api.schema';

describe('Risk schema contract', () => {
  it('parses meters, correlation groups, cooldowns, and blocks', () => {
    const parsed = RiskStatusSchema.parse({
      trade_count_hour: 2,
      trade_count_day: 7,
      circuit_breaker: { state: 'inactive', triggered_at: null, expires_at: null, reason: null },
      thresholds: {
        max_single_pct: 0.45,
        max_total_exposure_pct: 0.8,
        max_margin_used_pct: 0.35,
        max_drawdown_pct: 0.12,
      },
      redis_available: true,
      daily_loss_pct: 0.8,
      drawdown_pct: 2.1,
      total_exposure_pct: 42,
      cvar_95: 3.4,
      correlation_groups: [{ name: 'BTC-correlated', open: 1, pairs: ['BTC/USDT'] }],
      cooldowns: [{ pair: 'BTC/USDT', until_seconds: 1680, kind: 'same-pair' }],
      recent_blocks: [
        { ts: '2026-04-24T04:32:00+00:00', cycle_id: 'cycle-1', rule: 'CooldownCheck', detail: 'active' },
      ],
    });
    expect(parsed.correlation_groups).toHaveLength(1);
    expect(parsed.cooldowns[0]?.kind).toBe('same-pair');
  });
});

describe('Metrics schema contract', () => {
  it('parses the histogram and cost series', () => {
    const parsed = MetricsSummarySchema.parse({
      counters: {
        trades_total: 142,
        orders_placed: 142,
        orders_failed: 0,
        risk_rejections: 12,
        debate_skipped_total: 23,
      },
      percentiles: { pipeline_p50_ms: 1250, pipeline_p95_ms: 4800, execution_p50_ms: 320, execution_p95_ms: 880 },
      collected_at: '2026-04-24T14:32:08+00:00',
      llm_calls_24h: 612,
      llm_cost_24h: 8.42,
      cache_hit_rate: 0.72,
      decisions_per_day: 6,
      latency_histogram: [{ upper_bound_s: 1, count: 10 }],
      cost_14d: [{ ts: '2026-04-24', cost_usd: 8.42 }],
    });
    expect(parsed.llm_calls_24h).toBe(612);
    expect(parsed.cost_14d).toHaveLength(1);
  });
});

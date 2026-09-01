import { describe, expect, it } from 'vitest';

import { MetricsSummarySchema } from '@/types/api.schema';

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

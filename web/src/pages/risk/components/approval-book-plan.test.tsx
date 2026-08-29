import { describe, expect, it } from 'vitest';

import { ApprovalRequestSchema } from '@/types/api.schema';

describe('approval book plan contract', () => {
  it('requires a strict immutable book proposal without numeric input controls', () => {
    const result = ApprovalRequestSchema.safeParse({
      approval_id: 'a',
      cycle_id: 'c',
      book_id: 'b',
      pair: 'BTC/USDT',
      config_revision: 1,
      status: 'pending',
      created_at: '2026-08-29T00:00:00Z',
      decided_at: null,
      proposal: {
        version: 1,
        book_id: 'b',
        capital_scope: 'real',
        config_revision: 1,
        pair: { symbol: 'BTC/USDT' },
        requested_target_exposure: '0.1',
        target_exposure: '0.1',
        risk: {
          passed: true,
          requested_target_exposure: '0.1',
          capped_target_exposure: '0.1',
          connection_weights: [],
          connection_targets: [],
          rejected_by: '',
          reason: '',
          cap_source: '',
        },
        connection_risks: [],
        connection_plans: [],
        unavailable_connections: [],
        errors: [],
        ready: true,
      },
    });
    expect(result.success).toBe(true);
  });
});

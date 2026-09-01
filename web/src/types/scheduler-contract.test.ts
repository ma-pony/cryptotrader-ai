import { describe, expect, it } from 'vitest';

import { ApiErrorSchema, ScheduleRuleSchema, TriggerEventSchema } from './api.schema';

const commonRule = {
  id: 'rule-1',
  name: 'BTC price',
  pair: 'BTC/USDT',
  cooldown_minutes: 30,
  enabled: true,
  ttl_expires_at: null,
  created_by: 'operator',
  schedule_depth: 0,
  created_at: '2026-09-01T00:00:00Z',
  updated_at: '2026-09-01T00:00:00Z',
  in_cooldown: false,
  last_triggered_at: null,
};

describe('scheduler response contracts', () => {
  it('rejects parameters that do not match the rule discriminator', () => {
    expect(
      ScheduleRuleSchema.safeParse({
        ...commonRule,
        trigger_type: 'price_threshold',
        parameters: { threshold_pct: 0.1 },
      }).success,
    ).toBe(false);
  });

  it.each([
    {},
    { pair: 'BTC/USDT', price: 49_000, ts: 1_777_593_600, unexpected: true },
  ])('rejects an empty or malformed persisted trigger snapshot', (priceSnapshot) => {
    expect(
      TriggerEventSchema.safeParse({
        id: 'event-1',
        rule_id: 'rule-1',
        triggered_at: '2026-09-01T00:00:00Z',
        trigger_reason: 'price crossed',
        price_snapshot: priceSnapshot,
        analysis_commit_id: null,
        schedule_depth: 0,
        cooldown_skipped: false,
      }).success,
    ).toBe(false);
  });
});

describe('API error details contract', () => {
  it('accepts only string field errors', () => {
    expect(
      ApiErrorSchema.safeParse({
        code: 'validation_error',
        message: 'Invalid fields',
        details: { fieldErrors: { 'signals.weight': { private: true } } },
      }).success,
    ).toBe(false);
  });
});

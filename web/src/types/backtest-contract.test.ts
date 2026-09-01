import { describe, expect, it } from 'vitest';

import { BacktestRunStatusSchema } from './api.schema';
import { researchRun } from '@/test/research-fixture';

describe('backtest response contract', () => {
  it('rejects an undeclared field in a persisted decision', () => {
    const run = researchRun();
    run.result!.decisions = [
      {
        cycle_id: 'cycle-1',
        status: 'completed',
        config_revision: 1,
        ts: null,
        components: null,
        fusion: null,
        target_position: null,
        books: null,
        unexpected_legacy_field: true,
      } as never,
    ];

    expect(BacktestRunStatusSchema.safeParse(run).success).toBe(false);
  });

  it('rejects non-JSON values in dynamic research evidence', () => {
    const run = researchRun();
    (run.result!.data_coverage as Record<string, unknown>).candles = undefined;

    expect(BacktestRunStatusSchema.safeParse(run).success).toBe(false);
  });
});

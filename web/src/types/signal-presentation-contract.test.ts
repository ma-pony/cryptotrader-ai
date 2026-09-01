import { execFileSync } from 'node:child_process';
import { join, resolve } from 'node:path';
import { beforeAll, describe, expect, it } from 'vitest';
import { PaginatedCyclesSchema } from './api.schema';

// Produce the actual API DTO from domain models and an isolated SQLite journal.
// No copied JSON fixture: a backend/frontend contract mismatch must fail here.
const persistedDto = `
import asyncio
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from api.routes.response_dto import PaginatedCyclesOut, cycle_out
from cryptotrader.db import get_engine
from cryptotrader.journal.store import MultiVenueCycleStore
from cryptotrader.migrations.workbench import migrate_workbench_schema
from cryptotrader.signals.presentation import (
    EvaluationReference, Series, SeriesBlock, SeriesPoint, TimelineBlock, TimelineEntry,
)
from tests.test_multi_venue_journal import _record

async def main():
    with TemporaryDirectory() as temporary:
        database_url = f"sqlite+aiosqlite:///{Path(temporary) / 'contract.db'}"
        await migrate_workbench_schema(database_url)
        store = MultiVenueCycleStore(database_url)
        try:
            records = []
            for index, offset in enumerate(("+00:00", "+08:00")):
                at = datetime.fromisoformat("2026-08-31T09:00:00" + offset)
                due = at + timedelta(hours=1)
                record = _record(cycle_id=f"timezone-{index}")
                signal = replace(record.component_signals[0], blocks=(
                    SeriesBlock(title="保存曲线", forecast_start=due,
                      evaluation_target="candle_close" if index == 0 else None, series=(
                        Series(name="实际输出", points=(SeriesPoint(time=at, value="101.2300"),)),
                    )),
                    TimelineBlock(title="保存意见", entries=(
                        TimelineEntry(time=at, actor="研究员", body="完整意见"),
                    )),
                ), evaluation_reference=EvaluationReference(
                    reference_time=at, reference_price="101.2300", due_at=due,
                    interval="1h", market_source_id="default",
                ))
                record = replace(record, component_signals=(signal, record.component_signals[1]))
                await store.save(record)
                restored = await store.get(record.cycle_id)
                assert restored.component_signals == record.component_signals
                records.append(cycle_out(restored))
            print(PaginatedCyclesOut(items=records, total=2, page=1, size=20, has_next=False).model_dump_json())
        finally:
            await (await get_engine(store.database_url)).dispose()

asyncio.run(main())
`;

function savedEvidence(page: ReturnType<typeof PaginatedCyclesSchema.parse>, index: number) {
  const signal = page.items[index]?.shared_signals.components[0];
  const curve = signal?.blocks.find((block) => block.kind === 'series');
  const timeline = signal?.blocks.find((block) => block.kind === 'timeline');
  const point = curve?.series[0]?.points[0];
  const entry = timeline?.entries[0];
  const reference = signal?.evaluation_reference;
  if (!curve || !point || !entry || !reference) throw new Error('missing saved evidence');
  return { curve, point, entry, reference };
}

describe('persisted result time contract', () => {
  let savedPage: unknown;

  beforeAll(() => {
    const repository = resolve(process.cwd(), '..');
    const isolatedEnv = { ...process.env };
    delete isolatedEnv.DATABASE_URL;
    delete isolatedEnv.CONFIG_MASTER_KEY;
    delete isolatedEnv.RUN_CONTAINER_BOOTSTRAP_CHECK;
    savedPage = JSON.parse(
      execFileSync(join(repository, '.venv/bin/python'), ['-c', persistedDto], {
        cwd: repository,
        env: isolatedEnv,
        encoding: 'utf8',
        timeout: 12000,
      }),
    );
  }, 15000);

  it('accepts UTC and explicit offsets from domain → SQLite → DTO without changing saved times', () => {
    const page = PaginatedCyclesSchema.parse(savedPage);
    for (const [index, suffix] of ['Z', '+08:00'].entries()) {
      const { curve, point, entry, reference } = savedEvidence(page, index);
      expect(point).toEqual({
        time: `2026-08-31T09:00:00${suffix}`,
        value: '101.2300',
      });
      expect(curve.forecast_start).toBe(`2026-08-31T10:00:00${suffix}`);
      expect(curve.evaluation_target).toBe(index === 0 ? 'candle_close' : null);
      expect(entry.time).toBe(`2026-08-31T09:00:00${suffix}`);
      expect(reference.reference_time).toBe(`2026-08-31T09:00:00${suffix}`);
      expect(reference.due_at).toBe(`2026-08-31T10:00:00${suffix}`);
    }
  });

  it.each(['point', 'forecast_start', 'entry', 'reference_time', 'due_at'] as const)(
    'still rejects a naive %s instead of guessing a browser timezone',
    (field) => {
      const page = PaginatedCyclesSchema.parse(savedPage);
      const { curve, point, entry, reference } = savedEvidence(page, 0);
      const naive = '2026-08-31T09:00:00';
      if (field === 'point') point.time = naive;
      else if (field === 'forecast_start') curve.forecast_start = naive;
      else if (field === 'entry') entry.time = naive;
      else reference[field] = naive;
      expect(PaginatedCyclesSchema.safeParse(page).success).toBe(false);
    },
  );
});

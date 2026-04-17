import { execSync } from 'node:child_process';
import { test, expect } from '@playwright/test';

/**
 * FR-915 e2e gate: After Phase 8 (Streamlit physical removal) this test MUST pass.
 * Until then it is expected to fail — running rg from repo root must return 0 hits
 * for any case-insensitive match of "streamlit" outside of node_modules / .venv.
 */
test('repo contains zero references to streamlit (FR-915)', () => {
  let output: string;
  try {
    output = execSync(
      "rg -i --hidden --glob='!node_modules' --glob='!.venv' --glob='!web/node_modules' --glob='!.git' --files-with-matches 'streamlit' .. || true",
      { cwd: process.cwd(), encoding: 'utf8' },
    );
  } catch (err) {
    output = String(err);
  }
  const hits = output.split('\n').filter((line) => line.trim().length > 0);
  expect(hits, `Streamlit references still present: ${hits.join(', ')}`).toEqual([]);
});

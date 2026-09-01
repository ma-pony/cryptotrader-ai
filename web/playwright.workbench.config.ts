import path from 'node:path';
import { defineConfig, devices } from '@playwright/test';

const repositoryRoot = path.resolve(import.meta.dirname, '..');
const python = path.join(repositoryRoot, '.venv', 'bin', 'python');
const node = '/Users/rccpony/.nvm/versions/node/v24.19.0/bin/node';
const vite = path.join(import.meta.dirname, 'node_modules', 'vite', 'bin', 'vite.js');
const registryVariant = process.env.WORKBENCH_REGISTRY_VARIANT ?? 'extension';
if (!['baseline', 'extension'].includes(registryVariant)) {
  throw new Error(`Unknown workbench registry variant: ${registryVariant}`);
}
const backendModule = registryVariant === 'baseline' ? 'tests.workbench_baseline_app:app' : 'tests.workbench_app:app';

const backendTestCommand = `/usr/bin/env -u DATABASE_URL -u CONFIG_MASTER_KEY ${python} -m uvicorn ${backendModule} --host 127.0.0.1 --port 8011`;
const frontendPreviewCommand = `${node} ${vite} preview --host 127.0.0.1 --port 4174 --strictPort`;

export default defineConfig({
  testDir: './tests/e2e',
  testMatch:
    registryVariant === 'baseline'
      ? ['baseline-catalog.spec.ts']
      : ['workbench.spec.ts', 'extension-catalog.spec.ts', 'mobile-layout.spec.ts'],
  fullyParallel: false,
  workers: 1,
  forbidOnly: true,
  retries: 0,
  reporter: 'list',
  timeout: 90_000,
  expect: { timeout: 10_000 },
  use: {
    baseURL: 'http://127.0.0.1:4174',
    actionTimeout: 10_000,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
  webServer: [
    {
      command: backendTestCommand,
      cwd: repositoryRoot,
      url: 'http://127.0.0.1:8011/health',
      reuseExistingServer: false,
      timeout: 120_000,
    },
    {
      command: frontendPreviewCommand,
      cwd: import.meta.dirname,
      url: 'http://127.0.0.1:4174',
      reuseExistingServer: false,
      timeout: 120_000,
      env: { VITE_API_BASE_URL: 'http://127.0.0.1:8011' },
    },
  ],
});

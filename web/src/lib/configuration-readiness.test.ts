import { expect, it } from 'vitest';
import { runtimeConfigFixture } from '@/test/runtime-config-fixture';
import { toRuntimeDocument } from '@/hooks/use-runtime-config';
import { workflowCatalog } from '@/test/configuration-workflow';
import {
  bookErrors,
  connectionChecksReady,
  connectionFingerprint,
  type Connection,
  type ConnectionCheck,
} from './configuration-readiness';

const paper: Connection = {
  id: 'paper',
  label: 'Paper',
  adapter_id: 'paper',
  environment: 'paper',
  enabled: true,
  canary_only: false,
  leverage: 1,
  margin_mode: 'cross',
  parameters: {},
};

it('binds readiness to every saved connection parameter and credential timestamp', () => {
  const document = toRuntimeDocument(runtimeConfigFixture().document);
  const connection = { ...paper };
  document.execution.connections = [connection];
  const checks = {
    [connection.id]: {
      fingerprint: connectionFingerprint(connection, 'first'),
      health: { healthy: true },
    } as ConnectionCheck,
  };
  expect(connectionChecksReady(document, checks, { [connection.id]: { updatedAt: 'first' } }, workflowCatalog)).toBe(true);
  expect(connectionChecksReady(document, checks, { [connection.id]: { updatedAt: 'rotated' } }, workflowCatalog)).toBe(false);
  connection.parameters = { initial_equity: 20000 };
  expect(connectionChecksReady(document, checks, { [connection.id]: { updatedAt: 'first' } }, workflowCatalog)).toBe(false);
});

it('requires valid scope-safe unique allocations that total 100 percent', () => {
  const connection = paper;
  const book = {
    id: 'book',
    label: 'Book',
    capital_scope: 'simulated' as const,
    enabled: true,
    hitl_required: true,
    allocations: [{ connection_id: connection.id, enabled: true, weight: 1 }],
  };
  expect(bookErrors([book], [connection], workflowCatalog, (key) => key)).toEqual({});
  for (const invalid of [
    { ...connection, enabled: false },
    { ...connection, canary_only: true },
    { ...connection, environment: 'live' as const },
  ]) {
    expect(bookErrors([book], [invalid], workflowCatalog, (key) => key)['execution.books.0.allocations.0.weight']).toBe(
      'invalidConnection',
    );
  }
  expect(
    bookErrors([book, { ...book, id: 'other' }], [connection], workflowCatalog, (key) => key)['execution.books.1.allocations.0.weight'],
  ).toBe('duplicateConnection');
  expect(
    bookErrors([{ ...book, allocations: [{ ...book.allocations[0]!, weight: 0.4 }] }], [connection], workflowCatalog, (key) => key)[
      'execution.books.0.allocations.0.weight'
    ],
  ).toBe('bookWeight');
});

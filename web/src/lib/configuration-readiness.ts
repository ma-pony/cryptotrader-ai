import type { ConfigurationDraft, ConnectionHealth, RuntimeDocument } from '@/types/api';

export type BookDraft = ConfigurationDraft<RuntimeDocument['execution']['books'][number]>;
export type Connection = RuntimeDocument['execution']['connections'][number];
export type ConnectionCheck = { fingerprint: string; health: ConnectionHealth };
export const connectionFingerprint = (connection: Connection, updatedAt?: string | null) =>
  JSON.stringify({ connection, updatedAt: updatedAt ?? null });
export const eligibleConnection = (scope: BookDraft['capital_scope'], connection: Connection) =>
  connection.enabled &&
  !connection.canary_only &&
  (scope === 'real' ? connection.environment === 'live' : connection.environment !== 'live');

export function bookErrors(books: BookDraft[], connections: Connection[], message: (key: string) => string) {
  const errors: Record<string, string> = {};
  const ids = new Set<string>();
  const used = new Set<string>();
  books.forEach((book, index) => {
    const path = `execution.books.${index}`;
    if (!book.id.trim() || ids.has(book.id)) errors[`${path}.id`] = message('requiredId');
    ids.add(book.id);
    if (!book.label.trim()) errors[`${path}.label`] = message('required');
    const local = new Set<string>();
    let total = 0;
    book.allocations.forEach((allocation, row) => {
      const field = `${path}.allocations.${row}.weight`;
      const connection = connections.find((item) => item.id === allocation.connection_id);
      if (!connection || !eligibleConnection(book.capital_scope, connection) || local.has(allocation.connection_id))
        errors[field] = message('invalidConnection');
      local.add(allocation.connection_id);
      if (
        typeof allocation.weight !== 'number' ||
        !Number.isFinite(allocation.weight) ||
        allocation.weight < 0 ||
        allocation.weight > 1
      )
        errors[field] = message('numberInvalid');
      if (!allocation.enabled || !book.enabled) return;
      if (used.has(allocation.connection_id)) errors[field] = message('duplicateConnection');
      used.add(allocation.connection_id);
      total += typeof allocation.weight === 'number' ? allocation.weight : 0;
    });
    if (book.enabled && Math.abs(total - 1) > 1e-9) {
      const first = book.allocations.findIndex((item) => item.enabled);
      errors[first < 0 ? `${path}.enabled` : `${path}.allocations.${first}.weight`] ??= message('bookWeight');
    }
  });
  return errors;
}

export function connectionChecksReady(
  document: RuntimeDocument,
  checks: Record<string, ConnectionCheck>,
  credentials: Record<string, { updatedAt: string | null }>,
) {
  const enabled = document.execution.connections.filter((item) => item.enabled);
  return (
    enabled.length > 0 &&
    enabled.every((item) => {
      const check = checks[item.id];
      return (
        check?.health.healthy && check.fingerprint === connectionFingerprint(item, credentials[item.id]?.updatedAt)
      );
    })
  );
}

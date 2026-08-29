import { describe, expect, it } from 'vitest';
import { JsonValueSchema, RuntimeConfigSchema } from '@/types/api.schema';
import { assertRuntimeJsonDocument, decodeJsonValue, toRuntimeDocument } from './use-runtime-config';

describe('runtime config response decoder', () => {
  it('rejects every semantically impossible JsonValue envelope field', () => {
    const base = { kind: 'null', boolean_value: null, number_value: null, string_value: null, datetime_value: null, pair_value: null, items: [], entries: [] };
    expect(JsonValueSchema.safeParse({ ...base, boolean_value: false }).success).toBe(false);
    expect(JsonValueSchema.safeParse({ ...base, kind: 'number', number_value: '1', items: [{}] }).success).toBe(false);
    expect(JsonValueSchema.safeParse({ ...base, kind: 'array', items: [], entries: [{ key: 'bad', value: base }] }).success).toBe(false);
    expect(JsonValueSchema.safeParse({ ...base, kind: 'object', items: [{}] }).success).toBe(false);
  });
  it('recursively converts DTO envelopes into ordinary writable JSON', () => {
    expect(decodeJsonValue({ kind: 'object', boolean_value: null, number_value: null, string_value: null, datetime_value: null, pair_value: null, items: [], entries: [{ key: 'nested', value: { kind: 'array', boolean_value: null, number_value: null, string_value: null, datetime_value: null, pair_value: null, items: [{ kind: 'number', boolean_value: null, number_value: '2.5', string_value: null, datetime_value: null, pair_value: null, items: [], entries: [] }], entries: [] } }] })).toEqual({ nested: [2.5] });
  });
  it('rejects malformed typed envelopes instead of putting them back', () => {
    expect(() => decodeJsonValue({ kind: 'number' } as never)).toThrow('Invalid runtime numeric parameter');
  });
  it('strips credential state while retaining plain connection parameters', () => {
    const document = toRuntimeDocument({ system: { active: false }, market_data: { source_id: 'market', parameters: [] }, llm: { models: {} }, signals: { components: [], neutral_threshold: .2, max_target_ratio: 1, atr_stop_multiplier: 2, reward_ratio: 2, hitl_required: false }, risk: {}, execution: { allocation_policy: 'weighted', books: [], connections: [{ id: 'demo', label: 'Demo', adapter_id: 'okx', environment: 'demo', enabled: true, credential_configured: true, credential_updated_at: null, leverage: 1, margin_mode: 'cross', parameters: [{ key: 'sandbox', value: { kind: 'boolean', boolean_value: true } }] }] }, hitl: {}, scheduler: {}, triggers: {}, notifications: {}, infrastructure: {} } as never);
    expect(document.execution.connections[0]).toEqual(expect.objectContaining({ parameters: { sandbox: true } }));
    expect(document.execution.connections[0]).not.toHaveProperty('credential_configured');
  });
  it('fails closed when a strict runtime response has an unknown field', () => {
    expect(RuntimeConfigSchema.safeParse({ revision: 1, updated_at: '2026-08-28T00:00:00Z', setup_required: true, document: { system: { active: false, injected: true }, market_data: { source_id: 'market', parameters: [] }, llm: { models: {} }, signals: { components: [] }, risk: {}, execution: { connections: [], books: [], allocation_policy: 'weighted' }, hitl: {}, scheduler: {}, triggers: {}, notifications: {}, infrastructure: {} } }).success).toBe(false);
  });

  it.each([
    ['undefined', { nested: undefined }],
    ['function', { nested: () => 'lost' }],
    ['NaN', { nested: Number.NaN }],
    ['infinity', { nested: Infinity }],
    ['nested invalid value', { nested: [{ still: Number.NEGATIVE_INFINITY }] }],
  ])('rejects runtime JSON containing %s before it can be serialized', (_name, value) => {
    expect(() => assertRuntimeJsonDocument(value)).toThrow('Invalid runtime JSON');
  });

  it('accepts recursive ordinary JSON without rewriting it', () => {
    const value = { object: { array: [null, true, 1.25, 'ok', { deeper: ['yes'] }] } };
    expect(assertRuntimeJsonDocument(value)).toBe(value);
  });
});

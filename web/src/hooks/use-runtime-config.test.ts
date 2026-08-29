import { describe, expect, it } from 'vitest';
import { decodeJsonValue, toRuntimeDocument } from './use-runtime-config';

describe('runtime config response decoder', () => {
  it('recursively converts DTO envelopes into ordinary writable JSON', () => {
    expect(decodeJsonValue({ kind: 'object', entries: [{ key: 'nested', value: { kind: 'array', items: [{ kind: 'number', number_value: '2.5' }] } }] })).toEqual({ nested: [2.5] });
  });
  it('rejects malformed typed envelopes instead of putting them back', () => {
    expect(() => decodeJsonValue({ kind: 'number' } as never)).toThrow('Invalid runtime numeric parameter');
  });
  it('strips credential state while retaining plain connection parameters', () => {
    const document = toRuntimeDocument({ system: { active: false }, market_data: { source_id: 'market', parameters: [] }, llm: { models: {} }, signals: { components: [], neutral_threshold: .2, max_target_ratio: 1, atr_stop_multiplier: 2, reward_ratio: 2, hitl_required: false }, risk: {}, execution: { allocation_policy: 'weighted', books: [], connections: [{ id: 'demo', label: 'Demo', adapter_id: 'okx', environment: 'demo', enabled: true, credential_configured: true, credential_updated_at: null, leverage: 1, margin_mode: 'cross', parameters: [{ key: 'sandbox', value: { kind: 'boolean', boolean_value: true } }] }] }, hitl: {}, scheduler: {}, triggers: {}, notifications: {}, infrastructure: {} });
    expect(document.execution.connections[0]).toEqual(expect.objectContaining({ parameters: { sandbox: true } }));
    expect(document.execution.connections[0]).not.toHaveProperty('credential_configured');
  });
});

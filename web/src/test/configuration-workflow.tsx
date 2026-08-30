import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render } from '@testing-library/react';
import { MemoryRouter } from 'react-router';
import { vi } from 'vitest';
import { App } from '@/App';
import { runtimeConfigFixture } from './runtime-config-fixture';
import { configurationCatalogFixture, pluginFields } from './configuration-catalog-fixture';
import type { ConfigurationCatalog, RuntimeConfig, RuntimeDocument, RuntimeJsonValue } from '@/types/api';
import type { JsonValueOut } from '@/types/api.schema';
import { MarketDataContext, type MarketDataContextValue } from '@/contexts/market-data/market-data-context';

const offlineMarket: MarketDataContextValue = {
  connectionStatus: 'disconnected',
  subscribe: () => {},
  unsubscribe: () => {},
  getPrice: () => undefined,
  subscribeToPrice: () => () => {},
};

export const workflowCatalog: ConfigurationCatalog = {
  ...configurationCatalogFixture,
  components: ['kronos', 'llm_committee'].map((id) => ({
    ...configurationCatalogFixture.components[0]!,
    id,
    label: { zh_CN: id === 'kronos' ? 'Kronos' : 'LLM 委员会', en_US: id },
    fields: [],
  })),
  venues: [
    {
      id: 'paper',
      label: { zh_CN: 'Paper 模拟账户', en_US: 'Paper' },
      description: { zh_CN: '', en_US: '' },
      environments: ['paper'],
      credential_fields: [],
      margin_modes: ['cross'],
      fields: [
        {
          ...pluginFields[0]!,
          key: 'initial_equity',
          label: { zh_CN: '模拟初始资金', en_US: 'Initial simulated equity' },
          default_value: { ...pluginFields[0]!.default_value, number_value: '10000' },
          minimum: null,
          exclusive_minimum: 0,
          maximum: null,
          unit: 'USDT',
        },
      ],
    },
    ...['okx', 'bybit'].map((id) => ({
      id,
      label: { zh_CN: id.toUpperCase(), en_US: id.toUpperCase() },
      description: { zh_CN: '', en_US: '' },
      fields: [],
      environments: id === 'okx' ? ['demo', 'live'] : ['demo', 'testnet', 'live'],
      credential_fields: id === 'okx' ? ['api_key', 'secret', 'passphrase'] : ['api_key', 'secret'],
      margin_modes: ['cross', 'isolated'] as ('cross' | 'isolated')[],
    })),
  ],
};
export const paperConnection = {
  id: 'paper',
  label: 'Paper',
  adapter_id: 'paper',
  environment: 'paper' as const,
  enabled: true,
  canary_only: false,
  credential_configured: false,
  credential_updated_at: null,
  leverage: 1,
  margin_mode: 'cross' as const,
  parameters: [],
};
export function workflowConfig() {
  const base = runtimeConfigFixture();
  return runtimeConfigFixture({
    setup_required: true,
    document: {
      ...base.document,
      system: { active: false },
      market_data: { news_credential_configured: false, news_credential_updated_at: null, source_id: 'default', parameters: [] },
      signals: {
        ...base.document.signals,
        components: [{ component_id: 'kronos', enabled: true, weight: 1, parameters: [] }],
      },
      execution: {
        ...base.document.execution,
        connections: [paperConnection],
        books: [
          {
            id: 'sim',
            label: '模拟主账户',
            capital_scope: 'simulated',
            enabled: true,
            hitl_required: true,
            allocations: [{ connection_id: 'paper', enabled: true, weight: 1 }],
          },
        ],
      },
    },
  });
}
function encode(value: RuntimeJsonValue): JsonValueOut {
  const empty: JsonValueOut = {
    kind: 'null',
    boolean_value: null,
    number_value: null,
    string_value: null,
    datetime_value: null,
    pair_value: null,
    items: [],
    entries: [],
  };
  if (value === null) return empty;
  if (typeof value === 'number') return { ...empty, kind: 'number', number_value: String(value) };
  if (typeof value === 'string') return { ...empty, kind: 'string', string_value: value };
  if (typeof value === 'boolean') return { ...empty, kind: 'boolean', boolean_value: value };
  if (Array.isArray(value)) return { ...empty, kind: 'array', items: value.map(encode) };
  return {
    ...empty,
    kind: 'object',
    entries: Object.entries(value).map(([key, item]) => ({ key, value: encode(item) })),
  };
}
export function workflowHarness(path = '/setup', initial = workflowConfig(), requiredAccess?: string) {
  let saved = initial;
  const writes: { expected_revision: number; document: RuntimeDocument }[] = [];
  let failure: { status: number; detail: unknown } | undefined;
  let failReload = false;
  let failApply = false;
  const response = (value: unknown, status = 200) => Promise.resolve(new Response(JSON.stringify(value), { status }));
  const fetchMock = vi.fn((url: string, init?: RequestInit) => {
    if (requiredAccess && new Headers(init?.headers).get('X-API-Key') !== requiredAccess) {
      return response({ detail: 'Invalid API key' }, 401);
    }
    if (url.endsWith('/api/config/catalog')) return response(workflowCatalog);
    if (url.endsWith('/api/scheduler/status'))
      return response({ enabled: false, next_pair: null, next_run_at: null, redis_available: true });
    if (url.includes('/api/config/credentials/')) {
      if (failure) return response({ detail: failure.detail }, failure.status);
      const gateway = url.endsWith('/llm-gateway');
      const news = url.endsWith('/news-provider');
      saved = {
        ...saved,
        revision: saved.revision + 1,
        applied_revision: saved.revision + 1,
        apply_status: 'applied',
        document: {
          ...saved.document,
          market_data: { ...saved.document.market_data, ...(news ? { news_credential_configured: true, news_credential_updated_at: '2026-08-30T13:00:00Z' } : {}) },
          llm: {
            ...saved.document.llm,
            ...(gateway
              ? { gateway_credential_configured: true, gateway_credential_updated_at: '2026-08-30T13:00:00Z' }
              : {}),
          },
          security: {
            ...saved.document.security,
            ...(!gateway && !news
              ? { access_credential_configured: true, access_credential_updated_at: '2026-08-30T13:00:00Z' }
              : {}),
          },
        },
      };
      return response({ revision: saved.revision, configured: true, updated_at: '2026-08-30T13:00:00Z' });
    }
    if (url.endsWith('/api/config') && init?.method === 'PUT') {
      const body = JSON.parse(init.body as string) as (typeof writes)[number];
      writes.push(body);
      if (failure) return response({ detail: failure.detail }, failure.status);
      const doc = body.document;
      saved = runtimeConfigFixture({
        ...saved,
        revision: saved.revision + 1,
        setup_required: !doc.system.active,
        apply_status: failApply ? 'failed' : 'applied',
        applied_revision: failApply ? saved.applied_revision : saved.revision + 1,
        document: {
          ...doc,
          security: {
            ...doc.security,
            access_credential_configured: saved.document.security.access_credential_configured,
            access_credential_updated_at: saved.document.security.access_credential_updated_at,
          },
          llm: {
            ...doc.llm,
            gateway_credential_configured: saved.document.llm.gateway_credential_configured,
            gateway_credential_updated_at: saved.document.llm.gateway_credential_updated_at,
          },
          signals: {
            ...doc.signals,
            components: doc.signals.components.map((item) => ({
              ...item,
              parameters: encode(item.parameters).entries,
            })),
          },
          market_data: { ...saved.document.market_data, ...doc.market_data, parameters: encode(doc.market_data.parameters).entries },
          execution: {
            ...doc.execution,
            connections: doc.execution.connections.map((item) => ({
              ...item,
              credential_configured:
                saved.document.execution.connections.find((c) => c.id === item.id)?.credential_configured ?? false,
              credential_updated_at:
                saved.document.execution.connections.find((c) => c.id === item.id)?.credential_updated_at ?? null,
              parameters: encode(item.parameters).entries,
            })),
          },
        },
      });
      return failApply ? response({ detail: 'Runtime configuration cannot be applied' }, 503) : response(saved);
    }
    if (url.endsWith('/test')) {
      const id = url.split('/').at(-2)!;
      const connection = saved.document.execution.connections.find((item) => item.id === id)!;
      if (failure) return response({ detail: failure.detail }, failure.status);
      return response({
        connection_id: id,
        checked_at: '2026-08-30T12:00:00Z',
        healthy: true,
        environment: connection.environment,
        credential_configured: connection.credential_configured,
        capabilities: {
          market_types: [],
          native_protection: false,
          hedge_mode: false,
          reduce_only: true,
          supported_order_types: [],
        },
      });
    }
    if (url.includes('/venue-connections/') && url.endsWith('/credentials')) {
      if (failure) return response({ detail: failure.detail }, failure.status);
      const id = url.split('/').at(-2)!;
      saved = {
        ...saved,
        revision: saved.revision + 1,
        document: {
          ...saved.document,
          execution: {
            ...saved.document.execution,
            connections: saved.document.execution.connections.map((item) =>
              item.id === id
                ? { ...item, credential_configured: true, credential_updated_at: '2026-08-30T13:00:00Z' }
                : item,
            ),
          },
        },
      };
      return response({
        revision: saved.revision,
        credential: { configured: true, updated_at: '2026-08-30T13:00:00Z' },
      });
    }
    if (url.includes('/api/venue-connections') && (init?.method === 'POST' || init?.method === 'PUT')) {
      const body = JSON.parse(init.body as string) as RuntimeDocument['execution']['connections'][number] & {
        expected_revision: number;
      };
      const { expected_revision: _revision, ...input } = body;
      const existing = saved.document.execution.connections.find((item) => item.id === body.id);
      const connection = {
        ...input,
        credential_configured: existing?.credential_configured ?? false,
        credential_updated_at: existing?.credential_updated_at ?? null,
        parameters: encode(body.parameters).entries,
      };
      saved = {
        ...saved,
        revision: saved.revision + 1,
        document: {
          ...saved.document,
          execution: {
            ...saved.document.execution,
            connections: saved.document.execution.connections.some((c) => c.id === body.id)
              ? saved.document.execution.connections.map((c) => (c.id === body.id ? connection : c))
              : [...saved.document.execution.connections, connection],
          },
        },
      };
      return response({ revision: saved.revision, connection });
    }
    if (url.endsWith('/api/config')) return failReload ? response({ detail: 'unavailable' }, 503) : response(saved);
    return response({});
  });
  vi.stubGlobal('fetch', fetchMock);
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
  const view = render(
    <QueryClientProvider client={client}>
      <MarketDataContext.Provider value={offlineMarket}>
        <MemoryRouter initialEntries={[path]}>
          <App />
        </MemoryRouter>
      </MarketDataContext.Provider>
    </QueryClientProvider>,
  );
  return {
    ...view,
    client,
    fetchMock,
    writes,
    saved: () => saved,
    setSaved: (value: RuntimeConfig) => {
      saved = value;
    },
    fail: (status: number, detail: unknown) => {
      failure = { status, detail };
    },
    clearFailure: () => {
      failure = undefined;
    },
    failReload: (value: boolean) => {
      failReload = value;
    },
    failApply: (value: boolean) => {
      failApply = value;
    },
  };
}

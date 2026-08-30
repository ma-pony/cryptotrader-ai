import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import '@/lib/i18n';
import { runtimeConfigFixture } from '@/test/runtime-config-fixture';
import { configurationCatalogFixture } from '@/test/configuration-catalog-fixture';
import { useConfigurationDraft, validateConfigurationSection } from './use-configuration-draft';
import { RUNTIME_CONFIG_QUERY_KEY, toRuntimeDocument } from './use-runtime-config';

afterEach(() => vi.unstubAllGlobals());

it.each([
  ['exclusive_minimum', 0, false],
  ['exclusive_minimum', 0.00001, true],
  ['minimum', 0, true],
  ['exclusive_maximum', 0, false],
  ['exclusive_maximum', -0.00001, true],
  ['maximum', 0, true],
] as const)('validates %s against %s with inclusive and exclusive semantics', (bound, value, valid) => {
  const source = configurationCatalogFixture.market_sources[0]!;
  const catalog = { ...configurationCatalogFixture, market_sources: [{ ...source, fields: [{
    ...source.fields[0]!, minimum: null, maximum: null, exclusive_minimum: null, exclusive_maximum: null, [bound]: 0,
  }] }] };
  const document = toRuntimeDocument(runtimeConfigFixture().document);
  document.market_data = { source_id: 'default', parameters: { threshold: value } };
  expect(validateConfigurationSection('market', document, catalog, (key) => key)).toEqual(
    valid ? {} : { 'market_data.parameters.threshold': 'numberInvalid' },
  );
});

function harness() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
  client.setQueryData(RUNTIME_CONFIG_QUERY_KEY, runtimeConfigFixture());
  const hook = renderHook(() => useConfigurationDraft(configurationCatalogFixture), {
    wrapper: ({ children }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>,
  });
  return { client, ...hook };
}

it('saves only the selected section over the latest baseline and retains other pending sections', async () => {
  const { result, client } = harness();
  act(() => {
    result.current.update('llm', { ...result.current.document!.llm, base_url: 'https://new.example' });
    result.current.update('risk', {
      ...result.current.document!.risk,
      loss: { ...result.current.document!.risk.loss, max_drawdown_pct: 0.05 },
    });
  });
  const base = runtimeConfigFixture();
  act(() => {
    client.setQueryData(
      RUNTIME_CONFIG_QUERY_KEY,
      runtimeConfigFixture({
        revision: 2,
        document: { ...base.document, infrastructure: { redis_url: 'redis://newhost:6379/0' } },
      }),
    );
  });
  let submitted: Record<string, unknown> | undefined;
  vi.stubGlobal(
    'fetch',
    vi.fn().mockImplementation((_url, init: RequestInit) => {
      submitted = JSON.parse(init.body as string);
      return Promise.resolve(
        new Response(
          JSON.stringify(
            runtimeConfigFixture({
              revision: 3,
              document: {
                ...base.document,
                llm: { ...base.document.llm, base_url: 'https://new.example' },
                infrastructure: { redis_url: 'redis://newhost:6379/0' },
              },
            }),
          ),
          { status: 200 },
        ),
      );
    }),
  );
  await act(async () => {
    await result.current.save('models');
  });
  expect(submitted).toMatchObject({
    expected_revision: 2,
    document: {
      llm: { base_url: 'https://new.example' },
      risk: { loss: { max_drawdown_pct: 0.15 } },
      infrastructure: { redis_url: 'redis://newhost:6379/0' },
    },
  });
  expect(result.current.isDirty('models')).toBe(false);
  expect(result.current.isDirty('risk')).toBe(true);
  expect(result.current.document!.risk.loss.max_drawdown_pct).toBe(0.05);
  expect(result.current.status.models).toBe('saved');
  act(() => result.current.update('llm', { ...result.current.document!.llm, timeout: 60 }));
  expect(result.current.status.models).toBeUndefined();
  expect(result.current.isDirty('models')).toBe(true);
});

it('rejects a cleared numeric field without sending a request and preserves it after failed saves', async () => {
  const { result } = harness();
  const fetchMock = vi
    .fn()
    .mockResolvedValue(
      new Response(JSON.stringify({ detail: 'Runtime configuration changed; reload and retry' }), { status: 409 }),
    );
  vi.stubGlobal('fetch', fetchMock);
  act(() =>
    result.current.update('risk', {
      ...result.current.document!.risk,
      loss: { ...result.current.document!.risk.loss, max_drawdown_pct: '' },
    }),
  );
  await act(async () => {
    expect(await result.current.save('risk')).toBe(false);
  });
  expect(fetchMock).not.toHaveBeenCalled();
  expect(result.current.errors['risk.loss.max_drawdown_pct']).toBeTruthy();
  expect(result.current.document!.risk.loss.max_drawdown_pct).toBe('');
  act(() =>
    result.current.update('risk', {
      ...result.current.document!.risk,
      loss: { ...result.current.document!.risk.loss, max_drawdown_pct: 0.05 },
    }),
  );
  await act(async () => {
    expect(await result.current.save('risk')).toBe(false);
  });
  expect(result.current.isDirty('risk')).toBe(true);
  expect(result.current.document!.risk.loss.max_drawdown_pct).toBe(0.05);
  await waitFor(() => expect(result.current.conflict).toBe(true));
});

it('reloads latest saved values while retaining edits unless discard is explicitly confirmed', async () => {
  const { result } = harness();
  act(() => result.current.update('hitl', { approval_ttl_minutes: 45 }));
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue(new Response(JSON.stringify(runtimeConfigFixture({ revision: 2 })), { status: 200 })),
  );
  await act(async () => {
    await result.current.reload();
  });
  expect(result.current.document!.hitl.approval_ttl_minutes).toBe(45);
  act(() => result.current.discard('risk'));
  expect(result.current.document!.hitl.approval_ttl_minutes).toBe(15);
});

it('uses a refreshed baseline after an edit is restored while preserving another dirty section', async () => {
  const { result } = harness();
  const originalRisk = result.current.document!.risk;
  act(() => {
    result.current.update('risk', {
      ...originalRisk,
      loss: { ...originalRisk.loss, max_drawdown_pct: 0.05 },
    });
    result.current.update('llm', { ...result.current.document!.llm, base_url: 'https://pending.example' });
  });
  expect(result.current.isDirty('risk')).toBe(true);
  act(() => result.current.update('risk', structuredClone(originalRisk)));
  expect(result.current.isDirty('risk')).toBe(false);
  const base = runtimeConfigFixture();
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify(
          runtimeConfigFixture({
            revision: 2,
            document: {
              ...base.document,
              risk: { ...base.document.risk, loss: { ...base.document.risk.loss, max_drawdown_pct: 0.2 } },
            },
          }),
        ),
        { status: 200 },
      ),
    ),
  );
  await act(async () => {
    await result.current.reload();
  });
  expect(result.current.document!.risk.loss.max_drawdown_pct).toBe(0.2);
  expect(result.current.isDirty('risk')).toBe(false);
  expect(result.current.document!.llm.base_url).toBe('https://pending.example');
  expect(result.current.isDirty('models')).toBe(true);
});

it('keeps edits and reports a failed explicit reload', async () => {
  const { result } = harness();
  act(() => result.current.update('hitl', { approval_ttl_minutes: 45 }));
  vi.stubGlobal(
    'fetch',
    vi
      .fn()
      .mockResolvedValue(
        new Response(JSON.stringify({ detail: 'Runtime configuration is unavailable' }), { status: 503 }),
      ),
  );
  await act(async () => {
    await result.current.reload();
  });
  expect(result.current.document!.hitl.approval_ttl_minutes).toBe(45);
  expect(result.current.failure).toBeTruthy();
});

it('compares restored book ownership without stale connections and retains other drafts', async () => {
  const { result, client } = harness();
  const original = result.current.document!.execution;
  act(() => {
    result.current.update('execution', { ...original, live_order_execution_enabled: true });
    result.current.update('llm', { ...result.current.document!.llm, base_url: 'https://pending.example' });
  });
  const base = runtimeConfigFixture();
  act(() => {
    client.setQueryData(RUNTIME_CONFIG_QUERY_KEY, {
      ...base,
      revision: 2,
      document: {
        ...base.document,
        execution: {
          ...base.document.execution,
          connections: [
            {
              id: 'paper',
              label: 'Newest connection',
              adapter_id: 'paper',
              environment: 'paper',
              enabled: true,
              canary_only: false,
              leverage: 1,
              margin_mode: 'cross',
              credential_configured: false,
              credential_updated_at: null,
              parameters: [],
            },
          ],
        },
      },
    });
  });
  await waitFor(() => expect(result.current.document!.execution.connections[0]?.label).toBe('Newest connection'));
  act(() => result.current.update('execution', original));
  expect(result.current.isDirty('books')).toBe(false);
  expect(result.current.document!.execution.connections[0]?.label).toBe('Newest connection');
  expect(result.current.isDirty('models')).toBe(true);
});

it.each([90, 15])('does not discard an edit to %s made while an earlier version is saving', async (minutes) => {
  const { result } = harness();
  let finish: (response: Response) => void = () => undefined;
  vi.stubGlobal(
    'fetch',
    vi.fn().mockImplementation(
      () =>
        new Promise<Response>((resolve) => {
          finish = resolve;
        }),
    ),
  );
  act(() => result.current.update('hitl', { approval_ttl_minutes: 45 }));
  let saving: Promise<boolean>;
  act(() => {
    saving = result.current.save('risk');
  });
  await waitFor(() => expect(result.current.isSaving).toBe(true));
  act(() => result.current.update('hitl', { approval_ttl_minutes: minutes }));
  const base = runtimeConfigFixture();
  await act(async () => {
    finish(
      new Response(
        JSON.stringify(
          runtimeConfigFixture({ revision: 2, document: { ...base.document, hitl: { approval_ttl_minutes: 45 } } }),
        ),
        { status: 200 },
      ),
    );
    await saving;
  });
  expect(result.current.document!.hitl.approval_ttl_minutes).toBe(minutes);
  expect(result.current.isDirty('risk')).toBe(true);
  expect(result.current.status.risk).toBeUndefined();
});

it('releases a restored baseline edit when the in-flight save fails', async () => {
  const { result, client } = harness();
  let finish: (response: Response) => void = () => undefined;
  vi.stubGlobal(
    'fetch',
    vi.fn().mockImplementation(
      () =>
        new Promise<Response>((resolve) => {
          finish = resolve;
        }),
    ),
  );
  act(() => result.current.update('hitl', { approval_ttl_minutes: 45 }));
  let saving: Promise<boolean>;
  act(() => {
    saving = result.current.save('risk');
  });
  await waitFor(() => expect(result.current.isSaving).toBe(true));
  act(() => result.current.update('hitl', { approval_ttl_minutes: 15 }));
  await act(async () => {
    finish(new Response(JSON.stringify({ detail: 'Unavailable' }), { status: 503 }));
    expect(await saving).toBe(false);
  });
  expect(result.current.isDirty('risk')).toBe(false);
  const base = runtimeConfigFixture();
  act(() => {
    client.setQueryData(
      RUNTIME_CONFIG_QUERY_KEY,
      runtimeConfigFixture({ revision: 2, document: { ...base.document, hitl: { approval_ttl_minutes: 30 } } }),
    );
  });
  await waitFor(() => expect(result.current.document!.hitl.approval_ttl_minutes).toBe(30));
  expect(result.current.isDirty('risk')).toBe(false);
});

it('leaves optional factory-default lists sparse but rejects an explicitly blank numeric parameter', () => {
  const number = configurationCatalogFixture.market_sources[0]!.fields[0]!;
  const catalog = {
    ...configurationCatalogFixture,
    market_sources: [
      {
        ...configurationCatalogFixture.market_sources[0]!,
        fields: [
          number,
          {
            ...number,
            key: 'symbols',
            kind: 'string_list' as const,
            default_value: { ...number.default_value, kind: 'null' as const, number_value: null },
          },
        ],
      },
    ],
  };
  const document = toRuntimeDocument(runtimeConfigFixture().document);
  document.market_data = { source_id: 'default', parameters: {} };
  expect(validateConfigurationSection('market', document, catalog, (key) => key)).toEqual({});
  document.market_data.parameters.threshold = '';
  expect(validateConfigurationSection('market', document, catalog, (key) => key)).toEqual({
    'market_data.parameters.threshold': 'numberInvalid',
  });
});

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import '@/lib/i18n';
import { runtimeConfigFixture } from '@/test/runtime-config-fixture';
import { configurationCatalogFixture } from '@/test/configuration-catalog-fixture';
import { useConfigurationDraft, validateConfigurationSection } from './use-configuration-draft';
import { RUNTIME_CONFIG_QUERY_KEY, toRuntimeDocument } from './use-runtime-config';

afterEach(() => vi.unstubAllGlobals());

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

it('does not discard an edit made while an earlier version of the section is saving', async () => {
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
  act(() => result.current.update('hitl', { approval_ttl_minutes: 90 }));
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
  expect(result.current.document!.hitl.approval_ttl_minutes).toBe(90);
  expect(result.current.isDirty('risk')).toBe(true);
  expect(result.current.status.risk).toBeUndefined();
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

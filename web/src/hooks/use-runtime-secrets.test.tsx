import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { renderHook } from '@testing-library/react';
import { expect, it, vi } from 'vitest';

import { useRuntimeSecrets } from './use-runtime-secrets';
import { RUNTIME_CONFIG_QUERY_KEY } from './use-runtime-config';
import { runtimeConfigFixture } from '@/test/runtime-config-fixture';
import { useSettingsStore } from '@/stores/use-settings-store';
import { RUNTIME_CONFIG_CONFLICT_QUERY_KEY } from './runtime-config-conflict';

it('writes an LLM token directly, advances config revision, and leaves mutation cache secret-free', async () => {
  const token = 'gateway-test-token';
  const base = runtimeConfigFixture();
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(new Response(JSON.stringify({ revision: 2, configured: true, updated_at: '2026-08-29T00:00:00Z' }), { status: 200 }))
    .mockResolvedValueOnce(new Response(JSON.stringify({ ...base, revision: 2, applied_revision: 2, document: { ...base.document, llm: { ...base.document.llm, gateway_credential_configured: true } } }), { status: 200 }));
  vi.stubGlobal('fetch', fetchMock);
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  client.setQueryData(RUNTIME_CONFIG_QUERY_KEY, runtimeConfigFixture());
  const hook = renderHook(() => useRuntimeSecrets(), { wrapper: ({ children }) => <QueryClientProvider client={client}>{children}</QueryClientProvider> });

  await hook.result.current.writeLlmGateway(1, token);

  expect(JSON.parse((fetchMock.mock.calls[0]![1] as RequestInit).body as string)).toEqual({ expected_revision: 1, token });
  expect(client.getQueryData(RUNTIME_CONFIG_QUERY_KEY)).toMatchObject({ revision: 2, applied_revision: 2, apply_status: 'applied', document: { llm: { gateway_credential_configured: true } } });
  expect(JSON.stringify(client.getMutationCache().getAll().map((mutation) => mutation.state.variables))).not.toContain(token);
  expect(JSON.stringify(client.getQueryCache().getAll().map((query) => query.state))).not.toContain(token);
});

it.each([false, true])('hands off the acknowledged API key before refresh and reports refresh failure=%s safely', async (failRefresh) => {
  useSettingsStore.getState().setApiKey('old-key-marker');
  const token = 'new-key-marker';
  const fetchMock = vi.fn().mockImplementation((url: string, init: RequestInit) => {
    const credentialWrite = url.endsWith('/api-access');
    expect(new Headers(init.headers).get('X-API-Key')).toBe(credentialWrite ? 'old-key-marker' : token);
    return Promise.resolve(new Response(JSON.stringify(credentialWrite
      ? { revision: 2, configured: true, updated_at: '2026-08-30T00:00:00Z' }
      : failRefresh ? { detail: 'unavailable' } : runtimeConfigFixture({ revision: 2, applied_revision: 2 })),
    { status: !credentialWrite && failRefresh ? 503 : 200 }));
  });
  vi.stubGlobal('fetch', fetchMock);
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
  client.setQueryData(RUNTIME_CONFIG_QUERY_KEY, runtimeConfigFixture());
  const hook = renderHook(() => useRuntimeSecrets(), { wrapper: ({ children }) => <QueryClientProvider client={client}>{children}</QueryClientProvider> });
  const saved = await hook.result.current.writeApiAccess(1, token);
  expect(saved).toMatchObject({ revision: 2, savedNeedsReload: failRefresh });
  expect(fetchMock.mock.calls.filter(([url]) => url.endsWith('/api/config'))).toHaveLength(1);
  expect(useSettingsStore.getState().apiKey).toBe(token);
  if (failRefresh) expect(client.getQueryData(RUNTIME_CONFIG_CONFLICT_QUERY_KEY)).toBe(true);
  else expect(client.getQueryData(RUNTIME_CONFIG_QUERY_KEY)).toMatchObject({ applied_revision: 2, apply_status: 'applied' });
  expect(JSON.stringify(client.getQueryCache().getAll().map((query) => query.state))).not.toContain(token);
  expect(client.getMutationCache().getAll()).toHaveLength(0);
  expect(JSON.stringify(localStorage)).not.toContain(token);
  useSettingsStore.getState().reset();
});

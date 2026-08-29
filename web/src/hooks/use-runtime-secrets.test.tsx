import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { renderHook } from '@testing-library/react';
import { expect, it, vi } from 'vitest';

import { useRuntimeSecrets } from './use-runtime-secrets';
import { RUNTIME_CONFIG_QUERY_KEY } from './use-runtime-config';
import { runtimeConfigFixture } from '@/test/runtime-config-fixture';

it('writes an LLM token directly, advances config revision, and leaves mutation cache secret-free', async () => {
  const token = 'gateway-test-token';
  const fetchMock = vi.fn().mockResolvedValueOnce(new Response(JSON.stringify({ revision: 2, configured: true, updated_at: '2026-08-29T00:00:00Z' }), { status: 200 }));
  vi.stubGlobal('fetch', fetchMock);
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  client.setQueryData(RUNTIME_CONFIG_QUERY_KEY, runtimeConfigFixture());
  const hook = renderHook(() => useRuntimeSecrets(), { wrapper: ({ children }) => <QueryClientProvider client={client}>{children}</QueryClientProvider> });

  await hook.result.current.writeLlmGateway(1, token);

  expect(JSON.parse((fetchMock.mock.calls[0]![1] as RequestInit).body as string)).toEqual({ expected_revision: 1, token });
  expect(client.getQueryData(RUNTIME_CONFIG_QUERY_KEY)).toMatchObject({ revision: 2, document: { llm: { gateway_credential_configured: true } } });
  expect(JSON.stringify(client.getMutationCache().getAll().map((mutation) => mutation.state.variables))).not.toContain(token);
});

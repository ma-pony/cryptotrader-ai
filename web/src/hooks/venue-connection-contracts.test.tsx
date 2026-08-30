import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { renderHook } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import { useVenueConnections } from './use-venue-connections';
import { RUNTIME_CONFIG_QUERY_KEY } from './use-runtime-config';
import { RUNTIME_CONFIG_CONFLICT_QUERY_KEY } from './runtime-config-conflict';
import { workflowConfig } from '@/test/configuration-workflow';

it('retains a successful create when follow-up refresh fails and marks the shared conflict', async () => {
  const initial = workflowConfig();
  const connection = { ...initial.document.execution.connections[0]!, id: 'new-paper', label: 'New Paper' };
  vi.stubGlobal(
    'fetch',
    vi
      .fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ revision: 2, connection }), { status: 201 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ detail: 'unavailable' }), { status: 503 })),
  );
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  client.setQueryData(RUNTIME_CONFIG_QUERY_KEY, initial);
  const { result } = renderHook(() => useVenueConnections(), {
    wrapper: ({ children }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>,
  });
  const saved = await result.current.create.mutateAsync({
    ...connection,
    expected_revision: 1,
    parameters: { initial_equity: 10000 },
  });
  expect(saved.savedNeedsReload).toBe(true);
  expect(client.getQueryData(RUNTIME_CONFIG_QUERY_KEY)).toMatchObject({
    revision: 2,
    document: { execution: { connections: [{ id: 'paper' }, { id: 'new-paper' }] } },
  });
  expect(client.getQueryData(RUNTIME_CONFIG_CONFLICT_QUERY_KEY)).toBe(true);
});
it('sends credentials outside mutation storage and preserves acknowledged metadata if refresh fails', async () => {
  const request = vi
    .fn()
    .mockResolvedValueOnce(
      new Response(
        JSON.stringify({ revision: 2, credential: { configured: true, updated_at: '2026-08-30T13:00:00Z' } }),
        { status: 200 },
      ),
    )
    .mockResolvedValueOnce(new Response(JSON.stringify({ detail: 'unavailable' }), { status: 503 }));
  vi.stubGlobal('fetch', request);
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  client.setQueryData(RUNTIME_CONFIG_QUERY_KEY, workflowConfig());
  const { result } = renderHook(() => useVenueConnections(), {
    wrapper: ({ children }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>,
  });
  const saved = await result.current.putCredentials({
    id: 'paper',
    expectedRevision: 1,
    credentials: { api_key: 'fake-key', secret: 'fake-signing' }, // pragma: allowlist secret -- test fixture
  });
  expect(saved.savedNeedsReload).toBe(true);
  expect(JSON.parse((request.mock.calls[0]![1] as RequestInit).body as string)).toEqual({
    expected_revision: 1,
    credentials: { api_key: 'fake-key', secret: 'fake-signing' }, // pragma: allowlist secret -- test fixture
  });
  expect(client.getQueryData(RUNTIME_CONFIG_QUERY_KEY)).toMatchObject({
    revision: 2,
    document: {
      execution: { connections: [{ credential_configured: true, credential_updated_at: '2026-08-30T13:00:00Z' }] },
    },
  });
  expect(client.getMutationCache().getAll()).toHaveLength(0);
});

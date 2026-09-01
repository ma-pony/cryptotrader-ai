import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, renderHook, waitFor } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import type { ReactNode } from 'react';
import { useConnectionChecks } from './use-connection-checks';
import type { Connection } from '@/lib/configuration-readiness';

const connection: Connection = {
  id: 'okx',
  label: 'OKX',
  adapter_id: 'okx',
  environment: 'demo',
  enabled: true,
  leverage: 1,
  margin_mode: 'cross',
  canary_only: false,
  parameters: {},
};
const health = {
  connection_id: 'okx',
  healthy: true,
  checked_at: '2026-08-30T12:00:00Z',
  environment: 'demo',
  credential_configured: true,
  capabilities: null,
  error_code: null,
};
function wrapper() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  );
}

it('loads only persisted checks on mount and runs a new check only when explicitly requested', async () => {
  const fetch = vi.fn((_url: string, init?: RequestInit) =>
    Promise.resolve(new Response(JSON.stringify(init?.method === 'POST' ? health : null))),
  );
  vi.stubGlobal('fetch', fetch);
  const { result, rerender } = renderHook(
    ({ updatedAt }) =>
      useConnectionChecks([connection], {
        okx: { configured: true, updatedAt },
      }),
    { wrapper: wrapper(), initialProps: { updatedAt: 'first' } },
  );
  await waitFor(() => expect(fetch.mock.calls.filter(([, init]) => init?.method === 'GET')).toHaveLength(1));
  expect(fetch.mock.calls.filter(([, init]) => init?.method === 'POST')).toHaveLength(0);
  await act(() => result.current.checkConnection('okx'));
  await waitFor(() => expect(result.current.checks.okx?.health.healthy).toBe(true));
  expect(fetch.mock.calls.filter(([, init]) => init?.method === 'POST')).toHaveLength(1);
  rerender({ updatedAt: 'rotated' });
  await waitFor(() => expect(fetch.mock.calls.filter(([, init]) => init?.method === 'GET')).toHaveLength(2));
  expect(fetch.mock.calls.filter(([, init]) => init?.method === 'POST')).toHaveLength(1);
});

it('runs an explicit POST even while the matching persisted GET is still pending', async () => {
  let resolveGet: ((value: Response) => void) | undefined;
  const fetch = vi.fn((_url: string, init?: RequestInit) => {
    if (init?.method === 'POST') return Promise.resolve(new Response(JSON.stringify(health)));
    return new Promise<Response>((resolve) => {
      resolveGet = resolve;
    });
  });
  vi.stubGlobal('fetch', fetch);
  const { result } = renderHook(
    () => useConnectionChecks([connection], { okx: { configured: true, updatedAt: 'saved' } }),
    { wrapper: wrapper() },
  );
  await waitFor(() => expect(fetch.mock.calls.filter(([, init]) => init?.method === 'GET')).toHaveLength(1));
  try {
    act(() => {
      void result.current.checkConnection('okx');
    });
    await waitFor(() => expect(fetch.mock.calls.filter(([, init]) => init?.method === 'POST')).toHaveLength(1));
  } finally {
    resolveGet?.(new Response(JSON.stringify(null)));
  }
  await waitFor(() => expect(result.current.checks.okx?.health.healthy).toBe(true));
});

it('loads persisted checks for enabled environments that declare no credentials', async () => {
  const fetch = vi.fn((_url: string, _init?: RequestInit) => Promise.resolve(new Response(JSON.stringify(health))));
  vi.stubGlobal('fetch', fetch);
  renderHook(
    () =>
      useConnectionChecks([{ ...connection, environment: 'sandbox' }], { okx: { configured: false, updatedAt: null } }),
    { wrapper: wrapper() },
  );
  await waitFor(() => expect(fetch.mock.calls.filter(([, init]) => init?.method === 'GET')).toHaveLength(1));
});

it('loads persisted results on a fresh page without another platform request, including failures', async () => {
  const failed = { ...health, healthy: false, error_code: 'authentication_failed' };
  const fetch = vi.fn((_url: string, init?: RequestInit) =>
    Promise.resolve(new Response(JSON.stringify(init?.method === 'POST' ? health : failed))),
  );
  vi.stubGlobal('fetch', fetch);
  const { result } = renderHook(
    () =>
      useConnectionChecks([connection], {
        okx: { configured: true, updatedAt: 'saved' },
      }),
    { wrapper: wrapper() },
  );
  await waitFor(() => expect(result.current.checks.okx?.health.error_code).toBe('authentication_failed'));
  expect(fetch.mock.calls.filter(([, init]) => init?.method === 'POST')).toHaveLength(0);
  await act(() => result.current.checkConnection('okx'));
  await waitFor(() => expect(result.current.checks.okx?.health.healthy).toBe(true));
  expect(fetch.mock.calls.filter(([, init]) => init?.method === 'POST')).toHaveLength(1);
});

it('does not automatically check disabled connections', () => {
  const fetch = vi.fn();
  vi.stubGlobal('fetch', fetch);
  const { result } = renderHook(
    () =>
      useConnectionChecks([connection, { ...connection, id: 'disabled', enabled: false }], {
        disabled: { configured: true, updatedAt: 'saved' },
      }),
    { wrapper: wrapper() },
  );
  expect(result.current.checks).toEqual({});
  expect(fetch.mock.calls.filter(([, init]) => init?.method === 'GET')).toHaveLength(1);
});

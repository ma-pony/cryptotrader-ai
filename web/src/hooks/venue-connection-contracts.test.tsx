import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { renderHook } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { useVenueConnections } from './use-venue-connections';

describe('venue connection write recovery', () => {
  it('keeps a successful create as saved when the follow-up config refresh fails', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ revision: 2, connection: { id: 'paper-1', label: 'Paper', adapter_id: 'paper', environment: 'paper', enabled: true, credential_configured: false, credential_updated_at: null, leverage: 1, margin_mode: 'cross', parameters: [] } }), { status: 201 }))
      .mockResolvedValueOnce(new Response('down', { status: 503 }));
    vi.stubGlobal('fetch', fetchMock);
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const hook = renderHook(() => useVenueConnections(), { wrapper: ({ children }) => <QueryClientProvider client={client}>{children}</QueryClientProvider> });
    const result = await hook.result.current.create.mutateAsync({ expected_revision: 1, id: 'paper-1', label: 'Paper', adapter_id: 'paper', environment: 'paper', enabled: true, leverage: 1, margin_mode: 'cross', parameters: {} });
    expect(result.savedNeedsReload).toBe(true);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});

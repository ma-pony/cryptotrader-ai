import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { renderHook, waitFor } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import { configurationCatalogFixture } from '@/test/configuration-catalog-fixture';
import { useConfigurationCatalog } from './use-configuration-catalog';

afterEach(() => vi.unstubAllGlobals());
it('loads typed installed definitions independently of configured components', async () => {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockImplementation((url: string) => {
      expect(url).toContain('/api/config/catalog');
      return Promise.resolve(new Response(JSON.stringify(configurationCatalogFixture), { status: 200 }));
    }),
  );
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const hook = renderHook(() => useConfigurationCatalog(), {
    wrapper: ({ children }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>,
  });
  expect(hook.result.current.isLoading).toBe(true);
  await waitFor(() => expect(hook.result.current.isSuccess).toBe(true));
  expect(hook.result.current.data?.components[0]?.fields[3]).toMatchObject({
    key: 'debate.max_rounds',
    kind: 'integer',
    minimum: 1,
    maximum: 10,
  });
});

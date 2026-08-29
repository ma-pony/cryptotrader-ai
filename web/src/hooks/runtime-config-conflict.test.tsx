import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { renderHook, waitFor } from '@testing-library/react';
import { act } from 'react';
import { expect, it } from 'vitest';
import { clearRuntimeConfigConflict, setRuntimeConfigConflict, useRuntimeConfigConflict } from './runtime-config-conflict';

it('shares conflict only within one QueryClient', async () => {
  const firstClient = new QueryClient(); const secondClient = new QueryClient();
  const first = renderHook(() => useRuntimeConfigConflict(), { wrapper: ({ children }) => <QueryClientProvider client={firstClient}>{children}</QueryClientProvider> });
  const sameClient = renderHook(() => useRuntimeConfigConflict(), { wrapper: ({ children }) => <QueryClientProvider client={firstClient}>{children}</QueryClientProvider> });
  const isolated = renderHook(() => useRuntimeConfigConflict(), { wrapper: ({ children }) => <QueryClientProvider client={secondClient}>{children}</QueryClientProvider> });
  act(() => { setRuntimeConfigConflict(firstClient); });
  await waitFor(() => expect(first.result.current).toBe(true));
  expect(sameClient.result.current).toBe(true);
  expect(isolated.result.current).toBe(false);
  act(() => { clearRuntimeConfigConflict(firstClient); });
  await waitFor(() => expect(first.result.current).toBe(false));
  expect(isolated.result.current).toBe(false);
});

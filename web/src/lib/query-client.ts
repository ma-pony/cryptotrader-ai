import { MutationCache, QueryCache, QueryClient } from '@tanstack/react-query';

import { ApiError } from './api-client';

const handleError = (error: unknown) => {
  if (error instanceof ApiError) {
    console.error(`[ApiError] ${String(error.status)} ${error.code}: ${error.message}`, error.traceId ?? '');
    return;
  }
  console.error('[QueryError]', error);
};

export const queryClient = new QueryClient({
  queryCache: new QueryCache({ onError: handleError }),
  mutationCache: new MutationCache({ onError: handleError }),
  defaultOptions: {
    queries: {
      staleTime: 5_000,
      gcTime: 5 * 60_000,
      retry: 1,
      refetchOnWindowFocus: false,
      refetchOnReconnect: true,
    },
    mutations: {
      retry: 0,
    },
  },
});

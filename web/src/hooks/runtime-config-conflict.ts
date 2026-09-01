import { useQuery, type QueryClient } from '@tanstack/react-query';

/** A non-fetching, per-QueryClient UI state.  It deliberately is not module state. */
export const RUNTIME_CONFIG_CONFLICT_QUERY_KEY = ['runtime-config-conflict'] as const;

export const setRuntimeConfigConflict = (client: QueryClient) => {
  client.setQueryData(RUNTIME_CONFIG_CONFLICT_QUERY_KEY, true);
  void client.invalidateQueries({ queryKey: ['runtime-status'] });
  void client.invalidateQueries({ queryKey: ['trading-scope'] });
};

export const clearRuntimeConfigConflict = (client: QueryClient) =>
  client.setQueryData(RUNTIME_CONFIG_CONFLICT_QUERY_KEY, false);

export const useRuntimeConfigConflict = () => {
  return (
    useQuery({
      queryKey: RUNTIME_CONFIG_CONFLICT_QUERY_KEY,
      queryFn: () => Promise.resolve(false),
      enabled: false,
      staleTime: Infinity,
      initialData: false,
    }).data === true
  );
};

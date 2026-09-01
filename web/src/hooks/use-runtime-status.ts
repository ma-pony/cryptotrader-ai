import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '@/lib/api-client';
import { ReadinessSchema, RuntimeConfigSchema } from '@/types/api.schema';
import { RUNTIME_CONFIG_QUERY_KEY } from './use-runtime-config';

export function useRuntimeStatus() {
  return useQuery({
    queryKey: ['runtime-status'],
    queryFn: () => apiClient.get('/api/runtime/status', ReadinessSchema),
    refetchInterval: 10000,
  });
}

export function useSetAutomation() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (input: { enabled: boolean; expected_revision: number }) =>
      apiClient.put('/api/runtime/automation', input, RuntimeConfigSchema),
    onSuccess: (saved) => {
      client.setQueryData(RUNTIME_CONFIG_QUERY_KEY, saved);
    },
    onSettled: () => {
      void client.invalidateQueries({ queryKey: ['runtime-status'] });
      void client.invalidateQueries({ queryKey: ['trading-scope'] });
    },
  });
}

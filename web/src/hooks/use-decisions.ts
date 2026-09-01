import { keepPreviousData, useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '@/lib/api-client';
import { AnalysisQueuedSchema, DecisionListSchema, DecisionSchema } from '@/types/api.schema';

export function useDecisions(page = 1, componentId?: string) {
  const params = new URLSearchParams({ offset: String((page - 1) * 20), limit: '20' });
  if (componentId) params.set('component_id', componentId);
  return useQuery({
    queryKey: ['decisions', componentId, page],
    queryFn: () => apiClient.get(`/api/decisions?${params}`, DecisionListSchema),
    placeholderData: keepPreviousData,
    refetchInterval: (query) =>
      query.state.data?.items.some((item) => ['queued', 'running'].includes(item.status)) ? 1500 : false,
  });
}

export function useDecision(decisionId: string | undefined) {
  return useQuery({
    queryKey: ['decisions', 'detail', decisionId],
    queryFn: () => apiClient.get(`/api/decisions/${encodeURIComponent(decisionId!)}`, DecisionSchema),
    enabled: Boolean(decisionId),
    refetchInterval: (query) =>
      query.state.data && ['queued', 'running'].includes(query.state.data.status) ? 1500 : false,
  });
}

export function useStartAnalysis() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (input: { pair: string; expected_revision: number }) =>
      apiClient.post('/api/analyses', input, AnalysisQueuedSchema),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ['decisions'] });
    },
  });
}

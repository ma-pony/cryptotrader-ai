import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/lib/api-client';
import { ComponentEvaluationsSchema } from '@/types/api.schema';

export type EvaluationFilters = {
  pair: string;
  mode: string;
  config_revision: string;
  interval: string;
  status: string;
};

export function useComponentEvaluations(componentId: string, filters: EvaluationFilters, page: number) {
  const params = new URLSearchParams({ limit: '20', offset: String((page - 1) * 20) });
  Object.entries(filters).forEach(([key, value]) => {
    if (value) params.set(key, value);
  });
  return useQuery({
    queryKey: ['component-evaluations', componentId, filters, page],
    queryFn: () =>
      apiClient.get(
        `/api/components/${encodeURIComponent(componentId)}/evaluations?${params}`,
        ComponentEvaluationsSchema,
      ),
    refetchInterval: 60000,
  });
}

import { useQuery } from '@tanstack/react-query';

import { apiClient } from '@/lib/api-client';
import { PaginatedTriggerEventsSchema } from '@/types/api.schema';

export function useTriggerHistory(ruleId?: string, page = 1, size = 20) {
  return useQuery({
    queryKey: ['scheduler', 'history', { ruleId, page, size }],
    queryFn: () => {
      const params = new URLSearchParams({ page: String(page), size: String(size) });
      if (ruleId) params.set('rule_id', ruleId);
      return apiClient.get(`/api/scheduler/triggers?${params.toString()}`, PaginatedTriggerEventsSchema);
    },
    refetchInterval: 10000,
  });
}

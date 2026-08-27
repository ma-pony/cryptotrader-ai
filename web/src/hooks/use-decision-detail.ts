import { useQuery } from '@tanstack/react-query';

import { apiClient } from '@/lib/api-client';
import { DecisionDetailSchema } from '@/types/api.schema';

export const useDecisionDetail = (cycleId: string | undefined) =>
  useQuery({
    queryKey: ['decision-detail', cycleId],
    queryFn: () => apiClient.get(`/api/decisions/${cycleId}`, DecisionDetailSchema),
    enabled: !!cycleId,
  });

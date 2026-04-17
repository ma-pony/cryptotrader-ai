import { useQuery } from '@tanstack/react-query';

import { apiClient } from '@/lib/api-client';
import { DecisionDetailSchema } from '@/types/api.schema';

export const useDecisionDetail = (commitHash: string | undefined) =>
  useQuery({
    queryKey: ['decision-detail', commitHash],
    queryFn: () => apiClient.get(`/api/decisions/${commitHash}`, DecisionDetailSchema),
    enabled: !!commitHash,
  });

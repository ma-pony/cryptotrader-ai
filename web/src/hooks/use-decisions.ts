import { useQuery, keepPreviousData } from '@tanstack/react-query';

import { apiClient } from '@/lib/api-client';
import { PaginatedCyclesSchema } from '@/types/api.schema';

export const useDecisions = (filter: { page?: number; size?: number }) => {
  const params = new URLSearchParams();
  if (filter.page) params.set('page', String(filter.page));
  if (filter.size) params.set('size', String(filter.size));
  const qs = params.toString();

  return useQuery({
    queryKey: ['decisions', filter.page, filter.size],
    queryFn: () => apiClient.get(`/api/decisions${qs ? `?${qs}` : ''}`, PaginatedCyclesSchema),
    placeholderData: keepPreviousData,
  });
};

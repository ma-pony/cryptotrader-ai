import { useQuery, keepPreviousData } from '@tanstack/react-query';

import { apiClient } from '@/lib/api-client';
import type { DecisionListFilter } from '@/types/api';
import { PaginatedDecisionsSchema } from '@/types/api.schema';

export const useDecisions = (filter: DecisionListFilter) => {
  const params = new URLSearchParams();
  if (filter.pair) params.set('pair', filter.pair);
  if (filter.from) params.set('from', filter.from);
  if (filter.to) params.set('to', filter.to);
  if (filter.page) params.set('page', String(filter.page));
  if (filter.size) params.set('size', String(filter.size));
  const qs = params.toString();

  return useQuery({
    queryKey: ['decisions', filter],
    queryFn: () => apiClient.get(`/api/decisions${qs ? `?${qs}` : ''}`, PaginatedDecisionsSchema),
    placeholderData: keepPreviousData,
  });
};

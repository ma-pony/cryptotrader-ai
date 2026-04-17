import { useQuery } from '@tanstack/react-query';

import { apiClient } from '@/lib/api-client';
import { PortfolioSchema } from '@/types/api.schema';

export const usePortfolioSnapshot = () =>
  useQuery({
    queryKey: ['portfolio-snapshot'],
    queryFn: () => apiClient.get('/api/portfolio/snapshot', PortfolioSchema),
    refetchInterval: 10_000,
  });

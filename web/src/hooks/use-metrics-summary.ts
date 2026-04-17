import { useQuery } from '@tanstack/react-query';

import { apiClient } from '@/lib/api-client';
import { MetricsSummarySchema } from '@/types/api.schema';

export const useMetricsSummary = () =>
  useQuery({
    queryKey: ['metrics-summary'],
    queryFn: () => apiClient.get('/api/metrics/summary', MetricsSummarySchema),
    refetchInterval: 10_000,
  });

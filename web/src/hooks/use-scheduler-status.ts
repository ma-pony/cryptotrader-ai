import { useQuery } from '@tanstack/react-query';

import { apiClient } from '@/lib/api-client';
import { SchedulerStatusSchema } from '@/types/api.schema';

export const useSchedulerStatus = () =>
  useQuery({
    queryKey: ['scheduler-status'],
    queryFn: () => apiClient.get('/api/scheduler/status', SchedulerStatusSchema),
    refetchInterval: 10_000,
  });

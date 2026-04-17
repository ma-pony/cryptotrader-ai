import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

import { apiClient } from '@/lib/api-client';
import { CircuitBreakerResetSchema, RiskStatusSchema } from '@/types/api.schema';

export const useRiskStatus = () =>
  useQuery({
    queryKey: ['risk-status'],
    queryFn: () => apiClient.get('/api/risk/status', RiskStatusSchema),
    refetchInterval: 5_000,
  });

export const useResetCircuitBreaker = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => apiClient.post('/api/risk/circuit-breaker/reset', {}, CircuitBreakerResetSchema),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ['risk-status'] });
    },
  });
};

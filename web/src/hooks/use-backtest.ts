import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

import { apiClient } from '@/lib/api-client';
import type { BacktestParams } from '@/types/api';
import {
  BacktestCancelResponseSchema,
  BacktestRunResponseSchema,
  BacktestRunStatusSchema,
  BacktestSessionsListSchema,
} from '@/types/api.schema';

export const useBacktestSessions = () =>
  useQuery({
    queryKey: ['backtest-sessions'],
    queryFn: () => apiClient.get('/api/backtest/sessions', BacktestSessionsListSchema),
  });

export const useBacktestRun = (runId: string | undefined) =>
  useQuery({
    queryKey: ['backtest-run', runId],
    queryFn: () => apiClient.get(`/api/backtest/runs/${runId}`, BacktestRunStatusSchema),
    enabled: !!runId,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      if (status === 'completed' || status === 'failed' || status === 'canceled') return false;
      return 5_000;
    },
  });

export const useStartBacktest = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (params: Omit<BacktestParams, 'session_name'>) =>
      apiClient.post('/api/backtest/run', params, BacktestRunResponseSchema),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ['backtest-sessions'] });
    },
  });
};

export const useCancelBacktest = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (runId: string) => apiClient.delete(`/api/backtest/runs/${runId}`, BacktestCancelResponseSchema),
    onSuccess: (_data, runId) => {
      void qc.invalidateQueries({ queryKey: ['backtest-run', runId] });
    },
  });
};

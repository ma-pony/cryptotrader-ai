import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '@/lib/api-client';
import type { BacktestParams } from '@/types/api';
import {
  BacktestCancelResponseSchema,
  BacktestRunResponseSchema,
  BacktestRunStatusSchema,
  BacktestRunsSchema,
  BacktestComparisonSchema,
} from '@/types/api.schema';

export const isActiveBacktest = (status: string) => status === 'queued' || status === 'running';

export const useBacktestRuns = (offset = 0) =>
  useQuery({
    queryKey: ['backtest-runs', offset],
    queryFn: () => apiClient.get(`/api/backtest/runs?limit=20&offset=${offset}`, BacktestRunsSchema),
    refetchInterval: (query) => (query.state.data?.items.some((run) => isActiveBacktest(run.status)) ? 5000 : false),
  });
export const useLoadBacktestRun = () =>
  useMutation({
    mutationFn: (id: string) => apiClient.get(`/api/backtest/runs/${encodeURIComponent(id)}`, BacktestRunStatusSchema),
  });
export const useBacktestRun = (runId: string | undefined) =>
  useQuery({
    queryKey: ['backtest-run', runId],
    queryFn: () => apiClient.get(`/api/backtest/runs/${encodeURIComponent(runId!)}`, BacktestRunStatusSchema),
    enabled: !!runId,
    refetchInterval: (query) => (query.state.data && !isActiveBacktest(query.state.data.status) ? false : 5000),
  });
export const useBacktestComparison = (left: string | null, right: string | null) =>
  useQuery({
    queryKey: ['backtest-comparison', left, right],
    queryFn: () =>
      apiClient.get(
        `/api/backtest/runs/compare?left=${encodeURIComponent(left!)}&right=${encodeURIComponent(right!)}`,
        BacktestComparisonSchema,
      ),
    enabled: !!left && !!right,
  });
export const useStartBacktest = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (params: BacktestParams) => apiClient.post('/api/backtest/runs', params, BacktestRunResponseSchema),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ['backtest-runs'] });
    },
  });
};
export const useCancelBacktest = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (runId: string) => apiClient.delete(`/api/backtest/runs/${runId}`, BacktestCancelResponseSchema),
    onSuccess: (_data, runId) => {
      void qc.invalidateQueries({ queryKey: ['backtest-run', runId] });
      void qc.invalidateQueries({ queryKey: ['backtest-runs'] });
    },
  });
};

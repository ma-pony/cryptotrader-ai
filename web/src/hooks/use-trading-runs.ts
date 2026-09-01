import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '@/lib/api-client';
import { AnalysisQueuedSchema, TradingScopeSchema } from '@/types/api.schema';

export function useTradingScope(pair: string) {
  return useQuery({
    queryKey: ['trading-scope', pair],
    queryFn: () => apiClient.get(`/api/trading-runs/scope?${new URLSearchParams({ pair })}`, TradingScopeSchema),
    enabled: Boolean(pair),
  });
}

export function useStartTrading() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (input: { pair: string; expected_revision: number; confirmed_book_ids: string[] }) =>
      apiClient.post('/api/trading-runs', input, AnalysisQueuedSchema),
    onSettled: () => {
      void client.invalidateQueries({ queryKey: ['trading-scope'] });
      void client.invalidateQueries({ queryKey: ['runtime-status'] });
      void client.invalidateQueries({ queryKey: ['decisions'] });
    },
  });
}

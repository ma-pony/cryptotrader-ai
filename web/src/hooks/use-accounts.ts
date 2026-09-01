import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '@/lib/api-client';
import { AccountSchema, AccountsSchema, AccountFillsSchema, AccountIncomeSchema } from '@/types/api.schema';

export const useAccounts = () =>
  useQuery({
    queryKey: ['accounts'],
    queryFn: () => apiClient.get('/api/accounts', AccountsSchema),
    refetchInterval: 60_000,
  });
export const useAccount = (id: string) =>
  useQuery({
    queryKey: ['accounts', id],
    queryFn: () => apiClient.get(`/api/accounts/${encodeURIComponent(id)}`, AccountSchema),
    enabled: Boolean(id),
    refetchInterval: 60_000,
  });
export const useSyncAccount = (id: string) => {
  const client = useQueryClient();
  return useMutation({
    mutationFn: () => apiClient.post(`/api/accounts/${encodeURIComponent(id)}/sync`, undefined, AccountSchema),
    onSettled: async () => {
      await Promise.all([
        client.invalidateQueries({ queryKey: ['accounts'] }),
        client.invalidateQueries({ queryKey: ['portfolio-books'] }),
      ]);
    },
  });
};
export const useAccountHistory = (
  id: string,
  filters: { symbol: string; start: string; end: string; offset: number },
  enabled: boolean,
) => {
  const query = new URLSearchParams({ offset: String(filters.offset), limit: '50' });
  for (const key of ['symbol', 'start', 'end'] as const) if (filters[key]) query.set(key, filters[key]);
  const base = `/api/accounts/${encodeURIComponent(id)}`;
  const fills = useQuery({
    queryKey: ['accounts', id, 'fills', filters],
    queryFn: () => apiClient.get(`${base}/fills?${query}`, AccountFillsSchema),
    enabled,
  });
  const income = useQuery({
    queryKey: ['accounts', id, 'income', filters],
    queryFn: () => apiClient.get(`${base}/income?${query}`, AccountIncomeSchema),
    enabled,
  });
  return { fills, income };
};

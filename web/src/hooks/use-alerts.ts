import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '@/lib/api-client';
import { AlertListSchema, AlertOverviewSchema, AlertSchema, DeliverySchema } from '@/types/api.schema';

export const ALERTS_QUERY_KEY = ['alerts'] as const;
export function useAlerts(filters: { connectionId?: string | undefined; bookId?: string | undefined } = {}) {
  const client = useQueryClient();
  const params = new URLSearchParams();
  if (filters.connectionId) params.set('connection_id', filters.connectionId);
  if (filters.bookId) params.set('book_id', filters.bookId);
  const suffix = params.size ? `?${params.toString()}` : '';
  const query = useQuery({ queryKey: [...ALERTS_QUERY_KEY, filters.connectionId, filters.bookId], queryFn: () => apiClient.get(`/api/alerts${suffix}`, AlertListSchema) });
  const read = useMutation({ mutationFn: (id: string) => apiClient.post(`/api/alerts/${encodeURIComponent(id)}/read`, {}, AlertSchema), onSuccess: () => client.invalidateQueries({ queryKey: ALERTS_QUERY_KEY }) });
  return { ...query, markRead: read.mutate, isMarkingRead: read.isPending };
}

export function useAlertOverview() {
  const client = useQueryClient();
  const query = useQuery({ queryKey: [...ALERTS_QUERY_KEY, 'overview'], queryFn: () => apiClient.get('/api/alerts/overview', AlertOverviewSchema) });
  const retry = useMutation({ mutationFn: (id: string) => apiClient.post(`/api/alerts/deliveries/${encodeURIComponent(id)}/retry`, {}, DeliverySchema), onSuccess: () => client.invalidateQueries({ queryKey: ALERTS_QUERY_KEY }) });
  return { ...query, retry: retry.mutate, isRetrying: retry.isPending };
}

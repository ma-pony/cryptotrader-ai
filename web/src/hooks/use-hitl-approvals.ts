import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

import { apiClient } from '@/lib/api-client';
import { HitlPendingListSchema, HitlRespondSchema } from '@/types/api.schema';

export const useHitlPending = () =>
  useQuery({
    queryKey: ['hitl-pending'],
    queryFn: () => apiClient.get('/api/hitl/pending', HitlPendingListSchema),
    refetchInterval: 5_000,
  });

export const useHitlRespond = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ approvalId, decision, comment }: { approvalId: string; decision: 'approve' | 'reject'; comment?: string }) =>
      apiClient.post(`/api/hitl/${approvalId}/respond`, { decision, comment: comment ?? '' }, HitlRespondSchema),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ['hitl-pending'] });
    },
  });
};

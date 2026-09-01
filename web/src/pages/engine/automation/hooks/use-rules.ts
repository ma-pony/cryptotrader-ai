import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

import { apiClient } from '@/lib/api-client';
import { ScheduleRuleListSchema, ScheduleRuleSchema } from '@/types/api.schema';

type RulePayload = {
  name: string;
  trigger_type: string;
  pair: string;
  parameters: Record<string, unknown>;
  cooldown_minutes: number;
};

export function useRules(enabled?: boolean) {
  return useQuery({
    queryKey: ['scheduler', 'rules', { enabled }],
    queryFn: () => {
      const params = enabled !== undefined ? `?enabled=${String(enabled)}` : '';
      return apiClient.get(`/api/scheduler/rules${params}`, ScheduleRuleListSchema);
    },
    refetchInterval: 5000,
  });
}

export function useCreateRule() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: RulePayload) =>
      apiClient.post('/api/scheduler/rules', data, ScheduleRuleSchema),
    onSuccess: () => void qc.invalidateQueries({ queryKey: ['scheduler', 'rules'] }),
  });
}

export function useUpdateRule() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, ...data }: RulePayload & { id: string }) =>
      apiClient.put(`/api/scheduler/rules/${id}`, data, ScheduleRuleSchema),
    onSuccess: () => void qc.invalidateQueries({ queryKey: ['scheduler', 'rules'] }),
  });
}

export function useToggleRule() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => apiClient.patch(`/api/scheduler/rules/${id}/toggle`, {}, ScheduleRuleSchema),
    onSuccess: () => void qc.invalidateQueries({ queryKey: ['scheduler', 'rules'] }),
  });
}

export function useDeleteRule() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => apiClient.delete(`/api/scheduler/rules/${id}`, ScheduleRuleSchema),
    onSuccess: () => void qc.invalidateQueries({ queryKey: ['scheduler', 'rules'] }),
  });
}

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

import { apiClient } from '@/lib/api-client';
import type { SignalProfileUpdate } from '@/types/api';
import { SignalProfileSchema } from '@/types/api.schema';

const SIGNAL_PROFILE_QUERY_KEY = ['signal-profile'] as const;

export const useSignalProfile = () =>
  useQuery({
    queryKey: SIGNAL_PROFILE_QUERY_KEY,
    queryFn: () => apiClient.get('/api/signal-profile', SignalProfileSchema),
  });

export const useSaveSignalProfile = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (profile: SignalProfileUpdate) =>
      apiClient.put('/api/signal-profile', profile, SignalProfileSchema),
    onSuccess: (profile) => {
      queryClient.setQueryData(SIGNAL_PROFILE_QUERY_KEY, profile);
    },
  });
};

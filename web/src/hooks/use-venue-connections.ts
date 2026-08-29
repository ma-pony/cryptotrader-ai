import { useMutation, useQueryClient } from '@tanstack/react-query';

import { apiClient } from '@/lib/api-client';
import { ConnectionHealthSchema, CredentialMutationSchema, RuntimeConfigSchema, VenueMutationSchema } from '@/types/api.schema';
import { RUNTIME_CONFIG_QUERY_KEY } from './use-runtime-config';
import { setRuntimeConfigConflict } from './runtime-config-conflict';
import { ApiError } from '@/lib/api-client';

type ConnectionInput = {
  expected_revision: number;
  id?: string;
  label: string;
  adapter_id: string;
  environment: 'paper' | 'demo' | 'testnet' | 'live';
  enabled: boolean;
  leverage: number;
  margin_mode: string;
  parameters: Record<string, unknown>;
};
export type CredentialInput = { api_key: string; secret: string; passphrase?: string };

export const useVenueConnections = () => {
  const client = useQueryClient();
  const refreshConfig = () => client.fetchQuery({
    queryKey: RUNTIME_CONFIG_QUERY_KEY,
    queryFn: () => apiClient.get('/api/config', RuntimeConfigSchema),
    staleTime: 0,
  });
  const conflict = (error: unknown) => {
    if (error instanceof ApiError && error.status === 409) setRuntimeConfigConflict(client);
  };
  const create = useMutation({
    mutationFn: (body: ConnectionInput & { id: string }) =>
      apiClient.post('/api/venue-connections', body, VenueMutationSchema),
    onSuccess: refreshConfig,
    onError: conflict,
  });
  const update = useMutation({
    mutationFn: ({ id, body }: { id: string; body: ConnectionInput }) =>
      apiClient.put(`/api/venue-connections/${id}`, body, VenueMutationSchema),
    onSuccess: refreshConfig,
    onError: conflict,
  });
  // Credential material must never become mutation variables: this plain async boundary is intentionally not cached.
  const putCredentials = async ({
    id,
    expectedRevision,
    credentials,
  }: {
    id: string;
    expectedRevision: number;
    credentials: CredentialInput;
  }) => {
    let result;
    try {
      result = await apiClient.put(
        `/api/venue-connections/${id}/credentials`,
        { expected_revision: expectedRevision, credentials },
        CredentialMutationSchema,
      );
    } catch (error) {
      conflict(error);
      throw error;
    }
    await refreshConfig();
    return result;
  };
  const test = useMutation({
    mutationFn: (id: string) => apiClient.post(`/api/venue-connections/${id}/test`, {}, ConnectionHealthSchema),
  });
  return { create, update, putCredentials, test };
};

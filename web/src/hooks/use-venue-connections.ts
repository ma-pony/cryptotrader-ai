import { useMutation, useQueryClient } from '@tanstack/react-query';

import { apiClient } from '@/lib/api-client';
import { ConnectionHealthSchema, CredentialMutationSchema, VenueMutationSchema } from '@/types/api.schema';
import { RUNTIME_CONFIG_QUERY_KEY } from './use-runtime-config';

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
  const sync = () => client.invalidateQueries({ queryKey: RUNTIME_CONFIG_QUERY_KEY });
  const create = useMutation({ mutationFn: (body: ConnectionInput & { id: string }) => apiClient.post('/api/venue-connections', body, VenueMutationSchema), onSuccess: sync });
  const update = useMutation({ mutationFn: ({ id, body }: { id: string; body: ConnectionInput }) => apiClient.put(`/api/venue-connections/${id}`, body, VenueMutationSchema), onSuccess: sync });
  const credentials = useMutation({ mutationFn: ({ id, expectedRevision, credentials }: { id: string; expectedRevision: number; credentials: CredentialInput }) => apiClient.put(`/api/venue-connections/${id}/credentials`, { expected_revision: expectedRevision, credentials }, CredentialMutationSchema), onSuccess: sync });
  const test = useMutation({ mutationFn: (id: string) => apiClient.post(`/api/venue-connections/${id}/test`, {}, ConnectionHealthSchema) });
  return { create, update, credentials, test };
};

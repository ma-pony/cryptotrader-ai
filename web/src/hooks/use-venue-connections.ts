import { useMutation, useQueryClient } from '@tanstack/react-query';

import { apiClient } from '@/lib/api-client';
import { ConnectionHealthSchema, CredentialMutationSchema, RuntimeConfigSchema, VenueMutationSchema } from '@/types/api.schema';
import { RUNTIME_CONFIG_QUERY_KEY } from './use-runtime-config';
import { setRuntimeConfigConflict } from './runtime-config-conflict';
import { ApiError } from '@/lib/api-client';
import type { RuntimeConfig, RuntimeJsonObject } from '@/types/api';

type ConnectionInput = {
  expected_revision: number;
  id?: string;
  label: string;
  adapter_id: string;
  environment: 'paper' | 'demo' | 'testnet' | 'live';
  enabled: boolean;
  leverage: number;
  margin_mode: string;
  parameters: RuntimeJsonObject;
};
export type CredentialInput = { api_key: string; secret: string; passphrase?: string };

export const useVenueConnections = () => {
  const client = useQueryClient();
  const refreshConfig = () => client.fetchQuery({
    queryKey: RUNTIME_CONFIG_QUERY_KEY,
    queryFn: () => apiClient.get('/api/config', RuntimeConfigSchema),
    staleTime: 0,
  });
  const applyMutation = (mutation: { revision: number; connection: RuntimeConfig['document']['execution']['connections'][number] }) => {
    client.setQueryData<RuntimeConfig>(RUNTIME_CONFIG_QUERY_KEY, (current) => current ? {
      ...current,
      revision: mutation.revision,
      document: { ...current.document, execution: { ...current.document.execution, connections: current.document.execution.connections.some((item) => item.id === mutation.connection.id) ? current.document.execution.connections.map((item) => item.id === mutation.connection.id ? mutation.connection : item) : [...current.document.execution.connections, mutation.connection] } },
    } : current);
  };
  const writeThenRefresh = async (write: () => Promise<{ revision: number; connection: RuntimeConfig['document']['execution']['connections'][number] }>) => {
    const mutation = await write();
    applyMutation(mutation);
    try { await refreshConfig(); return { ...mutation, savedNeedsReload: false }; }
    catch { return { ...mutation, savedNeedsReload: true }; }
  };
  const conflict = (error: unknown) => {
    if (error instanceof ApiError && error.status === 409) setRuntimeConfigConflict(client);
  };
  const create = useMutation({
    mutationFn: (body: ConnectionInput & { id: string }) => writeThenRefresh(() => apiClient.post('/api/venue-connections', body, VenueMutationSchema)),
    onError: conflict,
  });
  const update = useMutation({
    mutationFn: ({ id, body }: { id: string; body: ConnectionInput }) => writeThenRefresh(() => apiClient.put(`/api/venue-connections/${id}`, body, VenueMutationSchema)),
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
    // The credential write has already committed. A failed best-effort refresh
    // must not make callers retain credential material and retry it blindly.
    try {
      await refreshConfig();
      return { ...result, savedNeedsReload: false };
    } catch {
      return { ...result, savedNeedsReload: true };
    }
  };
  const test = useMutation({
    mutationFn: (id: string) => apiClient.post(`/api/venue-connections/${id}/test`, {}, ConnectionHealthSchema),
  });
  return { create, update, putCredentials, test };
};

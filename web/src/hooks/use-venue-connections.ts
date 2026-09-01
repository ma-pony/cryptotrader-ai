import { useMutation, useQueryClient } from '@tanstack/react-query';

import { apiClient } from '@/lib/api-client';
import { CredentialMutationSchema, RuntimeConfigSchema, VenueMutationSchema } from '@/types/api.schema';
import { RUNTIME_CONFIG_QUERY_KEY } from './use-runtime-config';
import { setRuntimeConfigConflict } from './runtime-config-conflict';
import { ApiError } from '@/lib/api-client';
import type { RuntimeConfig, RuntimeJsonObject } from '@/types/api';

type ConnectionInput = {
  confirm_stop?: boolean;
  expected_revision: number;
  label: string;
  adapter_id: string;
  environment: string;
  enabled: boolean;
  canary_only: boolean;
  leverage: number;
  margin_mode: string;
  parameters: RuntimeJsonObject;
};
type CreateConnectionInput = ConnectionInput & { id: string };
export type CredentialInput = Record<string, string>;

export const useVenueConnections = () => {
  const client = useQueryClient();
  const refreshConfig = () =>
    client.fetchQuery({
      queryKey: RUNTIME_CONFIG_QUERY_KEY,
      queryFn: () => apiClient.get('/api/config', RuntimeConfigSchema),
      staleTime: 0,
    });
  const applyMutation = (mutation: {
    revision: number;
    connection: RuntimeConfig['document']['execution']['connections'][number];
  }) => {
    void client.invalidateQueries({ queryKey: ['runtime-status'] });
    void client.invalidateQueries({ queryKey: ['trading-scope'] });
    client.setQueryData<RuntimeConfig>(RUNTIME_CONFIG_QUERY_KEY, (current) =>
      current
        ? {
            ...current,
            revision: mutation.revision,
            document: {
              ...current.document,
              execution: {
                ...current.document.execution,
                connections: current.document.execution.connections.some((item) => item.id === mutation.connection.id)
                  ? current.document.execution.connections.map((item) =>
                      item.id === mutation.connection.id ? mutation.connection : item,
                    )
                  : [...current.document.execution.connections, mutation.connection],
              },
            },
          }
        : current,
    );
  };
  const applyCredentialMutation = (
    id: string,
    mutation: { revision: number; credential: { configured: boolean; updated_at: string | null } },
  ) => {
    void client.invalidateQueries({ queryKey: ['runtime-status'] });
    void client.invalidateQueries({ queryKey: ['trading-scope'] });
    client.setQueryData<RuntimeConfig>(RUNTIME_CONFIG_QUERY_KEY, (current) =>
      current
        ? {
            ...current,
            revision: mutation.revision,
            document: {
              ...current.document,
              execution: {
                ...current.document.execution,
                connections: current.document.execution.connections.map((connection) =>
                  connection.id === id
                    ? {
                        ...connection,
                        credential_configured: mutation.credential.configured,
                        credential_updated_at: mutation.credential.updated_at,
                      }
                    : connection,
                ),
              },
            },
          }
        : current,
    );
  };
  const writeThenRefresh = async (
    write: () => Promise<{
      revision: number;
      connection: RuntimeConfig['document']['execution']['connections'][number];
    }>,
  ) => {
    const mutation = await write();
    applyMutation(mutation);
    try {
      await refreshConfig();
      return { ...mutation, savedNeedsReload: false };
    } catch {
      setRuntimeConfigConflict(client);
      return { ...mutation, savedNeedsReload: true };
    }
  };
  const conflict = (error: unknown) => {
    if (error instanceof ApiError && error.status === 409) setRuntimeConfigConflict(client);
  };
  const create = useMutation({
    mutationFn: (body: CreateConnectionInput) =>
      writeThenRefresh(() => apiClient.post('/api/venue-connections', body, VenueMutationSchema)),
    onError: conflict,
  });
  const update = useMutation({
    mutationFn: ({ id, body }: { id: string; body: ConnectionInput }) =>
      writeThenRefresh(() => apiClient.put(`/api/venue-connections/${id}`, body, VenueMutationSchema)),
    onError: conflict,
  });
  // Credential material must never become mutation variables: this plain async boundary is intentionally not cached.
  const putCredentials = async ({
    id,
    expectedRevision,
    values,
  }: {
    id: string;
    expectedRevision: number;
    values: CredentialInput;
  }) => {
    let result;
    try {
      result = await apiClient.put(
        `/api/venue-connections/${id}/credentials`,
        { expected_revision: expectedRevision, values },
        CredentialMutationSchema,
      );
    } catch (error) {
      conflict(error);
      throw error;
    }
    applyCredentialMutation(id, result);
    // The credential write has already committed. A failed best-effort refresh
    // must not make callers retain credential material and retry it blindly.
    try {
      await refreshConfig();
      return { ...result, savedNeedsReload: false };
    } catch {
      setRuntimeConfigConflict(client);
      return { ...result, savedNeedsReload: true };
    }
  };
  const deleteCredentials = async ({ id, expectedRevision }: { id: string; expectedRevision: number }) => {
    let result;
    try {
      result = await apiClient.delete(
        `/api/venue-connections/${id}/credentials?expected_revision=${encodeURIComponent(String(expectedRevision))}`,
        CredentialMutationSchema,
      );
    } catch (error) {
      conflict(error);
      throw error;
    }
    applyCredentialMutation(id, result);
    try {
      await refreshConfig();
      return { ...result, savedNeedsReload: false };
    } catch {
      setRuntimeConfigConflict(client);
      return { ...result, savedNeedsReload: true };
    }
  };
  return { create, update, putCredentials, deleteCredentials };
};

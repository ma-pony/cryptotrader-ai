import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useMemo } from 'react';

import { ApiError, apiClient } from '@/lib/api-client';
import { RuntimeConfigSchema, type JsonValueOut } from '@/types/api.schema';
import type { RuntimeConfig, RuntimeDocument, RuntimeJsonObject, RuntimeJsonValue } from '@/types/api';
import {
  clearRuntimeConfigConflict,
  setRuntimeConfigConflict,
  useRuntimeConfigConflict,
} from './runtime-config-conflict';

export const RUNTIME_CONFIG_QUERY_KEY = ['runtime-config'] as const;


export const decodeJsonValue = (value: JsonValueOut): RuntimeJsonValue => {
  switch (value.kind) {
    case 'null':
      return null;
    case 'boolean':
      if (typeof value.boolean_value !== 'boolean') throw new Error('Invalid runtime boolean parameter');
      return value.boolean_value;
    case 'number': {
      if (typeof value.number_value !== 'string') throw new Error('Invalid runtime numeric parameter');
      const parsed = Number(value.number_value);
      if (!Number.isFinite(parsed)) throw new Error('Invalid runtime numeric parameter');
      return parsed;
    }
    case 'string':
      if (typeof value.string_value !== 'string') throw new Error('Invalid runtime string parameter');
      return value.string_value;
    case 'datetime':
      if (typeof value.datetime_value !== 'string') throw new Error('Invalid runtime datetime parameter');
      return value.datetime_value;
    case 'pair':
      if (typeof value.pair_value !== 'string') throw new Error('Invalid runtime pair parameter');
      return value.pair_value;
    case 'array':
      return value.items.map(decodeJsonValue);
    case 'object':
      return Object.fromEntries(value.entries.map((entry) => [entry.key, decodeJsonValue(entry.value)]));
  }
};

export const decodeEntries = (entries: { key: string; value: JsonValueOut }[]): RuntimeJsonObject =>
  Object.fromEntries(entries.map((entry) => [entry.key, decodeJsonValue(entry.value)]));

/** Converts response-only JsonValueOut envelopes to a plain writable document. */
export const toRuntimeDocument = (response: RuntimeConfig['document']): RuntimeDocument => ({
  ...response,
  market_data: { ...response.market_data, parameters: decodeEntries(response.market_data.parameters) },
  signals: {
    ...response.signals,
    components: response.signals.components.map((component) => ({
      ...component,
      parameters: decodeEntries(component.parameters),
    })),
  },
  execution: {
    ...response.execution,
    connections: response.execution.connections.map(
      ({ credential_configured: _configured, credential_updated_at: _updatedAt, parameters, ...connection }) => ({
        ...connection,
        parameters: decodeEntries(parameters),
      }),
    ),
  },
});

export const useRuntimeConfig = () => {
  const client = useQueryClient();
  const conflict = useRuntimeConfigConflict();
  const query = useQuery({
    queryKey: RUNTIME_CONFIG_QUERY_KEY,
    queryFn: () => apiClient.get('/api/config', RuntimeConfigSchema),
  });
  const mutation = useMutation({
    mutationFn: (document: RuntimeDocument) => {
      if (query.data?.revision === undefined)
        return Promise.reject(new Error('Runtime config revision is unavailable'));
      return apiClient.put('/api/config', { expected_revision: query.data.revision, document }, RuntimeConfigSchema);
    },
    onSuccess: (saved) => {
      client.setQueryData(RUNTIME_CONFIG_QUERY_KEY, saved);
    },
    onError: (error) => {
      if (error instanceof ApiError && error.status === 409) setRuntimeConfigConflict(client);
    },
  });
  const document = useMemo(() => (query.data ? toRuntimeDocument(query.data.document) : undefined), [query.data]);
  const credentialStates = useMemo(
    () =>
      Object.fromEntries(
        (query.data?.document.execution.connections ?? []).map((connection) => [
          connection.id,
          { configured: connection.credential_configured, updatedAt: connection.credential_updated_at },
        ]),
      ),
    [query.data],
  );
  const reload = async () => {
    const result = await query.refetch();
    if (result.isSuccess && !result.error) clearRuntimeConfigConflict(client);
    return result;
  };
  return {
    revision: query.data?.revision,
    document,
    credentialStates,
    setupRequired: query.data?.setup_required ?? false,
    updatedAt: query.data?.updated_at,
    replace: mutation.mutateAsync,
    reload,
    conflict,
    isLoading: query.isLoading,
    isError: query.isError,
    isSaving: mutation.isPending,
  };
};

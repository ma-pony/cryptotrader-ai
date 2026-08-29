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

/**
 * JSON.stringify silently drops undefined/functions and turns non-finite
 * numbers into null. Configuration writes must preserve the user's document
 * exactly, so reject those values before the request boundary.
 */
export const assertRuntimeJsonDocument = <T>(value: T): T => {
  const visit = (candidate: unknown, seen: Set<object>): void => {
    if (candidate === null || typeof candidate === 'string' || typeof candidate === 'boolean') return;
    if (typeof candidate === 'number') {
      if (Number.isFinite(candidate)) return;
      throw new Error('Invalid runtime JSON: numbers must be finite');
    }
    if (Array.isArray(candidate)) {
      if (seen.has(candidate)) throw new Error('Invalid runtime JSON: cyclic value');
      if (Object.getPrototypeOf(candidate) !== Array.prototype || Object.getOwnPropertySymbols(candidate).length > 0) throw new Error('Invalid runtime JSON: arrays must be ordinary');
      const names = Object.getOwnPropertyNames(candidate);
      if (names.length !== candidate.length + 1 || names[names.length - 1] !== 'length') throw new Error('Invalid runtime JSON: arrays must be dense');
      const lengthDescriptor = Object.getOwnPropertyDescriptor(candidate, 'length');
      const lengthValue: unknown = lengthDescriptor?.value;
      if (!lengthDescriptor || !('value' in lengthDescriptor) || lengthValue !== candidate.length || !lengthDescriptor.writable || lengthDescriptor.enumerable || lengthDescriptor.configurable) throw new Error('Invalid runtime JSON: array length must be ordinary');
      for (let index = 0; index < candidate.length; index += 1) {
        const descriptor = Object.getOwnPropertyDescriptor(candidate, String(index));
        if (!descriptor || !('value' in descriptor) || !descriptor.enumerable || !descriptor.writable || !descriptor.configurable) throw new Error('Invalid runtime JSON: array entries must be ordinary values');
      }
      seen.add(candidate);
      candidate.forEach((item) => visit(item, seen));
      seen.delete(candidate);
      return;
    }
    if (typeof candidate === 'object') {
      if (seen.has(candidate)) throw new Error('Invalid runtime JSON: cyclic value');
      const prototype: unknown = Object.getPrototypeOf(candidate);
      if (prototype !== Object.prototype && prototype !== null) throw new Error('Invalid runtime JSON: objects must be plain');
      if (Object.getOwnPropertySymbols(candidate).length > 0) throw new Error('Invalid runtime JSON: symbol keys are not supported');
      seen.add(candidate);
      Object.values(candidate as Record<string, unknown>).forEach((item) => visit(item, seen));
      seen.delete(candidate);
      return;
    }
    throw new Error('Invalid runtime JSON: values must be JSON primitives, arrays, or objects');
  };
  visit(value, new Set());
  return value;
};

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
  security: { enabled: response.security.enabled },
  llm: (({ gateway_credential_configured: _configured, gateway_credential_updated_at: _updatedAt, ...llm }) => llm)(response.llm),
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
      return apiClient.put('/api/config', { expected_revision: query.data.revision, document: assertRuntimeJsonDocument(document) }, RuntimeConfigSchema);
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
  const secretStates = useMemo(
    () => ({
      llmGateway: {
        configured: query.data?.document.llm.gateway_credential_configured ?? false,
        updatedAt: query.data?.document.llm.gateway_credential_updated_at ?? null,
      },
      apiAccess: {
        configured: query.data?.document.security.access_credential_configured ?? false,
        updatedAt: query.data?.document.security.access_credential_updated_at ?? null,
      },
    }),
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
    secretStates,
    setupRequired: query.data?.setup_required ?? false,
    applyStatus: query.data?.apply_status,
    appliedRevision: query.data?.applied_revision,
    applyError: query.data?.apply_error,
    updatedAt: query.data?.updated_at,
    replace: mutation.mutateAsync,
    reload,
    conflict,
    isLoading: query.isLoading,
    isError: query.isError && query.data === undefined,
    isSaving: mutation.isPending,
  };
};

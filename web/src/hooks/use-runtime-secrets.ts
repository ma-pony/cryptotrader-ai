import { useQueryClient } from '@tanstack/react-query';

import { ApiError, apiClient } from '@/lib/api-client';
import { RuntimeTokenMutationSchema } from '@/types/api.schema';
import type { RuntimeConfig } from '@/types/api';
import { setRuntimeConfigConflict } from './runtime-config-conflict';
import { RUNTIME_CONFIG_QUERY_KEY } from './use-runtime-config';

type SecretKind = 'llm-gateway' | 'api-access'; // pragma: allowlist secret

/** Direct writes deliberately bypass TanStack mutations so tokens never enter its cache. */
export const useRuntimeSecrets = () => {
  const client = useQueryClient();
  const write = async (kind: SecretKind, expectedRevision: number, token: string) => {
    try {
      const saved = await apiClient.put(
        `/api/config/credentials/${kind}`,
        { expected_revision: expectedRevision, token },
        RuntimeTokenMutationSchema,
      );
      client.setQueryData<RuntimeConfig>(RUNTIME_CONFIG_QUERY_KEY, (current) => {
        if (!current) return current;
        const llm = kind === 'llm-gateway'
          ? { ...current.document.llm, gateway_credential_configured: saved.configured, gateway_credential_updated_at: saved.updated_at }
          : current.document.llm;
        const security = kind === 'api-access'
          ? { ...current.document.security, access_credential_configured: saved.configured, access_credential_updated_at: saved.updated_at }
          : current.document.security;
        return { ...current, revision: saved.revision, updated_at: saved.updated_at, document: { ...current.document, llm, security } };
      });
      return saved;
    } catch (error) {
      if (error instanceof ApiError && error.status === 409) setRuntimeConfigConflict(client);
      throw error;
    }
  };
  return {
    writeLlmGateway: (revision: number, token: string) => write('llm-gateway', revision, token),
    writeApiAccess: (revision: number, token: string) => write('api-access', revision, token),
  };
};

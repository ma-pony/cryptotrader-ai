import { useQueryClient } from '@tanstack/react-query';

import { ApiError, apiClient } from '@/lib/api-client';
import { RuntimeConfigSchema, RuntimeTokenMutationSchema } from '@/types/api.schema';
import type { RuntimeConfig } from '@/types/api';
import { setRuntimeConfigConflict } from './runtime-config-conflict';
import { RUNTIME_CONFIG_QUERY_KEY } from './use-runtime-config';
import { useSettingsStore } from '@/stores/use-settings-store';

type SecretKind = 'llm-gateway' | 'api-access' | 'news-provider'; // pragma: allowlist secret

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
      void client.invalidateQueries({ queryKey: ['runtime-status'] });
      void client.invalidateQueries({ queryKey: ['trading-scope'] });
      // Subsequent authenticated requests must use the acknowledged key. The
      // settings store intentionally keeps it in memory only, never storage.
      if (kind === 'api-access') useSettingsStore.getState().setApiKey(token);
      client.setQueryData<RuntimeConfig>(RUNTIME_CONFIG_QUERY_KEY, (current) => {
        if (!current) return current;
        const llm =
          kind === 'llm-gateway'
            ? {
                ...current.document.llm,
                gateway_credential_configured: saved.configured,
                gateway_credential_updated_at: saved.updated_at,
              }
            : current.document.llm;
        const security =
          kind === 'api-access'
            ? {
                ...current.document.security,
                access_credential_configured: saved.configured,
                access_credential_updated_at: saved.updated_at,
              }
            : current.document.security;
        return {
          ...current,
          revision: saved.revision,
          updated_at: saved.updated_at,
          document: {
            ...current.document,
            llm,
            security,
            market_data:
              kind === 'news-provider'
                ? {
                    ...current.document.market_data,
                    news_credential_configured: saved.configured,
                    news_credential_updated_at: saved.updated_at,
                  }
                : current.document.market_data,
          },
        };
      });
      // The public document owns application status. Acknowledged credentials
      // are committed even if this follow-up read fails; never retry blindly.
      try {
        await client.fetchQuery({
          queryKey: RUNTIME_CONFIG_QUERY_KEY,
          queryFn: () => apiClient.get('/api/config', RuntimeConfigSchema),
          staleTime: 0,
          retry: false,
        });
        return { ...saved, savedNeedsReload: false };
      } catch {
        setRuntimeConfigConflict(client);
        return { ...saved, savedNeedsReload: true };
      }
    } catch (error) {
      // Failed publication or a lost response can follow persistence. Require
      // an authoritative reload instead of treating cached application as current.
      if (!(error instanceof ApiError) || error.status === 409 || error.status >= 500) setRuntimeConfigConflict(client);
      throw error;
    }
  };
  return {
    writeLlmGateway: (revision: number, token: string) => write('llm-gateway', revision, token),
    writeNewsProvider: (revision: number, token: string) => write('news-provider', revision, token),
    writeApiAccess: (revision: number, token: string) => write('api-access', revision, token),
  };
};

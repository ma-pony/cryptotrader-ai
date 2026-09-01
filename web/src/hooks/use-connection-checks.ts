import { useQueries, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '@/lib/api-client';
import { connectionFingerprint, type Connection, type ConnectionCheck } from '@/lib/configuration-readiness';
import { ConnectionHealthSchema } from '@/types/api.schema';

export function useConnectionChecks(
  connections: Connection[],
  credentials: Record<string, { configured: boolean; updatedAt: string | null }>,
) {
  const client = useQueryClient();
  const entries = connections.map((connection) => {
    const fingerprint = connectionFingerprint(connection, credentials[connection.id]?.updatedAt);
    return { connection, fingerprint, queryKey: ['venue-connection-check', connection.id, fingerprint] };
  });
  const runCheck = (id: string) => apiClient.post(`/api/venue-connections/${id}/test`, {}, ConnectionHealthSchema);
  const queries = useQueries({
    queries: entries.map(({ connection, queryKey }) => ({
      queryKey,
      enabled: connection.enabled,
      staleTime: Infinity,
      retry: false,
      queryFn: async () => {
        const saved = await apiClient.get(
          `/api/venue-connections/${connection.id}/check`,
          ConnectionHealthSchema.nullable(),
        );
        return saved;
      },
    })),
  });
  const checks: Record<string, ConnectionCheck> = {};
  const checking: Record<string, boolean> = {};
  const checkErrors: Record<string, boolean> = {};
  entries.forEach(({ connection, fingerprint }, index) => {
    const query = queries[index]!;
    checking[connection.id] = query.isFetching;
    checkErrors[connection.id] = query.isError;
    if (query.data && !query.isFetching && !query.isError) checks[connection.id] = { fingerprint, health: query.data };
  });
  const checkConnection = async (id: string, current?: Connection, updatedAt?: string | null) => {
    const entry = current
      ? {
          connection: current,
          fingerprint: connectionFingerprint(current, updatedAt),
          queryKey: ['venue-connection-check', current.id, connectionFingerprint(current, updatedAt)],
        }
      : entries.find(({ connection }) => connection.id === id);
    if (!entry) return;
    // Explicit checks must never coalesce with the mount-only GET on this key.
    await client.cancelQueries({ queryKey: entry.queryKey });
    // A failed explicit POST must not leave a prior green read displayed.
    client.setQueryData(entry.queryKey, null);
    const health = await runCheck(id);
    client.setQueryData(entry.queryKey, health);
  };
  return { checks, checking, checkErrors, checkConnection };
}

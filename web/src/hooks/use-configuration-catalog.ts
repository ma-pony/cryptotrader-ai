import { useQueries, useQuery } from '@tanstack/react-query';
import type { Connection } from '@/lib/configuration-readiness';
import { apiClient } from '@/lib/api-client';
import { ConfigurationCatalogSchema, VenueEnvironmentDefinitionSchema } from '@/types/api.schema';

export const CONFIGURATION_CATALOG_QUERY_KEY = ['configuration-catalog'] as const;
export function useConfigurationCatalog() {
  return useQuery({
    queryKey: CONFIGURATION_CATALOG_QUERY_KEY,
    queryFn: () => apiClient.get('/api/config/catalog', ConfigurationCatalogSchema),
    staleTime: 60_000,
  });
}
export function useVenueEnvironmentDefinition(adapterId: string, environment: string) {
  return useQuery({
    queryKey: ['venue-environment-definition', adapterId, environment],
    enabled: Boolean(adapterId && environment),
    queryFn: () =>
      apiClient.get(
        `/api/config/catalog/venues/${encodeURIComponent(adapterId)}?environment=${encodeURIComponent(environment)}`,
        VenueEnvironmentDefinitionSchema,
      ),
    staleTime: 60_000,
  });
}
export function useVenueEnvironmentDefinitions(connections: Connection[]) {
  const queries = useQueries({
    queries: connections.map((connection) => ({
      queryKey: ['venue-environment-definition', connection.adapter_id, connection.environment],
      queryFn: () =>
        apiClient.get(
          `/api/config/catalog/venues/${encodeURIComponent(connection.adapter_id)}?environment=${encodeURIComponent(connection.environment)}`,
          VenueEnvironmentDefinitionSchema,
        ),
      staleTime: 60_000,
    })),
  });
  return Object.fromEntries(connections.map((connection, index) => [connection.id, queries[index]?.data]));
}

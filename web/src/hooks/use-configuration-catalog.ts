import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/lib/api-client';
import { ConfigurationCatalogSchema } from '@/types/api.schema';

export const CONFIGURATION_CATALOG_QUERY_KEY = ['configuration-catalog'] as const;
export function useConfigurationCatalog() {
  return useQuery({
    queryKey: CONFIGURATION_CATALOG_QUERY_KEY,
    queryFn: () => apiClient.get('/api/config/catalog', ConfigurationCatalogSchema),
    staleTime: 60_000,
  });
}

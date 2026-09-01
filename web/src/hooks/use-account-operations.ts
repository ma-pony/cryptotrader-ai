import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '@/lib/api-client';
import { AccountOperationAcceptedSchema, AccountOperationSchema } from '@/types/api.schema';
import { RUNTIME_CONFIG_QUERY_KEY } from './use-runtime-config';
import { z } from 'zod';

export function useRemoveAccount(id: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (revision: number) =>
      apiClient.delete(
        `/api/venue-connections/${encodeURIComponent(id)}?expected_revision=${revision}`,
        z.object({ revision: z.number(), connection_id: z.string() }).strict(),
      ),
    onSuccess: () =>
      Promise.all(
        [RUNTIME_CONFIG_QUERY_KEY, ['accounts'], ['portfolio-books']].map((queryKey) =>
          client.invalidateQueries({ queryKey }),
        ),
      ),
  });
}

export function useAccountOperation(id: string | null) {
  const client = useQueryClient();
  return useQuery({
    queryKey: ['account-operations', id],
    enabled: Boolean(id),
    queryFn: async () => {
      const operation = await apiClient.get(
        `/api/account-operations/${encodeURIComponent(id!)}`,
        AccountOperationSchema,
      );
      if (operation.status !== 'preparing' && operation.status !== 'executing') {
        await Promise.all(
          [RUNTIME_CONFIG_QUERY_KEY, ['accounts'], ['portfolio-books'], ['trading-scope']].map((queryKey) =>
            client.invalidateQueries({ queryKey }),
          ),
        );
      }
      return operation;
    },
    refetchInterval: (query) => (['preparing', 'executing'].includes(query.state.data?.status ?? '') ? 500 : false),
  });
}

export function useAccountOperations(connectionId: string) {
  const client = useQueryClient();
  const prepare = useMutation({
    mutationFn: (body: {
      pair: string;
      kind: 'cancel_orders' | 'flatten';
      expected_revision: number;
      confirm_stop: boolean;
    }) =>
      apiClient.post(
        `/api/accounts/${encodeURIComponent(connectionId)}/operations/prepare`,
        body,
        AccountOperationAcceptedSchema,
      ),
  });
  const execute = useMutation({
    mutationFn: ({ id, version }: { id: string; version: number }) =>
      apiClient.post(
        `/api/account-operations/${encodeURIComponent(id)}/execute`,
        { plan_version: version },
        AccountOperationAcceptedSchema,
      ),
    onSuccess: (_, { id }) => client.invalidateQueries({ queryKey: ['account-operations', id] }),
  });
  return { prepare, execute };
}

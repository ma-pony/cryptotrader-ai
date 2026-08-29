import { keepPreviousData, useQuery } from '@tanstack/react-query';
import { apiClient } from '@/lib/api-client';
import { CycleSchema, PaginatedCyclesSchema } from '@/types/api.schema';

export const useMultiVenueCycles = (page = 1, size = 20) => useQuery({ queryKey: ['cycles', page, size], queryFn: () => apiClient.get(`/api/cycles?page=${page}&size=${size}`, PaginatedCyclesSchema), placeholderData: keepPreviousData });
export const useMultiVenueCycle = (cycleId: string | undefined) => useQuery({ queryKey: ['cycle', cycleId], enabled: Boolean(cycleId), queryFn: () => apiClient.get(`/api/cycles/${cycleId}`, CycleSchema) });

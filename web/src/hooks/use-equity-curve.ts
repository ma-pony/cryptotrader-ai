import { useQuery } from '@tanstack/react-query';

import { apiClient } from '@/lib/api-client';
import type { RangeWindow } from '@/types/api';
import { EquityCurveSchema } from '@/types/api.schema';

export const useEquityCurve = (range: RangeWindow) =>
  useQuery({
    queryKey: ['equity-curve', range],
    queryFn: () => apiClient.get(`/api/portfolio/equity-curve?range=${range}`, EquityCurveSchema),
  });

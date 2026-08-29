import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/lib/api-client';
import { PortfolioBooksSchema } from '@/types/api.schema';

export const PORTFOLIO_BOOKS_QUERY_KEY = ['portfolio-books'] as const;
export const usePortfolioBooks = () => useQuery({
  queryKey: PORTFOLIO_BOOKS_QUERY_KEY,
  queryFn: () => apiClient.get('/api/portfolio/books', PortfolioBooksSchema),
  refetchInterval: 10_000,
});

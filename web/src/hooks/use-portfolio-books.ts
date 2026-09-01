import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/lib/api-client';
import { PortfolioBooksSchema, AccountBookSchema } from '@/types/api.schema';

export const PORTFOLIO_BOOKS_QUERY_KEY = ['portfolio-books'] as const;
export const usePortfolioBooks = () =>
  useQuery({
    queryKey: PORTFOLIO_BOOKS_QUERY_KEY,
    queryFn: () => apiClient.get('/api/portfolio/books', PortfolioBooksSchema),
    refetchInterval: 10_000,
  });

export const usePortfolioBook = (id: string) =>
  useQuery({
    queryKey: [...PORTFOLIO_BOOKS_QUERY_KEY, id],
    queryFn: () => apiClient.get(`/api/portfolio/books/${encodeURIComponent(id)}`, AccountBookSchema),
    enabled: Boolean(id),
  });

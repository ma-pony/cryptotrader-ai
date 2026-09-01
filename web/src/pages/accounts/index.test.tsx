import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router';
import { describe, expect, it, vi } from 'vitest';

vi.mock('@/hooks/use-accounts', () => ({ useAccounts: vi.fn() }));
vi.mock('@/hooks/use-portfolio-books', () => ({ usePortfolioBooks: vi.fn() }));
vi.mock('@/hooks/use-configuration-catalog', () => ({ useConfigurationCatalog: vi.fn() }));

import { useAccounts } from '@/hooks/use-accounts';
import { usePortfolioBooks } from '@/hooks/use-portfolio-books';
import { useConfigurationCatalog } from '@/hooks/use-configuration-catalog';
import AccountsPage from './index';

describe('accounts facts', () => {
  it('falls back to actual adapter and environment identifiers when catalog labels are unavailable', () => {
    vi.mocked(useAccounts).mockReturnValue({
      isPending: false,
      isError: false,
      data: {
        items: [{
          connection_id: 'raw-id', label: '实际账户', adapter_id: 'adapter-actual', environment: 'environment-actual',
          capital_scope: 'simulated', enabled: true, book_ids: [], snapshot: null, failure_reason: null,
        }],
        simulated: [], real: [],
      },
    } as unknown as ReturnType<typeof useAccounts>);
    vi.mocked(usePortfolioBooks).mockReturnValue({
      data: { simulated: { books: [] }, real: { books: [] } }, isPending: false, isError: false,
    } as unknown as ReturnType<typeof usePortfolioBooks>);
    vi.mocked(useConfigurationCatalog).mockReturnValue({ data: undefined, isPending: false, isError: true } as unknown as ReturnType<typeof useConfigurationCatalog>);
    render(<MemoryRouter><AccountsPage /></MemoryRouter>);
    expect(screen.getByText(/adapter-actual · environment-actual · 未分配资金池/)).toBeVisible();
    expect(screen.queryByText(/待核对/)).not.toBeInTheDocument();
  });
});

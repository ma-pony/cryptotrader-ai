import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import i18n from '@/lib/i18n';
import DecisionsPage from '@/pages/decisions';
import { CycleSchema } from '@/types/api.schema';

const cycle = CycleSchema.parse({
  cycle_id: 'cycle-page-two', config_revision: 3, market_data_source_id: 'market', cycle_status: 'partial', execution_status: 'partial', requires_attention: true, created_at: '2026-08-29T00:00:00Z', books: [],
  shared_signals: { components: [], fused: null, target_position: null },
});
const response = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
const renderPage = () => render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}><MemoryRouter><DecisionsPage /></MemoryRouter></QueryClientProvider>);

describe('decisions canonical cycles projection', () => {
  beforeEach(async () => { await i18n.changeLanguage('en-US'); });
  afterEach(() => vi.unstubAllGlobals());

  it('uses canonical cycle pagination and keeps previous/next product controls', async () => {
    const fetchMock = vi.fn((url: string, init?: RequestInit) => {
      if (String(url) === '/api/cycles?page=1&size=20') return Promise.resolve(response({ items: [cycle], total: 21, page: 1, size: 20, has_next: true }));
      if (String(url) === '/api/cycles?page=2&size=20') return Promise.resolve(response({ items: [{ ...cycle, cycle_id: 'cycle-page-two-next' }], total: 21, page: 2, size: 20, has_next: false }));
      throw new Error(`unexpected request ${String(url)} ${init?.method}`);
    });
    vi.stubGlobal('fetch', fetchMock);
    renderPage();
    expect(await screen.findByRole('link', { name: /cycle-page-two/ })).toHaveAttribute('href', '/cycles/cycle-page-two');
    expect(screen.getByText('Page 1')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Previous' })).toBeDisabled();
    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: 'Next' }));
    expect(await screen.findByRole('link', { name: /cycle-page-two-next/ })).toHaveAttribute('href', '/cycles/cycle-page-two-next');
    expect(screen.getByText('Page 2')).toBeInTheDocument();
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith('/api/cycles?page=2&size=20', expect.objectContaining({ method: 'GET' })));
    expect(fetchMock.mock.calls.some(([url]) => String(url).includes('/api/decisions'))).toBe(false);
    await user.click(screen.getByRole('button', { name: 'Previous' }));
    expect(await screen.findByText('Page 1')).toBeInTheDocument();
  });

  it.each([
    ['loading', new Promise<Response>(() => undefined), 'Loading cycles…'],
    ['error', response({ detail: 'Internal server error' }, 500), 'Unable to load cycles.'],
    ['empty', response({ items: [], total: 0, page: 1, size: 20, has_next: false }), 'No cycles recorded.'],
  ])('renders a localized %s state', async (_state, result, expected) => {
    vi.stubGlobal('fetch', vi.fn().mockReturnValue(result));
    renderPage();
    expect(await screen.findByText(expected)).toBeInTheDocument();
  });
});

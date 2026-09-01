import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import i18n from '@/lib/i18n';
import DecisionsPage from '@/pages/decisions';
import { DecisionListSchema, DecisionSchema } from '@/types/api.schema';

const decision = DecisionSchema.parse({
  decision_id: 'decision-first-page',
  pair: 'BTC/USDT:USDT',
  mode: 'analysis',
  origin: 'manual',
  config_revision: 3,
  config_snapshot: [],
  status: 'completed',
  created_at: '2026-08-29T00:00:00Z',
  finished_at: '2026-08-29T00:01:00Z',
  components: [],
  fusion: null,
  target: null,
  books: [],
  failure: null,
  incomplete_fields: [],
});
const response = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
const renderPage = () =>
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <MemoryRouter>
        <DecisionsPage />
      </MemoryRouter>
    </QueryClientProvider>,
  );

describe('decisions persisted decision projection', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('en-US');
  });
  afterEach(() => vi.unstubAllGlobals());

  it('uses strict decision offset pagination and keeps previous/next product controls', async () => {
    const fetchMock = vi.fn((url: string, init?: RequestInit) => {
      if (String(url) === '/api/decisions?offset=0&limit=20')
        return Promise.resolve(
          response(DecisionListSchema.parse({ items: [decision], total: 21, offset: 0, limit: 20, has_next: true })),
        );
      if (String(url) === '/api/decisions?offset=20&limit=20')
        return Promise.resolve(
          response(
            DecisionListSchema.parse({
              items: [{ ...decision, decision_id: 'decision-second-page', pair: 'ETH/USDT:USDT' }],
              total: 21,
              offset: 20,
              limit: 20,
              has_next: false,
            }),
          ),
        );
      throw new Error(`unexpected request ${String(url)} ${init?.method}`);
    });
    vi.stubGlobal('fetch', fetchMock);
    renderPage();
    expect(await screen.findByRole('link', { name: /BTC\/USDT:USDT/ })).toHaveAttribute(
      'href',
      '/decisions/decision-first-page',
    );
    expect(screen.getByText('Page 1')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Previous' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Next' })).toBeEnabled();
    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: 'Next' }));
    expect(await screen.findByRole('link', { name: /ETH\/USDT:USDT/ })).toHaveAttribute(
      'href',
      '/decisions/decision-second-page',
    );
    expect(screen.getByText('Page 2')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Previous' })).toBeEnabled();
    expect(screen.getByRole('button', { name: 'Next' })).toBeDisabled();
    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(
        '/api/decisions?offset=20&limit=20',
        expect.objectContaining({ method: 'GET' }),
      ),
    );
    expect(fetchMock.mock.calls.some(([url]) => String(url).includes('/api/cycles'))).toBe(false);
    await user.click(screen.getByRole('button', { name: 'Previous' }));
    expect(await screen.findByText('Page 1')).toBeInTheDocument();
    expect(await screen.findByRole('link', { name: /BTC\/USDT:USDT/ })).toHaveAttribute(
      'href',
      '/decisions/decision-first-page',
    );
    expect(screen.getByRole('button', { name: 'Previous' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Next' })).toBeEnabled();
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/decisions?offset=0&limit=20',
      expect.objectContaining({ method: 'GET' }),
    );
  });

  it.each([
    ['loading', new Promise<Response>(() => undefined), 'Loading cycles…'],
    ['error', response({ detail: 'Internal server error' }, 500), 'Unable to load cycles.'],
    [
      'empty',
      response(DecisionListSchema.parse({ items: [], total: 0, offset: 0, limit: 20, has_next: false })),
      'No cycles recorded.',
    ],
  ])('renders a localized %s state', async (_state, result, expected) => {
    vi.stubGlobal('fetch', vi.fn().mockReturnValue(result));
    renderPage();
    expect(await screen.findByText(expected)).toBeInTheDocument();
  });
});

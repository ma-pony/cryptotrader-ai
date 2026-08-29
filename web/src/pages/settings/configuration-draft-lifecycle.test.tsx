import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import '@/lib/i18n';

import { runtimeConfigFixture } from '@/test/runtime-config-fixture';
import VenuesPage from './venues';
import ExecutionBooksPage from './execution-books';
import { VenueForm } from './venues/venue-form';
import { setRuntimeConfigConflict } from '@/hooks/runtime-config-conflict';

describe('configuration draft lifecycle', () => {
  const venueConfig = (label: string) => {
    const base = runtimeConfigFixture();
    return runtimeConfigFixture({
      document: {
        ...base.document,
        execution: {
          ...base.document.execution,
          connections: [{ id: 'paper', label, adapter_id: 'paper', environment: 'paper', enabled: true, credential_configured: false, credential_updated_at: null, leverage: 1, margin_mode: 'cross', parameters: [] }],
        },
      },
    });
  };
  it('does not render credential inputs for a Paper connection', () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<QueryClientProvider client={client}><VenueForm revision={1} connection={{ id: 'paper', label: 'Paper', adapter_id: 'paper', environment: 'paper', enabled: true, leverage: 1, margin_mode: 'cross', parameters: {} } as const} /></QueryClientProvider>);
    expect(screen.queryByLabelText('API Key')).not.toBeInTheDocument();
  });
  it('retains a dirty venue label when the same connection props refresh', () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const connection = { id: 'paper', label: 'Paper', adapter_id: 'paper', environment: 'paper' as const, enabled: true, leverage: 1, margin_mode: 'cross' as const, parameters: {} };
    const view = render(<QueryClientProvider client={client}><VenueForm revision={1} connection={connection} /></QueryClientProvider>);
    fireEvent.change(screen.getByLabelText('名称'), { target: { value: 'Dirty name' } });
    view.rerender(<QueryClientProvider client={client}><VenueForm revision={2} connection={{ ...connection, label: 'Server name' }} /></QueryClientProvider>);
    expect(screen.getByLabelText('名称')).toHaveValue('Dirty name');
  });

  it.each([
    ['venues', <VenuesPage />],
    ['execution books', <ExecutionBooksPage />],
  ])('offers an explicit reload control for a %s draft', async (_name, page) => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(new Response(JSON.stringify(runtimeConfigFixture()), { status: 200 })),
    );
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<QueryClientProvider client={client}>{page}</QueryClientProvider>);
    expect(await screen.findByRole('button', { name: '重新加载' })).toBeInTheDocument();
  });

  it('keeps a venue draft after a failed explicit reload', async () => {
    vi.stubGlobal('fetch', vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify(venueConfig('Paper')), { status: 200 }))
      .mockResolvedValueOnce(new Response('unavailable', { status: 503 })));
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<QueryClientProvider client={client}><VenuesPage /></QueryClientProvider>);
    const label = await screen.findByLabelText('名称');
    fireEvent.change(label, { target: { value: 'Unsaved' } });
    fireEvent.click(screen.getByRole('button', { name: '重新加载' }));
    await waitFor(() => expect(label).toHaveValue('Unsaved'));
  });

  it('resets a venue draft only after a successful explicit reload', async () => {
    vi.stubGlobal('fetch', vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify(venueConfig('Paper')), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify(venueConfig('Reloaded')), { status: 200 })));
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<QueryClientProvider client={client}><VenuesPage /></QueryClientProvider>);
    fireEvent.change(await screen.findByLabelText('名称'), { target: { value: 'Unsaved' } });
    fireEvent.click(screen.getByRole('button', { name: '重新加载' }));
    await waitFor(() => expect(screen.getByLabelText('名称')).toHaveValue('Reloaded'));
  });

  it('unblocks venue writes only after the page explicit reload receives a fresh snapshot', async () => {
    const base = venueConfig('Paper');
    vi.stubGlobal('fetch', vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify(base), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify(runtimeConfigFixture({ revision: 2, document: base.document })), { status: 200 })));
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<QueryClientProvider client={client}><VenuesPage /></QueryClientProvider>);
    await screen.findByLabelText('名称');
    setRuntimeConfigConflict(client);
    await waitFor(() => expect(screen.getByRole('button', { name: '保存连接' })).toBeDisabled());
    fireEvent.click(screen.getByRole('button', { name: '重新加载' }));
    await waitFor(() => expect(screen.getByRole('button', { name: '保存连接' })).toBeEnabled());
  });
});

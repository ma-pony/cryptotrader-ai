import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import '@/lib/i18n';
import { RUNTIME_CONFIG_QUERY_KEY } from '@/hooks/use-runtime-config';
import { runtimeConfigFixture } from '@/test/runtime-config-fixture';

import { AllocationPreview } from './allocation-preview';
import { validateBooks } from './book-form';
import ExecutionBooksPage from './index';

describe('ExecutionBooksPage', () => {
  it('blocks mixed capital scopes and previews exact weighted notionals', () => {
    render(<AllocationPreview equity={100000} targetExposure={0.5} allocations={[{ connectionId: 'paper', label: 'Paper', weight: 40 }, { connectionId: 'demo', label: 'Demo', weight: 60 }]} />);
    expect(screen.getByText('20,000 USDT')).toBeInTheDocument();
    expect(screen.getByText('30,000 USDT')).toBeInTheDocument();
    expect(validateBooks([{ id: 'sim', label: '模拟资金池', capital_scope: 'simulated', enabled: true, hitl_required: true, allocations: [{ connection_id: 'live', enabled: true, weight: 1 }] }], [{ id: 'live', label: 'Live', adapter_id: 'okx', environment: 'live', enabled: true, canary_only: false, leverage: 1, margin_mode: 'cross', parameters: {} }])).toContainEqual({ code: 'invalidSimulated' });
  });

  const configWithBook = (label = 'Server book') => {
    const base = runtimeConfigFixture();
    return runtimeConfigFixture({ document: {
      ...base.document,
      execution: {
        ...base.document.execution,
        connections: [{ id: 'paper', label: 'Paper', adapter_id: 'paper', environment: 'paper', enabled: true, credential_configured: false, credential_updated_at: null, leverage: 1, margin_mode: 'cross', parameters: [] }],
        books: [{ id: 'sim', label, capital_scope: 'simulated', enabled: true, hitl_required: false, allocations: [{ connection_id: 'paper', enabled: true, weight: 1 }] }],
      },
    } });
  };

  const renderPage = (client: QueryClient) => render(<QueryClientProvider client={client}><ExecutionBooksPage /></QueryClientProvider>);

  it('keeps a dirty book label when background runtime cache data changes', async () => {
    const initial = configWithBook();
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify(initial), { status: 200 })));
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    renderPage(client);
    const label = await screen.findByLabelText('名称');
    fireEvent.change(label, { target: { value: 'Dirty book' } });
    client.setQueryData(RUNTIME_CONFIG_QUERY_KEY, configWithBook('Background replacement'));
    await waitFor(() => expect(label).toHaveValue('Dirty book'));
  });

  it('retains a dirty book draft after failed explicit reload', async () => {
    vi.stubGlobal('fetch', vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify(configWithBook()), { status: 200 }))
      .mockResolvedValueOnce(new Response('down', { status: 503 })));
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    renderPage(client);
    const label = await screen.findByLabelText('名称');
    fireEvent.change(label, { target: { value: 'Dirty book' } });
    fireEvent.click(screen.getByRole('button', { name: '重新加载' }));
    await waitFor(() => expect(label).toHaveValue('Dirty book'));
  });

  it('resets a book draft only after a successful explicit reload', async () => {
    vi.stubGlobal('fetch', vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify(configWithBook()), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify(configWithBook('Reloaded book')), { status: 200 })));
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    renderPage(client);
    fireEvent.change(await screen.findByLabelText('名称'), { target: { value: 'Dirty book' } });
    fireEvent.click(screen.getByRole('button', { name: '重新加载' }));
    await waitFor(() => expect(screen.getByLabelText('名称')).toHaveValue('Reloaded book'));
  });

  it('disables save when allocation validation reports an error', async () => {
    const invalid = configWithBook();
    invalid.document.execution.books[0]!.allocations[0]!.weight = 0.5;
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify(invalid), { status: 200 })));
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    renderPage(client);
    expect(await screen.findByRole('button', { name: '保存完整配置' })).toBeDisabled();
  });
});

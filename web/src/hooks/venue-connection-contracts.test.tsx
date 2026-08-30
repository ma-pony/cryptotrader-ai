import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { fireEvent, render, renderHook, screen, waitFor } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { useVenueConnections } from './use-venue-connections';
import { VenueForm } from '@/pages/settings/venues/venue-form';
import '@/lib/i18n';
import i18n from '@/lib/i18n';
import { RUNTIME_CONFIG_QUERY_KEY } from './use-runtime-config';
import { RUNTIME_CONFIG_CONFLICT_QUERY_KEY } from './runtime-config-conflict';
import { runtimeConfigFixture } from '@/test/runtime-config-fixture';

describe('venue connection write recovery', () => {
  it('keeps a successful create as saved when the follow-up config refresh fails', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ revision: 2, connection: { id: 'paper-1', label: 'Paper', adapter_id: 'paper', environment: 'paper', enabled: true, canary_only: false, credential_configured: false, credential_updated_at: null, leverage: 1, margin_mode: 'cross', parameters: [] } }), { status: 201 }))
      .mockResolvedValueOnce(new Response('down', { status: 503 }));
    vi.stubGlobal('fetch', fetchMock);
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const hook = renderHook(() => useVenueConnections(), { wrapper: ({ children }) => <QueryClientProvider client={client}>{children}</QueryClientProvider> });
    const result = await hook.result.current.create.mutateAsync({ expected_revision: 1, id: 'paper-1', label: 'Paper', adapter_id: 'paper', environment: 'paper', enabled: true, canary_only: false, leverage: 1, margin_mode: 'cross', parameters: {} });
    expect(result.savedNeedsReload).toBe(true);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it('clears successful non-Paper credentials without caching secret variables when refresh fails', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ revision: 2, credential: { configured: true, updated_at: '2026-08-29T00:00:00Z' } }), { status: 200 }))
      .mockResolvedValueOnce(new Response('down', { status: 503 }));
    vi.stubGlobal('fetch', fetchMock);
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const base = runtimeConfigFixture();
    client.setQueryData(RUNTIME_CONFIG_QUERY_KEY, runtimeConfigFixture({ document: { ...base.document, execution: { ...base.document.execution, connections: [{ id: 'okx-demo', label: 'OKX Demo', adapter_id: 'okx', environment: 'demo', enabled: true, canary_only: false, credential_configured: false, credential_updated_at: null, leverage: 1, margin_mode: 'cross', parameters: [] }] } } }));
    render(
      <QueryClientProvider client={client}>
        <VenueForm revision={1} connection={{ id: 'okx-demo', label: 'OKX Demo', adapter_id: 'okx', environment: 'demo', enabled: true, canary_only: false, leverage: 1, margin_mode: 'cross', parameters: {} }} />
      </QueryClientProvider>,
    );
    fireEvent.change(screen.getByLabelText('访问 ID'), { target: { value: 'secret-marker-key' } });
    fireEvent.change(screen.getByLabelText('签名短语'), { target: { value: 'secret-marker-signing-key' } });
    fireEvent.change(screen.getByLabelText('Passphrase'), { target: { value: 'secret-marker-passphrase' } });
    fireEvent.click(screen.getByRole('button', { name: '保存访问资料' }));
    await waitFor(() => expect(screen.getByLabelText('访问 ID')).toHaveValue(''));
    expect(screen.getByLabelText('签名短语')).toHaveValue('');
    expect(screen.getByLabelText('Passphrase')).toHaveValue('');
    expect(client.getQueryData(RUNTIME_CONFIG_QUERY_KEY)).toMatchObject({ revision: 2, document: { execution: { connections: [{ id: 'okx-demo', credential_configured: true, credential_updated_at: '2026-08-29T00:00:00Z' }] } } });
    expect(screen.getByRole('status')).toHaveTextContent('连接已保存，但配置刷新失败；请重新加载后继续。');
    expect(screen.getByRole('button', { name: '保存访问资料' })).toBeDisabled();
    expect(JSON.stringify(client.getMutationCache().getAll().map((mutation) => mutation.state.variables))).not.toContain('secret-marker');
  });

  it('clears venue credentials after a failure or conflict and leaves the conflict recoverable', async () => {
    const marker = 'venue-rotation-secret-marker';
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ code: 'FAILED', message: marker }), { status: 500 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ code: 'REVISION_CONFLICT', message: marker }), { status: 409 }));
    vi.stubGlobal('fetch', fetchMock);
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <VenueForm revision={7} connection={{ id: 'okx-demo', label: 'OKX Demo', adapter_id: 'okx', environment: 'demo', enabled: true, canary_only: false, leverage: 1, margin_mode: 'cross', parameters: {} }} />
      </QueryClientProvider>,
    );
    const write = async () => {
      fireEvent.change(screen.getByLabelText('访问 ID'), { target: { value: `${marker}-key` } });
      fireEvent.change(screen.getByLabelText('签名短语'), { target: { value: `${marker}-signing` } });
      fireEvent.change(screen.getByLabelText('Passphrase'), { target: { value: `${marker}-phrase` } });
      fireEvent.click(screen.getByRole('button', { name: '保存访问资料' }));
      await waitFor(() => expect(screen.getByLabelText('访问 ID')).toHaveValue(''));
      expect(screen.getByLabelText('签名短语')).toHaveValue('');
      expect(screen.getByLabelText('Passphrase')).toHaveValue('');
      expect(document.body.textContent).not.toContain(marker);
      expect(JSON.stringify(client.getQueryData(RUNTIME_CONFIG_QUERY_KEY)) ?? '').not.toContain(marker);
      expect(JSON.stringify(client.getMutationCache().getAll().map((mutation) => mutation.state.variables))).not.toContain(marker);
    };
    await write();
    await write();
    await waitFor(() => expect(client.getQueryData(RUNTIME_CONFIG_CONFLICT_QUERY_KEY)).toBe(true));
    expect(fetchMock.mock.calls).toHaveLength(2);
    expect(String(fetchMock.mock.calls[1]![0])).toContain('/api/venue-connections/okx-demo/credentials');
    expect(JSON.parse((fetchMock.mock.calls[1]![1] as RequestInit).body as string)).toEqual({
      expected_revision: 7,
      credentials: { api_key: `${marker}-key`, secret: `${marker}-signing`, passphrase: `${marker}-phrase` },
    });
  });

  it('shows saved reload required and prevents a second create after a refresh failure', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ revision: 2, connection: { id: 'paper-2', label: 'Paper Two', adapter_id: 'paper', environment: 'paper', enabled: true, canary_only: false, credential_configured: false, credential_updated_at: null, leverage: 1, margin_mode: 'cross', parameters: [] } }), { status: 201 }))
      .mockResolvedValueOnce(new Response('down', { status: 503 }));
    vi.stubGlobal('fetch', fetchMock);
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<QueryClientProvider client={client}><VenueForm revision={1} /></QueryClientProvider>);
    fireEvent.change(screen.getByLabelText('连接 ID'), { target: { value: 'paper-2' } });
    fireEvent.change(screen.getByLabelText('名称'), { target: { value: 'Paper Two' } });
    fireEvent.click(screen.getByRole('button', { name: '创建连接' }));
    expect(await screen.findByRole('status')).toHaveTextContent('连接已保存，但配置刷新失败；请重新加载后继续。');
    const create = screen.getByRole('button', { name: '创建连接' });
    expect(create).toBeDisabled();
    fireEvent.click(create);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it('serializes isolated custom adapter parameters as ordinary JSON and blocks invalid parameter text', async () => {
    await i18n.changeLanguage('en-US');
    const fetchMock = vi.fn().mockResolvedValueOnce(new Response(JSON.stringify({ revision: 2, connection: { id: 'custom-1', label: 'Custom', adapter_id: 'custom', environment: 'demo', enabled: true, canary_only: false, credential_configured: false, credential_updated_at: null, leverage: 1, margin_mode: 'isolated', parameters: [] } }), { status: 201 })).mockResolvedValueOnce(new Response(JSON.stringify({ revision: 2, updated_at: 'x', setup_required: false, document: {} }), { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<QueryClientProvider client={client}><VenueForm revision={1} /></QueryClientProvider>);
    fireEvent.change(screen.getByLabelText('Connection ID'), { target: { value: 'custom-1' } });
    fireEvent.change(screen.getByLabelText('Name'), { target: { value: 'Custom' } });
    fireEvent.change(screen.getByLabelText('Adapter ID'), { target: { value: 'custom' } });
    fireEvent.change(screen.getByLabelText('Environment'), { target: { value: 'demo' } });
    fireEvent.change(screen.getByLabelText('Margin mode'), { target: { value: 'isolated' } });
    fireEvent.change(screen.getByLabelText('Connection parameters JSON'), { target: { value: '{"region":"sg"}' } });
    fireEvent.click(screen.getByRole('button', { name: 'Create connection' }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    const body = JSON.parse((fetchMock.mock.calls[0]![1] as RequestInit).body as string);
    expect(body).toMatchObject({ margin_mode: 'isolated', parameters: { region: 'sg' } });
  });

  it('persists the dedicated canary checkbox, shows its warning, and hydrates it after reload', async () => {
    await i18n.changeLanguage('en-US');
    const saved = { id: 'canary-1', label: 'Canary', adapter_id: 'bybit', environment: 'testnet', enabled: true, canary_only: true, credential_configured: false, credential_updated_at: null, leverage: 1, margin_mode: 'cross', parameters: [] };
    const fetchMock = vi.fn().mockResolvedValueOnce(new Response(JSON.stringify({ revision: 2, connection: saved }), { status: 201 })).mockResolvedValueOnce(new Response(JSON.stringify({ revision: 2, updated_at: 'x', setup_required: false, document: {} }), { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const view = render(<QueryClientProvider client={client}><VenueForm revision={1} /></QueryClientProvider>);
    fireEvent.change(screen.getByLabelText('Connection ID'), { target: { value: 'canary-1' } });
    fireEvent.change(screen.getByLabelText('Name'), { target: { value: 'Canary' } });
    fireEvent.change(screen.getByLabelText('Adapter ID'), { target: { value: 'bybit' } });
    fireEvent.change(screen.getByLabelText('Environment'), { target: { value: 'testnet' } });
    fireEvent.click(screen.getByLabelText('Canary-only validation'));
    expect(screen.getByText('For Canary validation only; never use this platform account or credentials manually or from another system.')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Create connection' }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    expect(JSON.parse((fetchMock.mock.calls[0]![1] as RequestInit).body as string)).toMatchObject({ canary_only: true });
    view.rerender(<QueryClientProvider client={client}><VenueForm revision={2} connection={{ id: 'canary-2', label: 'Reloaded', adapter_id: 'bybit', environment: 'testnet', enabled: true, canary_only: true, leverage: 1, margin_mode: 'cross', parameters: {} }} /></QueryClientProvider>);
    expect(screen.getByLabelText('Canary-only validation')).toBeChecked();
  });
});

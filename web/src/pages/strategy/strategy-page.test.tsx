import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import StrategyPage from './index';
import '@/lib/i18n';
import { runtimeConfigFixture } from '@/test/runtime-config-fixture';
import { RUNTIME_CONFIG_QUERY_KEY } from '@/hooks/use-runtime-config';
import { useSettingsStore } from '@/stores/use-settings-store';

describe('StrategyPage', () => {
  it('offers an explicit reload control while a strategy draft is open', async () => {
    const base = runtimeConfigFixture();
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify(runtimeConfigFixture({ document: { ...base.document, signals: { ...base.document.signals, components: [{ component_id: 'kronos', enabled: true, weight: 1, parameters: [] }] } } })), { status: 200 })));
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<QueryClientProvider client={client}><StrategyPage/></QueryClientProvider>);
    expect(await screen.findByRole('button', { name: '重新加载' })).toBeInTheDocument();
  });

  it('loads components from RuntimeConfig instead of a profile endpoint', async () => {
    const base = runtimeConfigFixture();
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify(runtimeConfigFixture({ revision: 3, document: { ...base.document, signals: { ...base.document.signals, components: [{ component_id: 'kronos', enabled: true, weight: 1, parameters: [] }] } } })), { status: 200 })));
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<QueryClientProvider client={client}><StrategyPage/></QueryClientProvider>);
    expect(await screen.findByText('kronos')).toBeInTheDocument();
    expect(fetch).toHaveBeenCalledWith(expect.stringContaining('/api/config'), expect.anything());
  });

  it('disables save for invalid component parameter JSON', async () => {
    const base = runtimeConfigFixture();
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify(runtimeConfigFixture({ document: { ...base.document, signals: { ...base.document.signals, components: [{ component_id: 'kronos', enabled: true, weight: 1, parameters: [] }] } } })), { status: 200 })));
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<QueryClientProvider client={client}><StrategyPage/></QueryClientProvider>);
    fireEvent.change(await screen.findByLabelText('kronos 参数'), { target: { value: '{bad' } });
    expect(screen.getByRole('button', { name: '保存完整配置' })).toBeDisabled();
  });

  it('keeps the live-order gate in a strategy save and saved runtime reload', async () => {
    const base = runtimeConfigFixture();
    const initial = runtimeConfigFixture({ document: { ...base.document, execution: { ...base.document.execution, live_order_execution_enabled: true }, signals: { ...base.document.signals, components: [{ component_id: 'kronos', enabled: true, weight: 1, parameters: [] }] } } });
    const saved = runtimeConfigFixture({ revision: 2, document: initial.document });
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify(initial), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify(saved), { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<QueryClientProvider client={client}><StrategyPage/></QueryClientProvider>);
    fireEvent.change(await screen.findByLabelText('kronos 参数'), { target: { value: '{"window": 20}' } });
    fireEvent.change(screen.getByLabelText('tech_agent'), { target: { value: 'new-model' } });
    fireEvent.change(screen.getByLabelText('中性阈值'), { target: { value: '30' } });
    fireEvent.click(screen.getByRole('button', { name: '保存完整配置' }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    const body = JSON.parse((fetchMock.mock.calls[1]![1] as RequestInit).body as string);
    expect(body).toMatchObject({ expected_revision: 1, document: { execution: { live_order_execution_enabled: true }, signals: { neutral_threshold: 0.3, components: [{ component_id: 'kronos', parameters: { window: 20 } }] }, llm: { models: { tech_agent: 'new-model' } } } });
    expect(client.getQueryData(RUNTIME_CONFIG_QUERY_KEY)).toMatchObject({ revision: 2, document: { execution: { live_order_execution_enabled: true } } });
  });

  it('keeps a strategy draft on background cache updates and failed reload, then resets only on successful reload', async () => {
    const base = runtimeConfigFixture();
    const config = (model: string) => runtimeConfigFixture({ document: { ...base.document, signals: { ...base.document.signals, components: [{ component_id: 'kronos', enabled: true, weight: 1, parameters: [] }] }, llm: { ...base.document.llm, models: { ...base.document.llm.models, tech_agent: model } } } });
    vi.stubGlobal('fetch', vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify(config('server')), { status: 200 }))
      .mockResolvedValueOnce(new Response('down', { status: 503 }))
      .mockResolvedValueOnce(new Response(JSON.stringify(config('reloaded')), { status: 200 })));
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<QueryClientProvider client={client}><StrategyPage/></QueryClientProvider>);
    const model = await screen.findByLabelText('tech_agent');
    fireEvent.change(model, { target: { value: 'draft' } });
    client.setQueryData(RUNTIME_CONFIG_QUERY_KEY, config('background'));
    await waitFor(() => expect(model).toHaveValue('draft'));
    fireEvent.click(screen.getByRole('button', { name: '重新加载' }));
    await waitFor(() => expect(model).toHaveValue('draft'));
    fireEvent.click(screen.getByRole('button', { name: '重新加载' }));
    await waitFor(() => expect(screen.getByLabelText('tech_agent')).toHaveValue('reloaded'));
  });

  it('keeps the strategy draft and renders an explicit conflict after a 409 save', async () => {
    const base = runtimeConfigFixture();
    const initial = runtimeConfigFixture({ document: { ...base.document, signals: { ...base.document.signals, components: [{ component_id: 'kronos', enabled: true, weight: 1, parameters: [] }] } } });
    vi.stubGlobal('fetch', vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify(initial), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ code: 'REVISION_CONFLICT', message: 'stale revision' }), { status: 409 })));
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<QueryClientProvider client={client}><StrategyPage/></QueryClientProvider>);
    const model = await screen.findByLabelText('tech_agent');
    fireEvent.change(model, { target: { value: 'draft-after-conflict' } });
    fireEvent.click(screen.getByRole('button', { name: '保存完整配置' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('配置已被其他操作更新，请重新加载');
    expect(model).toHaveValue('draft-after-conflict');
  });

  it('rotates gateway credentials through the page, clears every outcome, and never caches the token', async () => {
    await import('@/lib/i18n').then(({ default: i18n }) => i18n.changeLanguage('en-US'));
    const base = runtimeConfigFixture({ document: { ...runtimeConfigFixture().document, signals: { ...runtimeConfigFixture().document.signals, components: [{ component_id: 'kronos', enabled: true, weight: 1, parameters: [] }] } } });
    const token = 'strategy-secret-marker';
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify(base), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ revision: 2, configured: true, updated_at: '2026-08-30T00:00:00Z' }), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ code: 'FAILED', message: token }), { status: 500 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ code: 'REVISION_CONFLICT', message: token }), { status: 409 }));
    vi.stubGlobal('fetch', fetchMock);
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<QueryClientProvider client={client}><StrategyPage/></QueryClientProvider>);
    const input = await screen.findByLabelText('LLM gateway key');
    fireEvent.change(input, { target: { value: token } });
    fireEvent.click(screen.getByRole('button', { name: 'Rotate gateway key' }));
    await waitFor(() => expect(input).toHaveValue(''));
    expect(JSON.parse((fetchMock.mock.calls[1]![1] as RequestInit).body as string)).toEqual({ expected_revision: 1, token });
    expect(String(fetchMock.mock.calls[1]![0])).toContain('/api/config/credentials/llm-gateway');
    expect(JSON.stringify(client.getQueryData(RUNTIME_CONFIG_QUERY_KEY))).not.toContain(token);
    expect(document.body.textContent).not.toContain(token);
    fireEvent.change(input, { target: { value: token } });
    fireEvent.click(screen.getByRole('button', { name: 'Rotate gateway key' }));
    await waitFor(() => expect(input).toHaveValue(''));
    expect(document.body.textContent).not.toContain(token);
    fireEvent.change(input, { target: { value: token } });
    fireEvent.click(screen.getByRole('button', { name: 'Rotate gateway key' }));
    await waitFor(() => expect(input).toHaveValue(''));
    expect(JSON.stringify(client.getMutationCache().getAll().map((item) => item.state.variables))).not.toContain(token);
    useSettingsStore.getState().reset();
  });
});

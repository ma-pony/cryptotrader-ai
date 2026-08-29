import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router';
import { describe, expect, it, vi } from 'vitest';

import i18n from '@/lib/i18n';
import { App } from '@/App';
import SetupPage, { testFingerprint, validateLlmAdvanced, validateRiskSection } from './index';
import { runtimeConfigFixture } from '@/test/runtime-config-fixture';
import { RUNTIME_CONFIG_QUERY_KEY } from '@/hooks/use-runtime-config';
import { useSettingsStore } from '@/stores/use-settings-store';

describe('SetupPage', () => {
  it('rejects malformed or unknown risk JSON instead of accepting an arbitrary object', () => {
    expect(validateRiskSection({})).toBeUndefined();
    expect(validateRiskSection({ ...runtimeConfigFixture().document.risk, unexpected: true })).toBeUndefined();
    expect(validateRiskSection(runtimeConfigFixture().document.risk)).toEqual(runtimeConfigFixture().document.risk);
  });
  it('accepts only the exact finite LLM advanced configuration shape', () => {
    const llm = runtimeConfigFixture().document.llm;
    const valid = { streaming_models: ['analysis'], retry: llm.retry, model_costs: [{ name: 'analysis', input_usd_per_mtok: 1, output_usd_per_mtok: 2 }], timeout_seconds: 30 };
    expect(validateLlmAdvanced(valid)).toEqual(valid);
    expect(validateLlmAdvanced({ ...valid, extra: true })).toBeUndefined();
    expect(validateLlmAdvanced({ ...valid, retry: { ...valid.retry, retry_jitter: 'yes' } })).toBeUndefined();
    expect(validateLlmAdvanced({ ...valid, retry: { ...valid.retry, max_attempts: 1.5 } })).toBeUndefined();
    expect(validateLlmAdvanced({ ...valid, model_costs: [{ ...valid.model_costs[0], input_usd_per_mtok: Infinity }] })).toBeUndefined();
  });

  it('routes setup_required users into the ordered setup flow', async () => {
    await i18n.changeLanguage('zh-CN');
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify(runtimeConfigFixture({ setup_required: true, document: { ...runtimeConfigFixture().document, system: { active: false } } })), { status: 200 })));
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<QueryClientProvider client={client}><MemoryRouter initialEntries={['/']}><App /></MemoryRouter></QueryClientProvider>);
    expect(await screen.findByRole('heading', { name: '初始化交易系统' })).toBeInTheDocument();
    expect(screen.getByRole('list')).toHaveTextContent(/LLM.*信号组件.*行情来源.*平台连接.*执行资金池.*风控与审批.*调度器.*测试并激活/);
  });

  it('walks all eight setup stages and keeps activation disabled without a tested enabled connection', async () => {
    await i18n.changeLanguage('zh-CN');
    const base = runtimeConfigFixture();
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify(runtimeConfigFixture({ setup_required: true, document: { ...base.document, system: { active: false } } })), { status: 200 })));
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<QueryClientProvider client={client}><SetupPage /></QueryClientProvider>);
    expect(await screen.findByRole('heading', { name: 'LLM' })).toBeInTheDocument();
    for (const _stage of ['信号组件', '行情来源', '平台连接', '执行资金池', '风控与审批', '调度器', '测试并激活']) {
      fireEvent.click(screen.getByRole('button', { name: '下一阶段' }));
    }
    expect(await screen.findByRole('button', { name: '测试并激活' })).toBeDisabled();
  });

  it('invalidates a connection test fingerprint when label, config, or credential state changes', () => {
    const connection = { id: 'paper', label: 'Paper', adapter_id: 'paper', environment: 'paper' as const, enabled: true, leverage: 1, margin_mode: 'cross' as const, parameters: { sandbox: true } };
    const fingerprint = testFingerprint(connection, '2026-08-29T00:00:00Z');
    expect(testFingerprint({ ...connection, label: 'Renamed' }, '2026-08-29T00:00:00Z')).not.toBe(fingerprint);
    expect(testFingerprint({ ...connection, parameters: { sandbox: false } }, '2026-08-29T00:00:00Z')).not.toBe(fingerprint);
    expect(testFingerprint(connection, '2026-08-29T00:01:00Z')).not.toBe(fingerprint);
  });

  it('writes Setup gateway credentials exactly, clears them after success, failure, and conflict, then reloads safely', async () => {
    useSettingsStore.getState().reset();
    await i18n.changeLanguage('en-US');
    const marker = 'setup-gateway-secret-marker';
    const base = runtimeConfigFixture({ setup_required: true, document: { ...runtimeConfigFixture().document, system: { active: false } } });
    const reloaded = runtimeConfigFixture({ revision: 4, setup_required: true, document: { ...base.document, llm: { ...base.document.llm, gateway_credential_configured: true, gateway_credential_updated_at: '2026-08-30T00:00:00Z' } } });
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify(base), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify(base), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ revision: 2, configured: true, updated_at: '2026-08-30T00:00:00Z' }), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ code: 'FAILED', message: marker }), { status: 500 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ code: 'REVISION_CONFLICT', message: marker }), { status: 409 }))
      .mockResolvedValueOnce(new Response(JSON.stringify(reloaded), { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<QueryClientProvider client={client}><SetupPage /></QueryClientProvider>);
    const input = await screen.findByLabelText('LLM gateway key');
    const write = async () => {
      fireEvent.change(input, { target: { value: marker } });
      fireEvent.click(screen.getByRole('button', { name: /gateway key/i }));
      await waitFor(() => expect(input).toHaveValue(''));
      expect(document.body.textContent).not.toContain(marker);
      expect(JSON.stringify(client.getQueryData(RUNTIME_CONFIG_QUERY_KEY)) ?? '').not.toContain(marker);
      expect(JSON.stringify(client.getMutationCache().getAll().map((mutation) => mutation.state.variables))).not.toContain(marker);
    };
    await write();
    const gatewayWrite = fetchMock.mock.calls.find(([url]) => String(url).includes('/credentials/llm-gateway'))!;
    expect(JSON.parse((gatewayWrite[1] as RequestInit).body as string)).toEqual({ expected_revision: 1, token: marker });
    expect(client.getQueryData(RUNTIME_CONFIG_QUERY_KEY)).toMatchObject({ revision: 2, document: { llm: { gateway_credential_configured: true, gateway_credential_updated_at: '2026-08-30T00:00:00Z' } } });
    await write();
    await write();
    expect(await screen.findByText('Configuration was updated elsewhere. Reload to continue.')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Reload' }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(6));
    useSettingsStore.getState().reset();
  });

  it('keeps API access tokens in memory only and clears failed Chinese writes before conflict recovery', async () => {
    useSettingsStore.getState().reset();
    await i18n.changeLanguage('zh-CN');
    const marker = 'setup-api-secret-marker';
    const base = runtimeConfigFixture({ setup_required: true, document: { ...runtimeConfigFixture().document, system: { active: false } } });
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify(base), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify(base), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ revision: 2, configured: true, updated_at: '2026-08-30T00:00:00Z' }), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ code: 'FAILED', message: marker }), { status: 500 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ code: 'REVISION_CONFLICT', message: marker }), { status: 409 }))
      .mockResolvedValueOnce(new Response(JSON.stringify(runtimeConfigFixture({ revision: 4, setup_required: true, document: { ...base.document, security: { enabled: false, access_credential_configured: true, access_credential_updated_at: '2026-08-30T00:00:00Z' } } })), { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<QueryClientProvider client={client}><SetupPage /></QueryClientProvider>);
    for (let stage = 0; stage < 7; stage += 1) fireEvent.click(await screen.findByRole('button', { name: '下一阶段' }));
    const input = await screen.findByLabelText('API 访问密钥');
    const write = async () => {
      fireEvent.change(input, { target: { value: marker } });
      fireEvent.click(screen.getByRole('button', { name: /API 访问密钥/ }));
      await waitFor(() => expect(input).toHaveValue(''));
      expect(document.body.textContent).not.toContain(marker);
      expect(JSON.stringify(client.getQueryData(RUNTIME_CONFIG_QUERY_KEY)) ?? '').not.toContain(marker);
      expect(JSON.stringify(client.getMutationCache().getAll().map((mutation) => mutation.state.variables))).not.toContain(marker);
    };
    await write();
    expect(useSettingsStore.getState().apiKey).toBe(marker);
    const apiWrite = fetchMock.mock.calls.find(([url]) => String(url).includes('/credentials/api-access'))!;
    expect(JSON.parse((apiWrite[1] as RequestInit).body as string)).toEqual({ expected_revision: 1, token: marker });
    expect(client.getQueryData(RUNTIME_CONFIG_QUERY_KEY)).toMatchObject({ revision: 2, document: { security: { access_credential_configured: true, access_credential_updated_at: '2026-08-30T00:00:00Z' } } });
    await write();
    await write();
    expect(await screen.findByText('配置已被其他操作更新，请重新加载')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: '重新加载' }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(6));
    const reloadedRequest = fetchMock.mock.calls.at(-1)!;
    expect(new Headers((reloadedRequest[1] as RequestInit).headers).get('X-API-Key')).toBe(marker);
    useSettingsStore.getState().reset();
  });

  it('activates only after every enabled connection is tested and PUTs the complete current document', async () => {
    await i18n.changeLanguage('zh-CN');
    const base = runtimeConfigFixture();
    const document = {
      ...base.document,
      system: { active: false }, security: { enabled: true, access_credential_configured: false, access_credential_updated_at: null },
      signals: { ...base.document.signals, components: [{ component_id: 'kronos', enabled: true, weight: 1, parameters: [] }] },
      execution: { ...base.document.execution, connections: [{ id: 'paper', label: 'Paper', adapter_id: 'paper', environment: 'paper', enabled: true, credential_configured: false, credential_updated_at: null, leverage: 1, margin_mode: 'cross', parameters: [] }], books: [{ id: 'sim', label: 'Sim', capital_scope: 'simulated', enabled: true, hitl_required: false, allocations: [{ connection_id: 'paper', enabled: true, weight: 1 }] }] },
    };
    const initial = runtimeConfigFixture({ setup_required: true, document });
    const saved = runtimeConfigFixture({ revision: 3, setup_required: false, document: { ...document, system: { active: true }, security: { enabled: true, access_credential_configured: true, access_credential_updated_at: '2026-08-30T00:00:00Z' } } });
    const fetchMock = vi.fn((url: string, init?: RequestInit) => {
      if (url.includes('/test')) return Promise.resolve(new Response(JSON.stringify({ connection_id: 'paper', healthy: true, environment: 'paper', credential_configured: false, capabilities: { market_types: [], native_protection: false, hedge_mode: false, reduce_only: true, supported_order_types: [] } }), { status: 200 }));
      if (url.includes('/credentials/api-access')) return Promise.resolve(new Response(JSON.stringify({ revision: 2, configured: true, updated_at: '2026-08-30T00:00:00Z' }), { status: 200 }));
      if (init?.method === 'PUT') return Promise.resolve(new Response(JSON.stringify(saved), { status: 200 }));
      return Promise.resolve(new Response(JSON.stringify(initial), { status: 200 }));
    });
    vi.stubGlobal('fetch', fetchMock);
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<QueryClientProvider client={client}><SetupPage /></QueryClientProvider>);
    await screen.findByRole('heading', { name: 'LLM' });
    for (let stage = 2; stage <= 4; stage += 1) fireEvent.click(screen.getByRole('button', { name: '下一阶段' }));
    fireEvent.click(await screen.findByRole('button', { name: '测试连接' }));
    await waitFor(() => expect(fetchMock.mock.calls.some(([url]) => String(url).includes('/test'))).toBe(true));
    for (let stage = 5; stage <= 8; stage += 1) fireEvent.click(screen.getByRole('button', { name: '下一阶段' }));
    fireEvent.change(await screen.findByLabelText('API 访问密钥'), { target: { value: 'activation-api-key' } });
    fireEvent.click(screen.getByRole('button', { name: '保存 API 访问密钥' }));
    await waitFor(() => expect(fetchMock.mock.calls.some(([url]) => String(url).includes('/credentials/api-access'))).toBe(true));
    const activate = await screen.findByRole('button', { name: '测试并激活' });
    expect(activate).toBeEnabled();
    fireEvent.click(activate);
    await waitFor(() => expect(fetchMock.mock.calls.some(([, init]) => init?.method === 'PUT')).toBe(true));
    const put = fetchMock.mock.calls.find(([url, init]) => String(url).endsWith('/api/config') && init?.method === 'PUT')!;
    expect(new Headers((put[1] as RequestInit).headers).get('X-API-Key')).toBe('activation-api-key');
    const body = JSON.parse((put[1] as RequestInit).body as string);
    expect(body).toMatchObject({ expected_revision: 2, document: { system: { active: true }, signals: { components: [{ component_id: 'kronos', parameters: {} }] }, execution: { books: [{ id: 'sim', allocations: [{ connection_id: 'paper', weight: 1 }] }] } } });
    fireEvent.change(screen.getByLabelText('API 访问密钥'), { target: { value: 'rotated-api-key' } });
    fireEvent.click(screen.getByRole('button', { name: '轮换 API 访问密钥' }));
    await waitFor(() => expect(fetchMock.mock.calls.filter(([url]) => String(url).includes('/credentials/api-access'))).toHaveLength(2));
    const subsequent = fetchMock.mock.calls.filter(([url]) => String(url).includes('/credentials/api-access'))[1]!;
    expect(new Headers((subsequent[1] as RequestInit).headers).get('X-API-Key')).toBe('activation-api-key');
  });
});

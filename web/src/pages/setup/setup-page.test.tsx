import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router';
import { describe, expect, it, vi } from 'vitest';

import i18n from '@/lib/i18n';
import { App } from '@/App';
import SetupPage, { testFingerprint, validateLlmAdvanced, validateRiskSection } from './index';
import { runtimeConfigFixture } from '@/test/runtime-config-fixture';

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

  it('activates only after every enabled connection is tested and PUTs the complete current document', async () => {
    await i18n.changeLanguage('zh-CN');
    const base = runtimeConfigFixture();
    const document = {
      ...base.document,
      system: { active: false },
      signals: { ...base.document.signals, components: [{ component_id: 'kronos', enabled: true, weight: 1, parameters: [] }] },
      execution: { ...base.document.execution, connections: [{ id: 'paper', label: 'Paper', adapter_id: 'paper', environment: 'paper', enabled: true, credential_configured: false, credential_updated_at: null, leverage: 1, margin_mode: 'cross', parameters: [] }], books: [{ id: 'sim', label: 'Sim', capital_scope: 'simulated', enabled: true, hitl_required: false, allocations: [{ connection_id: 'paper', enabled: true, weight: 1 }] }] },
    };
    const initial = runtimeConfigFixture({ setup_required: true, document });
    const saved = runtimeConfigFixture({ revision: 2, setup_required: false, document: { ...document, system: { active: true } } });
    const fetchMock = vi.fn((url: string, init?: RequestInit) => {
      if (url.includes('/test')) return Promise.resolve(new Response(JSON.stringify({ connection_id: 'paper', healthy: true, environment: 'paper', credential_configured: false, capabilities: { market_types: [], native_protection: false, hedge_mode: false, reduce_only: true, supported_order_types: [] } }), { status: 200 }));
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
    const activate = await screen.findByRole('button', { name: '测试并激活' });
    expect(activate).toBeEnabled();
    fireEvent.click(activate);
    await waitFor(() => expect(fetchMock.mock.calls.some(([, init]) => init?.method === 'PUT')).toBe(true));
    const put = fetchMock.mock.calls.find(([, init]) => init?.method === 'PUT')!;
    const body = JSON.parse((put[1] as RequestInit).body as string);
    expect(body).toMatchObject({ expected_revision: 1, document: { system: { active: true }, signals: { components: [{ component_id: 'kronos', parameters: {} }] }, execution: { books: [{ id: 'sim', allocations: [{ connection_id: 'paper', weight: 1 }] }] } } });
  });
});

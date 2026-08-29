import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import StrategyPage from './index';
import '@/lib/i18n';
import { runtimeConfigFixture } from '@/test/runtime-config-fixture';

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
});

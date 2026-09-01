import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen } from '@testing-library/react';
import { MemoryRouter, useParams } from 'react-router';
import { describe, expect, it, vi } from 'vitest';

import { runtimeConfigFixture } from '@/test/runtime-config-fixture';
import '@/lib/i18n';

vi.mock('@/components/layout/app-shell', async () => {
  const { Outlet } = await import('react-router');
  return { AppShell: () => <Outlet /> };
});
vi.mock('@/pages/decisions/detail', () => ({
  default: function DecisionProbe() {
    return <div>decision-id:{useParams<{ decisionId?: string }>().decisionId}</div>;
  },
}));
vi.mock('@/pages/research/backtest-detail', () => ({
  default: function ResearchProbe() {
    return <div>research-run:{useParams<{ runId?: string }>().runId}</div>;
  },
}));
vi.mock('@/pages/accounts', () => ({ default: () => <h1>账户事实探针</h1> }));
vi.mock('@/pages/accounts/connection-detail', () => ({ default: () => <h1>连接事实探针</h1> }));
vi.mock('@/pages/accounts/book-detail', () => ({ default: () => <h1>资金池事实探针</h1> }));
vi.mock('@/pages/engine/component-detail', () => ({ default: () => <h1>组件历史探针</h1> }));
vi.mock('@/pages/settings/metrics', () => ({ default: () => <h1>指标事实探针</h1> }));
vi.mock('@/pages/settings/agent-profiles', () => ({ default: () => <h1>代理资料探针</h1> }));

import { App } from './App';

describe('named application routes', () => {
  it.each([
    ['/decisions/cycle-42', 'decision-id:cycle-42'],
    ['/research/backtests/run-45', 'research-run:run-45'],
  ])('passes named URL params through the rendered page for %s', async (path, expected) => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(new Response(JSON.stringify(runtimeConfigFixture()), { status: 200 })),
    );
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <MemoryRouter initialEntries={[path]}>
          <App />
        </MemoryRouter>
      </QueryClientProvider>,
    );
    expect(await screen.findByText(expected)).toBeInTheDocument();
  });

  it('shows the workbench at the root instead of redirecting to setup', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(new Response(JSON.stringify(runtimeConfigFixture()), { status: 200 })),
    );
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <MemoryRouter initialEntries={['/']}>
          <App />
        </MemoryRouter>
      </QueryClientProvider>,
    );
    expect(await screen.findByRole('heading', { name: '工作台' })).toBeInTheDocument();
  });

  it.each([
    '/setup',
    '/dashboard',
    '/cycles/cycle-42',
    '/chat/session-42',
    '/debate/cycle-42',
    '/risk',
    '/strategy',
    '/backtest',
    '/metrics',
    '/memory',
    '/market',
    '/settings/execution-books',
  ])('returns the not-found page for removed legacy link %s', async (path) => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(new Response(JSON.stringify(runtimeConfigFixture()), { status: 200 })),
    );
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <MemoryRouter initialEntries={[path]}>
          <App />
        </MemoryRouter>
      </QueryClientProvider>,
    );
    expect(await screen.findByText('404')).toBeInTheDocument();
  });

  it.each([
    ['/accounts', '账户事实探针'],
    ['/accounts/connections/paper', '连接事实探针'],
    ['/accounts/books/sim', '资金池事实探针'],
    ['/engine/components/kronos', '组件历史探针'],
    ['/settings/metrics', '指标事实探针'],
    ['/settings/agent-profiles', '代理资料探针'],
  ])('keeps read-only facts visible when configuration and catalog cannot be read: %s', async (path, expected) => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation((input: RequestInfo | URL) => {
        const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
        if (url.includes('/api/config/catalog'))
          return Promise.resolve(new Response(JSON.stringify({ detail: 'unavailable' }), { status: 503 }));
        if (url.includes('/api/config'))
          return Promise.resolve(new Response(JSON.stringify({ detail: 'unauthorized' }), { status: 401 }));
        return Promise.resolve(new Response(JSON.stringify({}), { status: 200 }));
      }),
    );
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <MemoryRouter initialEntries={[path]}>
          <App />
        </MemoryRouter>
      </QueryClientProvider>,
    );
    expect(await screen.findByRole('heading', { name: expected })).toBeInTheDocument();
  });
});

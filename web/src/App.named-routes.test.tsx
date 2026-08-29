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
vi.mock('@/pages/decisions', () => ({
  default: function DecisionProbe() { return <div>decision-cycle:{useParams<{ cycleId?: string }>().cycleId}</div>; },
}));
vi.mock('@/pages/debate', () => ({
  default: function DebateProbe() { return <div>debate-cycle:{useParams<{ cycleId?: string }>().cycleId}</div>; },
}));
vi.mock('@/pages/chat', () => ({
  default: function ChatProbe() { return <div>chat-session:{useParams<{ sessionId?: string }>().sessionId}</div>; },
}));

import { App } from './App';

describe('named application routes', () => {
  it.each([
    ['/decisions/cycle-42', 'decision-cycle:cycle-42'],
    ['/debate/cycle-43', 'debate-cycle:cycle-43'],
    ['/chat/session-44', 'chat-session:session-44'],
  ])('passes named URL params through the rendered page for %s', async (path, expected) => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify(runtimeConfigFixture()), { status: 200 })));
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<QueryClientProvider client={client}><MemoryRouter initialEntries={[path]}><App /></MemoryRouter></QueryClientProvider>);
    expect(await screen.findByText(expected)).toBeInTheDocument();
  });
});

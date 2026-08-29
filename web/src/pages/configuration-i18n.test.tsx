import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { App } from '@/App';
import i18n from '@/lib/i18n';
import ExecutionBooksPage from '@/pages/settings/execution-books';
import VenuesPage from '@/pages/settings/venues';
import StrategyPage from '@/pages/strategy';
import { runtimeConfigFixture } from '@/test/runtime-config-fixture';

const renderWithConfig = (page: React.ReactNode, config = runtimeConfigFixture()) => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify(config), { status: 200 })));
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={client}>{page}</QueryClientProvider>);
};

describe('configuration control plane in English', () => {
  afterEach(async () => {
    vi.unstubAllGlobals();
    await i18n.changeLanguage('zh-CN');
  });

  it('renders the setup-required flow in English', async () => {
    await i18n.changeLanguage('en-US');
    const base = runtimeConfigFixture();
    renderWithConfig(
      <MemoryRouter initialEntries={['/']}><App /></MemoryRouter>,
      runtimeConfigFixture({ setup_required: true, document: { ...base.document, system: { active: false } } }),
    );
    expect(await screen.findByRole('heading', { name: 'Commission trading system' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Next stage' })).toBeInTheDocument();
  });

  it('renders venue controls and fields in English', async () => {
    await i18n.changeLanguage('en-US');
    renderWithConfig(<VenuesPage />);
    expect(await screen.findByRole('heading', { name: 'Venue connections' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Add connection' })).toBeInTheDocument();
  });

  it('renders execution book controls in English', async () => {
    await i18n.changeLanguage('en-US');
    renderWithConfig(<ExecutionBooksPage />);
    expect(await screen.findByRole('heading', { name: 'Execution books' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Add execution book' })).toBeInTheDocument();
  });

  it('renders strategy controls in English', async () => {
    await i18n.changeLanguage('en-US');
    renderWithConfig(<StrategyPage />);
    expect(await screen.findByRole('heading', { name: 'Signal fusion strategy' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Save full configuration' })).toBeInTheDocument();
  });
});

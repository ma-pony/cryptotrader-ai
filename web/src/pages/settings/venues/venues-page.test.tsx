import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import i18n from '@/lib/i18n';
import VenuesPage from './index';
import { runtimeConfigFixture } from '@/test/runtime-config-fixture';

describe('VenuesPage', () => {
  it('shows configured state without ever rendering saved credential values', async () => {
    await i18n.changeLanguage('zh-CN');
    const base = runtimeConfigFixture();
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify(runtimeConfigFixture({ document: { ...base.document, execution: { ...base.document.execution, connections: [{ id: 'okx-demo', label: 'OKX Demo', adapter_id: 'okx', environment: 'demo', enabled: true, canary_only: false, credential_configured: true, credential_updated_at: null, leverage: 1, margin_mode: 'cross', parameters: [] }] } } })), { status: 200 })));
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<QueryClientProvider client={client}><VenuesPage /></QueryClientProvider>);
    expect(await screen.findByText('凭据已配置')).toBeInTheDocument();
    expect(screen.getByLabelText('访问 ID')).toHaveValue('');
    expect(document.body.textContent).not.toContain('secret-marker');
  });
});

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import i18n from '@/lib/i18n';
import VenuesPage from './index';

describe('VenuesPage', () => {
  it('shows configured state without ever rendering saved credential values', async () => {
    await i18n.changeLanguage('zh-CN');
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({ revision: 1, updated_at: '2026-08-28T00:00:00Z', setup_required: false, document: { system: { active: true }, market_data: { source_id: 'binance', parameters: [] }, llm: {}, signals: { components: [] }, risk: {}, execution: { connections: [{ id: 'okx-demo', label: 'OKX Demo', adapter_id: 'okx', environment: 'demo', enabled: true, credential_configured: true, credential_updated_at: null, leverage: 1, margin_mode: 'cross', parameters: [] }], books: [], allocation_policy: 'weighted' }, hitl: {}, scheduler: {}, triggers: {}, notifications: {}, infrastructure: {} } }), { status: 200 })));
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<QueryClientProvider client={client}><VenuesPage /></QueryClientProvider>);
    expect(await screen.findByText('凭据已配置')).toBeInTheDocument();
    expect(screen.getByLabelText('API Key')).toHaveValue('');
    expect(document.body.textContent).not.toContain('secret-marker');
  });
});

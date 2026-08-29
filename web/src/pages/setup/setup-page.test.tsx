import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router';
import { describe, expect, it, vi } from 'vitest';

import i18n from '@/lib/i18n';
import { App } from '@/App';

const document = {
  system: { active: false }, market_data: { source_id: 'binance', parameters: [] }, llm: {}, signals: { components: [] }, risk: {}, execution: { connections: [], books: [], allocation_policy: 'weighted' }, hitl: {}, scheduler: {}, triggers: {}, notifications: {}, infrastructure: {},
};

describe('SetupPage', () => {
  it('routes setup_required users into the ordered setup flow', async () => {
    await i18n.changeLanguage('zh-CN');
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({ setup_required: true, revision: 1, updated_at: '2026-08-28T00:00:00Z', document }), { status: 200 })));
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<QueryClientProvider client={client}><MemoryRouter initialEntries={['/']}><App /></MemoryRouter></QueryClientProvider>);
    expect(await screen.findByRole('heading', { name: '初始化交易系统' })).toBeInTheDocument();
    expect(screen.getByRole('list')).toHaveTextContent(/LLM.*信号组件.*行情来源.*平台连接.*执行资金池.*风控与审批.*调度器.*测试并激活/);
  });
});

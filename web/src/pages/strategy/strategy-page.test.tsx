import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import i18n from '@/lib/i18n';

import StrategyPage from './index';

const profile = (revision = 3) => ({
  revision,
  updated_at: '2026-08-28T01:23:45+00:00',
  components: [
    { component_id: 'kronos', enabled: true, weight: 0.6 },
    { component_id: 'llm_committee', enabled: true, weight: 0.4 },
  ],
  neutral_threshold: 0.2,
  max_target_ratio: 1,
  atr_stop_multiplier: 2,
  reward_ratio: 2,
  hitl_required: false,
  installed_components: [
    { component_id: 'kronos', display_name: 'Kronos', description: '时间序列基础模型' },
    { component_id: 'llm_committee', display_name: 'LLM 四智能体委员会', description: '四领域内部辩论' },
  ],
});

const renderPage = () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <StrategyPage />
    </QueryClientProvider>,
  );
};

afterEach(() => {
  vi.unstubAllGlobals();
});

beforeEach(async () => {
  await i18n.changeLanguage('zh-CN');
});

describe('StrategyPage', () => {
  it('disables save until enabled weights total 100%', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify(profile()), { status: 200 })));
    const user = userEvent.setup();
    renderPage();

    const kronosWeight = await screen.findByLabelText('Kronos 权重');
    await user.clear(kronosWeight);
    await user.type(kronosWeight, '50');

    expect(screen.getByText('权重总计 90%')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '保存并从下一周期生效' })).toBeDisabled();
  });

  it('saves the complete profile and renders the new revision', async () => {
    const fetchMock = vi.fn().mockImplementation((_input: RequestInfo | URL, init?: RequestInit) => {
      const response = init?.method === 'PUT' ? profile(4) : profile(3);
      return Promise.resolve(new Response(JSON.stringify(response), { status: 200 }));
    });
    vi.stubGlobal('fetch', fetchMock);
    const user = userEvent.setup();
    renderPage();

    expect(await screen.findByText('Revision 3')).toBeInTheDocument();
    expect(screen.getByText(/最后更新/)).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: '保存并从下一周期生效' }));

    expect(await screen.findByText('Revision 4')).toBeInTheDocument();
    expect(screen.getByText('已保存，从下一周期开始使用')).toBeInTheDocument();
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    const [, request] = fetchMock.mock.calls[1] as [RequestInfo | URL, RequestInit];
    expect(typeof request.body).toBe('string');
    const payload = JSON.parse(request.body as string);
    expect(payload).toEqual(
      expect.objectContaining({
        components: expect.arrayContaining([
          expect.objectContaining({ component_id: 'kronos', weight: 0.6 }),
        ]),
      }),
    );
    expect(payload).not.toHaveProperty('revision');
    expect(payload).not.toHaveProperty('updated_at');
  });
});

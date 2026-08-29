import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import i18n from '@/lib/i18n';
import { ApprovalRequestSchema, HitlRespondSchema } from '@/types/api.schema';

import { ApprovalQueueCard } from './approval-queue-card';

const approval = ApprovalRequestSchema.parse({
  approval_id: 'approval-1',
  cycle_id: 'cycle-partial',
  book_id: 'real-book',
  pair: 'ETH/USDT',
  config_revision: 8,
  status: 'pending',
  created_at: '2026-08-29T00:00:00Z',
  decided_at: null,
  proposal: {
    version: 1,
    book_id: 'real-book',
    capital_scope: 'real',
    config_revision: 8,
    pair: { symbol: 'ETH/USDT' },
    requested_target_exposure: '0.40',
    target_exposure: '0.35',
    risk: {
      passed: false,
      requested_target_exposure: '0.40',
      capped_target_exposure: '0.35',
      connection_weights: ['1.0'],
      connection_targets: [
        { book_id: 'real-book', connection_id: 'okx-live', weight: '1.0', book_equity: '30000', target_exposure: '0.35', target_signed_notional: '10500' },
      ],
      rejected_by: 'book-cap',
      reason: 'risk cap applied',
      cap_source: 'max_notional',
    },
    connection_risks: [{ connection_id: 'okx-live', passed: false, risk_increase: true, reason: 'leverage cap', operation: 'preflight' }],
    connection_plans: [{
      book_id: 'real-book', connection_id: 'okx-live', pair: { symbol: 'ETH/USDT' },
      current_signed_notional: '1000', target_signed_notional: '10500', delta_signed_notional: '9500',
      current_signed_amount: '0.25', target_signed_amount: '2.625', delta_signed_amount: '2.375', post_fill_signed_amount: '2.625',
      quote: { pair: { symbol: 'ETH/USDT' }, bid: '3999', ask: '4001', last: '4000' },
      execution_price: '4001', amount: '2.375', side: 'buy', reduce_only: false, market_type: 'swap',
      stop_loss: '3800', take_profit: '4400', old_protection_ids: ['sl-old', 'tp-old'],
      capabilities: { market_types: ['swap'], native_protection: true, hedge_mode: false, reduce_only: true, supported_order_types: ['market', 'stop_market'] },
    }],
    unavailable_connections: ['bybit-live'],
    errors: ['bybit-live unavailable'],
    ready: false,
  },
});

const renderQueue = () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <ApprovalQueueCard />
      </MemoryRouter>
    </QueryClientProvider>,
  );
};

const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
const apiError = (status: number) => json({ code: `HTTP_${status}`, message: 'request failed' }, status);

describe('approval book plan', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('zh-CN');
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('rejects response fields outside the strict HITL outcome contract', () => {
    expect(HitlRespondSchema.safeParse({
      approval_id: 'approval-1', cycle_id: 'cycle-partial', approval_status: 'approved', cycle_status: 'partial', execution_status: 'partial', requires_attention: true, leaked: true,
    }).success).toBe(false);
  });

  it('keeps the frozen complete proposal visible through a partial attention outcome after approval', async () => {
    const fetchMock = vi.fn((url: string, init?: RequestInit) => {
      if (String(url) === '/api/hitl/pending' && init?.method === 'GET') {
        return Promise.resolve(json(fetchMock.mock.calls.filter(([called]) => String(called) === '/api/hitl/pending').length === 1 ? [approval] : []));
      }
      if (String(url) === '/api/hitl/approval-1/respond' && init?.method === 'POST') {
        return Promise.resolve(json({ approval_id: 'approval-1', cycle_id: 'cycle-partial', approval_status: 'approved', cycle_status: 'partial', execution_status: 'partial', requires_attention: true }));
      }
      throw new Error(`unexpected request ${String(url)} ${init?.method}`);
    });
    vi.stubGlobal('fetch', fetchMock);

    renderQueue();
    await screen.findByText('real-book · ETH/USDT');
    for (const text of ['配置版本 8', '资金范围: real', '状态: 等待中', '创建时间: 2026-08-29T00:00:00Z', '请求敞口: 0.40', '目标敞口: 0.35', 'okx-live 0.35', 'okx-live leverage cap', 'bybit-live', 'bybit-live unavailable', 'okx-live · ETH/USDT', '3999/4001/4000', '数量: 2.375', '名义价值: 10500', '方向: buy', '只减仓: false', '保护: 3800/4400', '既有保护单: sl-old, tp-old', 'swap', 'market, stop_market']) {
      expect(screen.getAllByText(text, { exact: false }).length).toBeGreaterThan(0);
    }
    expect(screen.queryByRole('spinbutton')).not.toBeInTheDocument();

    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: '批准' }));
    await user.click(await screen.findByRole('button', { name: '确认批准' }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith('/api/hitl/approval-1/respond', expect.objectContaining({ method: 'POST' })));
    const post = fetchMock.mock.calls.find(([url, init]) => String(url) === '/api/hitl/approval-1/respond' && (init as RequestInit).method === 'POST');
    expect(JSON.parse((post?.[1] as RequestInit).body as string)).toEqual({ decision: 'approve' });
    expect(await screen.findByText('已批准 · 部分完成 · 部分完成')).toBeInTheDocument();
    expect(screen.getByText('需要关注')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: '查看周期' })).toHaveAttribute('href', '/cycles/cycle-partial');
    await i18n.changeLanguage('en-US');
    expect(await screen.findByText('Approved · Partial · Partial')).toBeInTheDocument();
    expect(screen.getByText('Requires attention')).toBeInTheDocument();
    expect(screen.queryByText('requires_attention')).not.toBeInTheDocument();
    await waitFor(() => expect(screen.getByText('No approvals pending.')).toBeInTheDocument());
    await user.click(screen.getByRole('button', { name: 'Dismiss' }));
    expect(screen.queryByText('Approved · Partial · Partial')).not.toBeInTheDocument();
  });

  it.each([409, 500])('keeps the proposal and reports a localized error when respond returns %i', async (status) => {
    const fetchMock = vi.fn((url: string, init?: RequestInit) => {
      if (String(url) === '/api/hitl/pending' && init?.method === 'GET') return Promise.resolve(json([approval]));
      if (String(url) === '/api/hitl/approval-1/respond' && init?.method === 'POST') return Promise.resolve(apiError(status));
      throw new Error(`unexpected request ${String(url)} ${init?.method}`);
    });
    vi.stubGlobal('fetch', fetchMock);
    renderQueue();
    await screen.findByText('real-book · ETH/USDT');
    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: '批准' }));
    await user.click(await screen.findByRole('button', { name: '确认批准' }));
    expect(await screen.findByText('审批响应失败，请求仍处于待处理状态。')).toBeInTheDocument();
    expect(screen.getByText('okx-live · ETH/USDT')).toBeInTheDocument();
    expect(screen.queryByText('已批准 · 部分完成 · 部分完成')).not.toBeInTheDocument();
  });

  it('has distinct loading, load-error, and empty queue states', async () => {
    const pending = new Promise<Response>(() => undefined);
    vi.stubGlobal('fetch', vi.fn().mockReturnValueOnce(pending));
    const { unmount } = renderQueue();
    expect(await screen.findByText('正在加载审批…')).toBeInTheDocument();
    unmount();

    vi.stubGlobal('fetch', vi.fn().mockResolvedValueOnce(apiError(500)));
    const failed = renderQueue();
    expect(await screen.findByText('无法加载审批。')).toBeInTheDocument();
    failed.unmount();

    vi.stubGlobal('fetch', vi.fn().mockResolvedValueOnce(json([])));
    renderQueue();
    expect(await screen.findByText('暂无待审批请求。')).toBeInTheDocument();
  });
});

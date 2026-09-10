import { fireEvent, render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('@/components/alerts/attention-list', () => ({ AttentionList: () => <p>当前没有需要关注的事项。</p> }));
vi.mock('@/components/trading/run-dialog', () => ({ RunDialog: () => <div>交易确认</div> }));
vi.mock('@/hooks/use-runtime-status', () => ({ useRuntimeStatus: vi.fn() }));
vi.mock('@/hooks/use-scheduler-status', () => ({ useSchedulerStatus: vi.fn() }));
vi.mock('@/hooks/use-decisions', () => ({ useDecisions: vi.fn(), useStartAnalysis: vi.fn() }));
vi.mock('@/hooks/use-accounts', () => ({ useAccounts: vi.fn() }));

import { useAccounts } from '@/hooks/use-accounts';
import { useDecisions, useStartAnalysis } from '@/hooks/use-decisions';
import { useRuntimeStatus } from '@/hooks/use-runtime-status';
import { useSchedulerStatus } from '@/hooks/use-scheduler-status';
import WorkbenchPage from './index';

const mutation = { isPending: false, isError: false, mutate: vi.fn() };

describe('workbench', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(useRuntimeStatus).mockReturnValue({
      isPending: false,
      isError: false,
      data: {
        analysis: { ready: true, reasons: [] },
        trading: {
          ready: false,
          reasons: [{ code: 'book_missing', message: '请先分配资金池', path: 'execution.books' }],
        },
        components: [],
        saved_revision: 7,
        applied_revision: 7,
        apply_error: null,
        automation_enabled: true,
        latest_run_at: null,
        execution_pairs: ['BTC/USDT:USDT'],
      },
    } as unknown as ReturnType<typeof useRuntimeStatus>);
    vi.mocked(useSchedulerStatus).mockReturnValue({
      isPending: false,
      isError: false,
      data: { enabled: false, next_pair: null, next_run_at: null, redis_available: true },
    } as unknown as ReturnType<typeof useSchedulerStatus>);
    vi.mocked(useStartAnalysis).mockReturnValue(mutation as unknown as ReturnType<typeof useStartAnalysis>);
    vi.mocked(useDecisions).mockReturnValue({
      isPending: false,
      isError: false,
      data: { items: [], total: 0, limit: 20, offset: 0, has_next: false },
    } as unknown as ReturnType<typeof useDecisions>);
    vi.mocked(useAccounts).mockReturnValue({
      isPending: false,
      isError: false,
      data: {
        items: [{ connection_id: 'paper', capital_scope: 'simulated', book_ids: [] }],
        simulated: [
          { amount: null, currency: 'USDT', unavailable_reason: '尚无账户快照', as_of: null, source: 'platform' },
        ],
        real: [],
      },
    } as unknown as ReturnType<typeof useAccounts>);
  });

  it('allows analysis only, explains the exact next field and never turns missing money into zero', () => {
    render(
      <MemoryRouter>
        <WorkbenchPage />
      </MemoryRouter>,
    );
    expect(screen.getByRole('heading', { name: '工作台' })).toBeVisible();
    expect(screen.getByRole('button', { name: '仅分析一次' })).toBeEnabled();
    expect(screen.getByRole('button', { name: '运行一次交易' })).toBeDisabled();
    expect(screen.getByText(/尚未分配到资金池/)).toBeVisible();
    expect(screen.getByText('字段：execution.books')).toBeVisible();
    expect(screen.getByRole('link', { name: '前往设置' })).toHaveAttribute('href', '/accounts/books#execution.books');
    expect(screen.getByText(/未知 USDT/)).toBeVisible();
    expect(screen.queryByText(/^0(?:\.0+)? USDT$/)).not.toBeInTheDocument();
  });

  it('describes enabled automation without claiming it is actively running', () => {
    render(
      <MemoryRouter>
        <WorkbenchPage />
      </MemoryRouter>,
    );
    expect(screen.getByText('自动运行已开启')).toBeVisible();
    expect(screen.getByText('当前没有已知运行中任务')).toBeVisible();
    expect(screen.queryByText('自动运行中')).not.toBeInTheDocument();
  });

  it.each([
    ['book_disabled', 'execution.books.sim.enabled', '/accounts/books#execution.books.sim'],
    ['no_allocations', 'execution.books.sim.allocations', '/accounts/books#execution.books.sim'],
    ['connection_disabled', 'execution.connections.paper', '/accounts/connections#execution.connections.paper'],
    ['execution_pairs_empty', 'execution.pairs', '/accounts/books#execution.pairs'],
    [
      'real_execution_not_authorized',
      'execution.live_order_execution_enabled',
      '/accounts/books#execution.live_order_execution_enabled',
    ],
    ['execution_lock_missing', 'infrastructure.redis_url', '/settings/security#infrastructure.redis_url'],
    ['future_reason', 'future.engine.option', '/engine#configuration'],
  ])('links readiness %s to its rendered configuration owner', (_code, path, href) => {
    vi.mocked(useRuntimeStatus).mockReturnValue({
      isPending: false,
      isError: false,
      data: {
        analysis: { ready: false, reasons: [{ code: _code, message: '需要处理', path }] },
        trading: { ready: false, reasons: [] },
        components: [],
        saved_revision: 7,
        applied_revision: 7,
        apply_error: null,
        automation_enabled: false,
        latest_run_at: null,
        execution_pairs: ['BTC/USDT:USDT'],
      },
    } as unknown as ReturnType<typeof useRuntimeStatus>);
    render(
      <MemoryRouter>
        <WorkbenchPage />
      </MemoryRouter>,
    );
    expect(screen.getByRole('link', { name: '前往设置' })).toHaveAttribute('href', href);
    expect(screen.getByText(`字段：${path}`)).toBeVisible();
  });

  it('keeps an explicitly cleared pair empty and submits only the replacement pair', () => {
    render(
      <MemoryRouter>
        <WorkbenchPage />
      </MemoryRouter>,
    );
    const input = screen.getByLabelText('交易对');
    const submit = screen.getByRole('button', { name: '仅分析一次' });
    expect(input).toHaveValue('BTC/USDT:USDT');
    fireEvent.change(input, { target: { value: '' } });
    expect(input).toHaveValue('');
    expect(submit).toBeDisabled();
    fireEvent.change(input, { target: { value: 'ETH/USDT:USDT' } });
    fireEvent.click(submit);
    expect(mutation.mutate).toHaveBeenCalledWith(
      { pair: 'ETH/USDT:USDT', expected_revision: 7 },
      expect.objectContaining({ onSuccess: expect.any(Function) }),
    );
  });

  it('submits only analysis when Enter is pressed in the pair field', async () => {
    render(
      <MemoryRouter>
        <WorkbenchPage />
      </MemoryRouter>,
    );
    await userEvent.type(screen.getByLabelText('交易对'), '{Enter}');
    expect(mutation.mutate).toHaveBeenCalledWith(
      { pair: 'BTC/USDT:USDT', expected_revision: 7 },
      expect.objectContaining({ onSuccess: expect.any(Function) }),
    );
    expect(screen.queryByText('交易确认')).not.toBeInTheDocument();
  });
});

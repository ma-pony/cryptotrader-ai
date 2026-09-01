import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, expect, it, vi } from 'vitest';
import { MemoryRouter } from 'react-router';
import type { ReactNode } from 'react';
import i18n from '@/lib/i18n';
import { BacktestForm } from './research/backtest-form';
import { RuleFormDialog } from './engine/automation/components/rule-form-dialog';
import { RuleTable } from './engine/automation/components/rule-table';
import type { ScheduleRule } from '@/types/api';
import { runtimeConfigFixture } from '@/test/runtime-config-fixture';
import { researchRun } from '@/test/research-fixture';

beforeEach(() => i18n.changeLanguage('zh-CN'));
const rule: ScheduleRule = {
  id: 'rule-1',
  name: 'BTC rule',
  trigger_type: 'price_threshold',
  pair: 'BTC/USDT',
  parameters: { direction: 'below', price: 50000 },
  enabled: true,
  cooldown_minutes: 60,
  created_at: '2026-08-20T00:00:00Z',
  updated_at: '2026-08-20T00:00:00Z',
  last_triggered_at: null,
  in_cooldown: false,
  ttl_expires_at: null,
  created_by: 'user',
  schedule_depth: 0,
};
function harness(element: ReactNode, fail = false, sessionFail = false) {
  const requests: { url: string; method: string; body: Record<string, unknown> }[] = [];
  vi.stubGlobal(
    'fetch',
    vi.fn((url: string, init?: RequestInit) => {
      const method = init?.method ?? 'GET';
      requests.push({ url, method, body: init?.body ? JSON.parse(init.body as string) : {} });
      if (url.endsWith('/api/config')) return Promise.resolve(new Response(JSON.stringify(runtimeConfigFixture())));
      const prior = researchRun({
        run_id: 'prior',
        params: {
          ...researchRun().params,
          pair: 'ETH/USDT',
          start: '2026-08-01',
          end: '2026-08-10',
          initial_equity: '4200',
        },
      });
      if (url.includes('/api/backtest/runs?'))
        return Promise.resolve(new Response(JSON.stringify({ items: [prior], limit: 20, offset: 0, has_next: false })));
      if (url.endsWith('/api/backtest/runs/prior'))
        return Promise.resolve(
          new Response(JSON.stringify(sessionFail ? { detail: 'unavailable' } : prior), {
            status: sessionFail ? 503 : 200,
          }),
        );
      return Promise.resolve(
        new Response(
          JSON.stringify(
            fail ? { detail: 'Unavailable' } : url.endsWith('/runs') ? { run_id: 'run-new', status: 'queued' } : rule,
          ),
          { status: fail ? 503 : 200 },
        ),
      );
    }),
  );
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <MemoryRouter>{element}</MemoryRouter>
    </QueryClientProvider>,
  );
  return requests;
}
function dates(start = '2026-08-01', end = '2026-08-10') {
  fireEvent.change(screen.getByLabelText('起始日期'), { target: { value: start } });
  fireEvent.change(screen.getByLabelText('结束日期'), { target: { value: end } });
}
it('prevents reversed backtest dates and focuses its linked end-date error', async () => {
  const requests = harness(<BacktestForm onRunStarted={vi.fn()} />);
  dates('2026-08-20', '2026-08-10');
  await waitFor(() => expect(screen.getByRole('button', { name: '运行回测' })).toBeEnabled());
  fireEvent.click(screen.getByRole('button', { name: '运行回测' }));
  expect(await screen.findByRole('alert')).toHaveTextContent('结束日期');
  expect(screen.getByLabelText('结束日期')).toHaveAttribute('aria-invalid', 'true');
  expect(screen.getByLabelText('结束日期')).toHaveFocus();
  expect(requests.filter((r) => r.url.endsWith('/runs'))).toHaveLength(0);
});
it('keeps a cleared capital blank and rejects future dates without starting', async () => {
  const requests = harness(<BacktestForm onRunStarted={vi.fn()} />);
  dates('2026-08-01', '2099-08-10');
  fireEvent.change(screen.getByLabelText('初始资金（USDT）'), { target: { value: '' } });
  expect(screen.getByLabelText('初始资金（USDT）')).toHaveValue(null);
  await waitFor(() => expect(screen.getByRole('button', { name: '运行回测' })).toBeEnabled());
  fireEvent.click(screen.getByRole('button', { name: '运行回测' }));
  expect(await screen.findByRole('alert')).toHaveTextContent('日期');
  expect(requests.filter((r) => r.url.endsWith('/runs'))).toHaveLength(0);
});
it('displays a backtest start failure and retains editable parameters', async () => {
  harness(<BacktestForm onRunStarted={vi.fn()} />, true);
  dates();
  await waitFor(() => expect(screen.getByRole('button', { name: '运行回测' })).toBeEnabled());
  fireEvent.click(screen.getByRole('button', { name: '运行回测' }));
  expect(await screen.findByRole('alert')).toHaveTextContent('失败');
  expect(screen.getByLabelText('起始日期')).toHaveValue('2026-08-01');
});
it('keeps one pending start and delivers its run ID despite attempted edits and resubmission', async () => {
  const started = vi.fn();
  harness(<BacktestForm onRunStarted={started} />);
  await screen.findByRole('combobox', { name: /历史|复用/ });
  dates();
  let release!: () => void;
  const pending = new Promise<void>((resolve) => {
    release = resolve;
  });
  const requests: RequestInit[] = [];
  const otherFetch = globalThis.fetch;
  vi.stubGlobal(
    'fetch',
    vi.fn((url: string, init: RequestInit) => {
      if (!url.endsWith('/api/backtest/runs') || init?.method !== 'POST') return otherFetch(url, init);
      requests.push(init);
      return pending.then(() => new Response(JSON.stringify({ run_id: 'delayed-run', status: 'queued' })));
    }),
  );
  const user = userEvent.setup();
  const submit = screen.getByRole('button', { name: '运行回测' });
  await waitFor(() => expect(submit).toBeEnabled());
  await user.click(submit);
  await waitFor(() => expect(requests).toHaveLength(1));
  try {
    await user.type(screen.getByLabelText('初始资金（USDT）'), '9');
    await user.click(submit);
    expect(requests).toHaveLength(1);
    await user.type(screen.getByLabelText('起始日期'), '2026-08-02');
    await user.type(screen.getByLabelText('结束日期'), '2026-08-09');
    for (const name of ['初始资金（USDT）', '起始日期', '结束日期', '币对']) {
      expect(screen.getByLabelText(name)).toBeDisabled();
    }
    expect(submit).toBeDisabled();
    fireEvent.submit(submit.closest('form')!);
    expect(requests).toHaveLength(1);
  } finally {
    release();
  }
  await waitFor(() => expect(started).toHaveBeenCalledExactlyOnceWith('delayed-run'));
  expect(requests).toHaveLength(1);
});
it('loads saved parameters through GET and never sends the old output session name', async () => {
  const started = vi.fn();
  const requests = harness(<BacktestForm onRunStarted={started} />);
  const selection = await screen.findByRole('combobox', { name: /历史|复用/ });
  await screen.findByRole('option', { name: /prior/ });
  fireEvent.change(selection, { target: { value: 'prior' } });
  await waitFor(() => expect(screen.getByLabelText('初始资金（USDT）')).toHaveValue(4200));
  await waitFor(() => expect(screen.getByRole('button', { name: '运行回测' })).toBeEnabled());
  fireEvent.click(screen.getByRole('button', { name: '运行回测' }));
  await waitFor(() => expect(started).toHaveBeenCalledWith('run-new'));
  expect(requests.find((r) => r.url.endsWith('/runs'))!.body).toEqual({
    start: '2026-08-01',
    end: '2026-08-10',
    pair: 'ETH/USDT',
    initial_equity: '4200',
    interval: '1h',
    fee_rate: '0.001',
    slippage_bps: '0',
    funding_assumption: 'available_only',
    name: null,
    snapshot_run_id: 'prior',
  });
});
it('shows a saved-parameter load failure without erasing the current dates', async () => {
  harness(<BacktestForm onRunStarted={vi.fn()} />, false, true);
  dates();
  await screen.findByRole('option', { name: /prior/ });
  fireEvent.change(await screen.findByRole('combobox', { name: /历史|复用/ }), { target: { value: 'prior' } });
  expect(await screen.findByRole('alert')).toHaveTextContent('失败');
  expect(screen.getByLabelText('起始日期')).toHaveValue('2026-08-01');
});
it.each([
  ['price_threshold', '目标价格'],
  ['pct_change', '时间窗口（分钟）'],
  ['candle_pattern', '连续根数'],
  ['funding_rate', '费率阈值 (%)'],
])('requires the active %s inputs and associates local errors', async (trigger, label) => {
  const requests = harness(<RuleFormDialog open onOpenChange={vi.fn()} rule={undefined} prefill={undefined} />);
  fireEvent.change(screen.getByLabelText('规则名称'), { target: { value: 'test' } });
  fireEvent.change(screen.getByLabelText('触发类型'), { target: { value: trigger } });
  fireEvent.click(screen.getByRole('button', { name: '保存' }));
  await waitFor(() => expect(screen.getByLabelText(label)).toHaveAttribute('aria-invalid', 'true'));
  expect(screen.getByLabelText(label)).toHaveAccessibleDescription(/填写|有效/);
  expect(requests.filter((r) => r.method === 'POST')).toHaveLength(0);
});
it('preserves funding percentage points and shows rule mutation failure', async () => {
  const requests = harness(<RuleFormDialog open onOpenChange={vi.fn()} rule={undefined} prefill={undefined} />, true);
  fireEvent.change(screen.getByLabelText('规则名称'), { target: { value: 'funding' } });
  fireEvent.change(screen.getByLabelText('触发类型'), { target: { value: 'funding_rate' } });
  fireEvent.change(screen.getByLabelText('费率阈值 (%)'), { target: { value: '0.1' } });
  fireEvent.click(screen.getByRole('button', { name: '保存' }));
  expect(await screen.findByRole('alert')).toHaveTextContent('失败');
  expect(requests.find((r) => r.method === 'POST')!.body.parameters).toEqual({ threshold_pct: 0.1 });
});
it.each(['toggle', 'delete'])('displays a rule %s failure without losing the row', async (operation) => {
  harness(<RuleTable rules={[rule]} onEdit={vi.fn()} />, true);
  if (operation === 'toggle') fireEvent.click(screen.getByRole('switch'));
  else {
    fireEvent.click(screen.getByRole('button', { name: '删除' }));
    fireEvent.click(screen.getByRole('button', { name: '确认删除' }));
  }
  expect(await screen.findByRole('alert')).toHaveTextContent('失败');
  expect(screen.getByText('BTC rule')).toBeInTheDocument();
});

it.each([
  ['funding_rate', '费率阈值 (%)', '0'],
  ['pct_change', '涨跌幅阈值 (%)', '0'],
  ['price_threshold', '冷却时间（分钟）', '1441'],
] as const)('rejects non-actionable or out-of-range %s values', async (trigger, label, value) => {
  const requests = harness(
    <RuleFormDialog
      open
      onOpenChange={vi.fn()}
      rule={undefined}
      prefill={{
        name: 'valid',
        trigger_type: trigger,
        parameters: { price: 50000, window_minutes: 15, threshold_pct: 3 },
      }}
    />,
  );
  fireEvent.change(screen.getByLabelText(label), { target: { value } });
  fireEvent.click(screen.getByRole('button', { name: '保存' }));
  await waitFor(() => expect(screen.getByLabelText(label)).toHaveAttribute('aria-invalid', 'true'));
  expect(requests.filter((r) => r.method === 'POST')).toHaveLength(0);
});

it.each(['POST', 'PUT'])(
  'associates a %s server field error with the scheduler input and retains the draft',
  async (method) => {
    harness(
      <RuleFormDialog
        open
        onOpenChange={vi.fn()}
        rule={method === 'PUT' ? rule : undefined}
        prefill={
          method === 'POST'
            ? { name: 'BTC rule', trigger_type: 'price_threshold', parameters: rule.parameters }
            : undefined
        }
      />,
    );
    const requests: string[] = [];
    vi.stubGlobal(
      'fetch',
      vi.fn((_url: string, init: RequestInit) => {
        requests.push(init.method!);
        return Promise.resolve(
          new Response(
            JSON.stringify({
              detail: [{ loc: ['body', 'pair'], msg: 'Invalid pair', type: 'value_error' }],
            }),
            { status: 422 },
          ),
        );
      }),
    );
    fireEvent.click(screen.getByRole('button', { name: '保存' }));
    await waitFor(() => expect(screen.getByLabelText('交易对')).toHaveAttribute('aria-invalid', 'true'));
    expect(screen.getByLabelText('交易对')).toHaveFocus();
    expect(screen.getByLabelText('规则名称')).toHaveValue('BTC rule');
    expect(requests).toEqual([method]);
  },
);

it('updates an existing rule with its active parameters and closes only on success', async () => {
  const close = vi.fn();
  const requests = harness(<RuleFormDialog open onOpenChange={close} rule={rule} prefill={undefined} />);
  fireEvent.change(screen.getByLabelText('目标价格'), { target: { value: '51000' } });
  fireEvent.click(screen.getByRole('button', { name: '保存' }));
  await waitFor(() => expect(close).toHaveBeenCalledWith(false));
  expect(requests.find((r) => r.method === 'PUT')?.body.parameters).toEqual({ direction: 'below', price: 51000 });
});

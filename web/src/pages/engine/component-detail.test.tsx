import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { expect, it } from 'vitest';
import { workflowHarness } from '@/test/configuration-workflow';

function harness(items: unknown[] = [], requiredAccess?: string) {
  return workflowHarness('/engine/components/kronos', undefined, requiredAccess, undefined, (url) =>
    url.includes('/api/decisions')
      ? Promise.resolve(
          new Response(JSON.stringify({ items, total: items.length, offset: 0, limit: 20, has_next: false })),
        )
      : undefined,
  );
}

it('keeps saved component history visible when configuration access is unavailable', async () => {
  harness([], 'missing-config-access');
  expect(await screen.findByRole('heading', { name: 'kronos', level: 1 })).toBeInTheDocument();
  expect(await screen.findByText('暂无运行历史')).toBeInTheDocument();
  await userEvent.click(screen.getByRole('button', { name: '配置' }));
  expect(await screen.findByRole('heading', { name: '解锁配置中心' })).toBeInTheDocument();
});

it('makes empty history actionable without triggering analysis or trading', async () => {
  const result = harness();
  expect(await screen.findByRole('heading', { name: 'Kronos', level: 1 })).toBeInTheDocument();
  expect(await screen.findByText('暂无运行历史')).toBeInTheDocument();
  expect(screen.getByRole('link', { name: '前往引擎配置' })).toHaveAttribute('href', '/engine#configuration');
  expect(
    result.fetchMock.mock.calls.filter(([url]) => url.includes('/chat') || url.includes('/analyses')),
  ).toHaveLength(0);
});

it('renders saved evidence, unknown cost and zero target without reinference', async () => {
  const result = harness([
    {
      decision_id: 'saved-1',
      pair: 'BTC/USDT:USDT',
      mode: 'analysis',
      origin: 'manual',
      config_snapshot: [],
      finished_at: '2026-01-01T00:01:00Z',
      failure: null,
      incomplete_fields: [],
      config_revision: 9,
      created_at: '2026-01-01T00:00:00Z',
      books: [],
      status: 'completed',
      components: [
        {
          component_id: 'kronos',
          direction: 'neutral',
          confidence: 0,
          reasoning: '门控拒绝',
          details: [],
          blocks: [{ kind: 'text', title: '运行说明', body: '本次没有预测' }],
          evaluation_reference: null,
          status: 'skipped',
          duration_ms: 12,
          usage: null,
          cost: null,
        },
      ],
      fusion: null,
      target: { side: 'flat', size_ratio: 0 },
    },
  ]);
  expect(await screen.findByText('本次没有预测')).toBeInTheDocument();
  expect(screen.getByText('费用未知')).toBeInTheDocument();
  expect(screen.getByText('目标持仓为零')).toBeInTheDocument();
  expect(screen.getByText('参考行情缺失，无法评估本次结果。')).toBeInTheDocument();
  await userEvent.click(screen.getByRole('button', { name: '历史' }));
  expect(screen.getByText('本次没有预测')).toBeInTheDocument();
  expect(result.fetchMock.mock.calls.filter(([, init]) => init?.method && init.method !== 'GET')).toHaveLength(0);
});

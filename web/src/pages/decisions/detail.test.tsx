import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { expect, it } from 'vitest';
import { workflowHarness } from '@/test/configuration-workflow';

const saved = {
  decision_id: 'saved-analysis',
  pair: 'BTC/USDT:USDT',
  mode: 'analysis',
  origin: 'manual',
  config_revision: 9,
  config_snapshot: [],
  created_at: '2026-08-31T10:00:00Z',
  finished_at: '2026-08-31T10:01:00Z',
  status: 'completed',
  books: [],
  failure: null,
  incomplete_fields: [],
  components: [
    {
      component_id: 'kronos',
      direction: 'long',
      confidence: 0.8,
      reasoning: '已保存判断',
      details: [],
      blocks: [{ kind: 'text', title: '结果', body: '本次预测已保存' }],
      evaluation_reference: null,
      status: 'completed',
      duration_ms: 12,
      usage: null,
      cost: null,
    },
  ],
  fusion: {
    score: 0.8,
    reasoning: '融合已完成',
    contributions: [{ component_id: 'kronos', weight: 1, signed_score: 0.8, weighted_score: 0.8 }],
  },
  target: { side: 'long', size_ratio: 0.8 },
};

it('opens the saved decision at its canonical URL and analysis ends at target without execution', async () => {
  const result = workflowHarness('/decisions/saved-analysis');
  const base = result.fetchMock.getMockImplementation()!;
  result.fetchMock.mockImplementation((url, init) =>
    url.includes('/api/decisions/') ? Promise.resolve(new Response(JSON.stringify(saved))) : base(url, init),
  );
  expect(await screen.findByText('本次预测已保存')).toBeInTheDocument();
  expect(screen.getByText('仅分析已完成，未访问交易账户，也未创建审批或订单。')).toBeInTheDocument();
  expect(screen.getByRole('heading', { name: '目标持仓' })).toBeInTheDocument();
  expect(screen.queryByRole('heading', { name: '资金池执行' })).not.toBeInTheDocument();
  expect(result.fetchMock.mock.calls.filter(([, init]) => init?.method && init.method !== 'GET')).toHaveLength(0);
});

it('starts analysis only from an explicit engine action using the saved revision', async () => {
  const result = workflowHarness('/engine');
  const base = result.fetchMock.getMockImplementation()!;
  result.fetchMock.mockImplementation((url, init) => {
    if (url.endsWith('/api/analyses'))
      return Promise.resolve(new Response(JSON.stringify({ decision_id: saved.decision_id }), { status: 202 }));
    if (url.includes('/api/decisions/')) return Promise.resolve(new Response(JSON.stringify(saved)));
    return base(url, init);
  });
  await userEvent.click(await screen.findByRole('button', { name: '仅分析，不交易' }));
  expect(await screen.findByText('本次预测已保存')).toBeInTheDocument();
  const writes = result.fetchMock.mock.calls.filter(([, init]) => init?.method === 'POST');
  expect(writes).toHaveLength(1);
  expect(JSON.parse(writes[0]![1]!.body as string)).toEqual({
    pair: 'BTC/USDT:USDT',
    expected_revision: result.saved().revision,
  });
});

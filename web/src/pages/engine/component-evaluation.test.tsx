import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { expect, it } from 'vitest';
import { workflowHarness } from '@/test/configuration-workflow';

const group = {
  component_id: 'kronos',
  pair: 'BTC/USDT',
  mode: 'analysis',
  config_revision: 7,
  interval: '2h',
  total: 5,
  pending: 1,
  matured_directional: 2,
  hits: 1,
  neutral: 1,
  skipped: 0,
  failed: 0,
  missing_market: 1,
  hit_rate: 0.5,
};

function harness(groups: unknown[] = [], items: unknown[] = []) {
  return workflowHarness('/engine/components/kronos', undefined, undefined, undefined, (url) => {
    if (url.includes('/evaluations'))
      return Promise.resolve(
        new Response(
          JSON.stringify({ items, summary: { groups }, total: items.length, limit: 20, offset: 0, has_next: false }),
        ),
      );
    if (url.includes('/api/decisions'))
      return Promise.resolve(
        new Response(JSON.stringify({ items: [], total: 0, offset: 0, limit: 20, has_next: false })),
      );
    return undefined;
  });
}

it('offers an effect tab with an honest empty denominator without starting a run', async () => {
  const result = harness();
  await userEvent.click(await screen.findByRole('button', { name: '效果' }));
  expect(await screen.findByText('尚无可评估样本')).toBeInTheDocument();
  expect(screen.getByText(/分母仅含已到期、有方向且取得截止行情的样本/)).toBeInTheDocument();
  expect(result.fetchMock.mock.calls.filter(([, init]) => init?.method && init.method !== 'GET')).toHaveLength(0);
});

it('keeps live and backtest rates separate and filters by original revision and period', async () => {
  const result = harness([group, { ...group, mode: 'backtest', hit_rate: null, matured_directional: 0, hits: 0 }]);
  await userEvent.click(await screen.findByRole('button', { name: '效果' }));
  expect(await screen.findByText('50.0%')).toBeInTheDocument();
  expect(screen.getByText('命中 1 / 有效方向样本 2')).toBeInTheDocument();
  expect(screen.getByText('尚无可评估样本')).toBeInTheDocument();
  expect(screen.getByText('实时分析 · BTC/USDT · 2h · 配置 7')).toBeInTheDocument();
  expect(screen.getByText('回测 · BTC/USDT · 2h · 配置 7')).toBeInTheDocument();
  await userEvent.selectOptions(screen.getByLabelText('样本来源'), 'backtest');
  expect(result.fetchMock.mock.calls.some(([url]) => url.includes('mode=backtest'))).toBe(true);
});

it('reads a saved forecast against later bars and links the unchanged decision', async () => {
  harness(
    [group],
    [
      {
        decision_id: 'frozen',
        component_id: 'kronos',
        pair: 'BTC/USDT',
        mode: 'analysis',
        config_revision: 7,
        interval: '2h',
        created_at: '2026-08-31T12:00:00Z',
        status: 'evaluated',
        direction: 'long',
        reference: {
          reference_time: '2026-08-31T12:00:00Z',
          reference_price: '100',
          due_at: '2026-08-31T14:00:00Z',
          interval: '2h',
          market_source_id: 'frozen-source',
        },
        actual_price: '110',
        actual_time: '2026-08-31T14:00:00Z',
        hit: true,
        return_ratio: '0.1',
        reason: null,
        cost: null,
        comparisons: [
          {
            title: '价格预测',
            name: '预测收盘价',
            timeframe: '4h',
            matched: 1,
            total: 2,
            mae: null,
            rmse: null,
            points: [
              {
                time: '2026-08-31T12:00:00Z',
                close_time: '2026-08-31T16:00:00Z',
                predicted: '108',
                actual: '110',
                difference: '-2',
                status: 'matched',
              },
              {
                time: '2026-08-31T16:00:00Z',
                close_time: '2026-08-31T20:00:00Z',
                predicted: '118',
                actual: null,
                difference: null,
                status: 'pending',
              },
            ],
          },
        ],
      },
    ],
  );
  await userEvent.click(await screen.findByRole('button', { name: '效果' }));
  expect(await screen.findByText('费用未知')).toBeInTheDocument();
  expect(screen.getByText(/市场价格变化 10.00%（不是交易收益）/)).toBeInTheDocument();
  expect(screen.getByRole('link', { name: /查看原决策/ })).toHaveAttribute('href', '/decisions/frozen');
  await userEvent.click(screen.getByText(/匹配 1\/2 · 完整曲线误差待齐全/));
  expect(screen.getByRole('columnheader', { name: '实际收盘时刻' })).toBeVisible();
  expect(screen.getByRole('cell', { name: '未收盘' })).toBeVisible();
  expect(screen.getByRole('cell', { name: '-2' })).toBeVisible();
});

it('explains frozen curve parameter failures without suggesting a market retry', async () => {
  harness(
    [],
    [
      {
        decision_id: 'invalid-curve',
        component_id: 'kronos',
        pair: 'BTC/USDT',
        mode: 'analysis',
        config_revision: 7,
        interval: '2h',
        created_at: '2026-08-31T12:00:00Z',
        status: 'failed',
        direction: 'long',
        reference: {
          reference_time: '2026-08-31T12:00:00Z',
          reference_price: '100',
          due_at: '2026-08-31T14:00:00Z',
          interval: '2h',
          market_source_id: 'frozen-source',
        },
        actual_price: null,
        actual_time: null,
        hit: null,
        return_ratio: null,
        reason: 'curve_initialization_failed',
        cost: null,
        comparisons: [],
      },
    ],
  );
  await userEvent.click(await screen.findByRole('button', { name: '效果' }));
  expect(await screen.findByText('原预测曲线参数无法解析，本样本未参与评估；可查看原决策。')).toBeInTheDocument();
  expect(screen.getByRole('link', { name: /查看原决策/ })).toHaveAttribute('href', '/decisions/invalid-curve');
  expect(screen.queryByText(/历史行情读取失败，后台将重试/)).not.toBeInTheDocument();
});

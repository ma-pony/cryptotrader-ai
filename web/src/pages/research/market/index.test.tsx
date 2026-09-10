import { screen, within } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import { workflowHarness } from '@/test/configuration-workflow';

vi.mock('./components/chart-tab-panel', () => ({ ChartTabPanel: () => <div>行情图表</div> }));
vi.mock('./components/market-sidebar', () => ({ MarketSidebar: () => null }));

it('keeps shared research navigation and labels the chart controls', async () => {
  workflowHarness('/research/market');
  await screen.findByRole('heading', { name: '行情看板' });
  const navigation = screen.getByRole('navigation', { name: '研究导航' });
  expect(within(navigation).getByRole('link', { name: '市场观察' })).toHaveAttribute('aria-current', 'page');
  expect(within(navigation).getByRole('link', { name: '仅分析' })).toHaveAttribute('href', '/research/analysis');
  expect(screen.getByRole('combobox', { name: '图表 K 线周期' })).toHaveValue('1h');
  expect(screen.getByRole('button', { name: '币安' })).toHaveAttribute('aria-pressed', 'true');
});

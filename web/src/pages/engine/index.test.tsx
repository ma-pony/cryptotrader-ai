import { fireEvent, screen, within } from '@testing-library/react';
import { expect, it } from 'vitest';
import { workflowHarness } from '@/test/configuration-workflow';

it('exposes registered component details and global timeframe through the engine route', async () => {
  workflowHarness('/engine');
  expect(await screen.findByRole('heading', { name: '引擎' })).toBeInTheDocument();
  expect(await screen.findByRole('link', { name: '查看 Kronos 结果' })).toHaveAttribute(
    'href',
    '/engine/components/kronos',
  );
  expect(await screen.findByLabelText('全局参考周期')).toHaveValue('1h');
  expect(screen.getByLabelText('信号评估周期')).toHaveValue('');
});

it('switches configuration sections without losing a draft or writing configuration', async () => {
  const h = workflowHarness('/engine#market');
  const timeframe = await screen.findByLabelText('全局参考周期');
  fireEvent.change(timeframe, { target: { value: '4h' } });
  const navigation = within(screen.getByRole('tablist', { name: '引擎配置分区' }));
  fireEvent.mouseDown(navigation.getByRole('tab', { name: /信号组件/ }), { button: 0, ctrlKey: false });
  expect(timeframe).not.toBeVisible();
  expect(screen.getByLabelText('信号评估周期')).toBeVisible();
  expect(navigation.getByRole('tab', { name: /行情与上下文/ })).toHaveAccessibleDescription('有未保存修改');
  fireEvent.mouseDown(navigation.getByRole('tab', { name: /行情与上下文/ }), { button: 0, ctrlKey: false });
  expect(timeframe).toBeVisible();
  expect(timeframe).toHaveValue('4h');
  expect(h.writes).toHaveLength(0);
});

import { screen } from '@testing-library/react';
import { expect, it } from 'vitest';
import { workflowHarness } from '@/test/configuration-workflow';

it('exposes registered component details and global timeframe through the engine route', async () => {
  workflowHarness('/engine');
  expect(await screen.findByRole('heading', { name: '引擎' })).toBeInTheDocument();
  expect(await screen.findByRole('link', { name: '查看 Kronos 结果' })).toHaveAttribute('href', '/engine/components/kronos');
  expect(await screen.findByLabelText('全局参考周期')).toHaveValue('1h');
  expect(screen.getByLabelText('信号评估周期')).toHaveValue('');
});

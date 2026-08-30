import { act, fireEvent, screen, waitFor } from '@testing-library/react';
import { beforeEach, expect, it } from 'vitest';
import i18n from '@/lib/i18n';
import { workflowHarness } from '@/test/configuration-workflow';
import { RUNTIME_CONFIG_QUERY_KEY } from '@/hooks/use-runtime-config';
beforeEach(() => i18n.changeLanguage('zh-CN'));
it('retains cross-section drafts through venue refresh and saves only the current section', async () => {
  const h = workflowHarness('/settings/models');
  fireEvent.change(await screen.findByLabelText('综合分析模型'), { target: { value: 'new-analysis' } });
  fireEvent.click(screen.getByRole('link', { name: '风控与审批' }));
  fireEvent.change(await screen.findByLabelText('最大回撤（%）'), { target: { value: '7' } });
  fireEvent.click(screen.getByRole('link', { name: '平台连接' }));
  const refreshed = h.saved();
  h.setSaved({
    ...refreshed,
    revision: 2,
    document: { ...refreshed.document, infrastructure: { redis_url: 'redis://refreshed:6379/0' } },
  });
  fireEvent.click(await screen.findByRole('button', { name: '重新加载' }));
  await waitFor(() => expect(h.client.getQueryData(RUNTIME_CONFIG_QUERY_KEY)).toMatchObject({ revision: 2 }));
  fireEvent.click(screen.getByRole('link', { name: '模型与网关' }));
  expect(await screen.findByLabelText('综合分析模型')).toHaveValue('new-analysis');
  fireEvent.click(screen.getByRole('button', { name: '保存配置' }));
  await waitFor(() => expect(h.writes).toHaveLength(1));
  expect(h.writes[0]).toMatchObject({
    expected_revision: 2,
    document: {
      llm: { models: { analysis: 'new-analysis' } },
      risk: { loss: { max_drawdown_pct: 0.15 } },
      infrastructure: { redis_url: 'redis://refreshed:6379/0' },
    },
  });
  fireEvent.click(screen.getByRole('link', { name: '风控与审批' }));
  expect(await screen.findByLabelText('最大回撤（%）')).toHaveValue(7);
  fireEvent.click(screen.getByRole('link', { name: '调度与触发器' }));
  fireEvent.change(await screen.findByLabelText('分析间隔（分钟）'), { target: { value: '45' } });
  fireEvent.click(screen.getByRole('button', { name: '保存配置' }));
  await waitFor(() => expect(h.writes).toHaveLength(2));
  expect(h.saved().document.llm.models.analysis).toBe('new-analysis');
  expect(h.saved().document.scheduler.interval_minutes).toBe(45);
  fireEvent.click(screen.getByRole('button', { name: '重新加载' }));
  expect(await screen.findByLabelText('分析间隔（分钟）')).toHaveValue(45);
});
it('retains a blank number across navigation and warns on actual page unload', async () => {
  workflowHarness('/settings/risk');
  fireEvent.change(await screen.findByLabelText('最大回撤（%）'), { target: { value: '' } });
  fireEvent.click(screen.getByRole('link', { name: '系统与通知' }));
  fireEvent.click(screen.getByRole('link', { name: '风控与审批' }));
  const number = await screen.findByLabelText('最大回撤（%）');
  expect(number).toHaveValue(null);
  fireEvent.click(screen.getByRole('button', { name: '保存配置' }));
  expect(number).toHaveAttribute('aria-invalid', 'true');
  const event = new Event('beforeunload', { cancelable: true });
  act(() => {
    window.dispatchEvent(event);
  });
  expect(event.defaultPrevented).toBe(true);
});

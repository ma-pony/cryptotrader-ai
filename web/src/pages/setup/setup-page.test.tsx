import { fireEvent, screen, waitFor, within } from '@testing-library/react';
import { beforeEach, expect, it } from 'vitest';
import i18n from '@/lib/i18n';
import { workflowConfig, workflowHarness } from '@/test/configuration-workflow';

beforeEach(() => i18n.changeLanguage('zh-CN'));
it('lets an inactive user reach all eight sections through the shared checklist', async () => {
  workflowHarness('/');
  expect(await screen.findByRole('heading', { name: '初始化交易系统' })).toBeInTheDocument();
  const checklist = screen.getByRole('navigation', { name: '初始化检查清单' });
  expect(within(checklist).getAllByRole('link')).toHaveLength(8);
  for (const label of [
    '模型与网关',
    '信号与权重',
    '行情数据',
    '平台连接',
    '执行资金池',
    '风控与审批',
    '调度与触发器',
    '系统与通知',
  ]) {
    fireEvent.click(screen.getByRole('link', { name: label }));
    expect(await screen.findByRole('heading', { name: label })).toBeInTheDocument();
  }
});
it('saves drawdown separately from activation and reloads the same daily fields', async () => {
  const h = workflowHarness();
  fireEvent.click(await screen.findByRole('link', { name: '风控与审批' }));
  fireEvent.change(await screen.findByLabelText('最大回撤（%）'), { target: { value: '3' } });
  fireEvent.click(screen.getByRole('button', { name: '保存配置' }));
  await waitFor(() => expect(h.writes).toHaveLength(1));
  expect(h.writes[0]).toMatchObject({
    document: { system: { active: false }, risk: { loss: { max_drawdown_pct: 0.03 } } },
  });
  fireEvent.click(screen.getByRole('button', { name: '重新加载' }));
  expect(await screen.findByLabelText('最大回撤（%）')).toHaveValue(3);
  fireEvent.click(screen.getByRole('link', { name: '初始化检查清单' }));
  expect(await screen.findByRole('button', { name: '激活交易系统' })).toBeDisabled();
});
it('activates only with saved valid sections and a current read-only check', async () => {
  const h = workflowHarness();
  expect(await screen.findByRole('button', { name: '激活交易系统' })).toBeDisabled();
  fireEvent.click(screen.getByRole('link', { name: '平台连接' }));
  fireEvent.click(await screen.findByRole('button', { name: '只读检查' }));
  await screen.findByText(/账户读取已验证/);
  fireEvent.click(screen.getByRole('link', { name: '初始化检查清单' }));
  await waitFor(() => expect(screen.getByRole('button', { name: '激活交易系统' })).toBeEnabled());
  expect(h.writes).toHaveLength(0);
  fireEvent.click(screen.getByRole('button', { name: '激活交易系统' }));
  await waitFor(() => expect(h.writes).toHaveLength(1));
  expect(h.writes[0]).toMatchObject({
    document: { system: { active: true }, execution: { live_order_execution_enabled: false } },
  });
});
it('invalidates readiness when a repeated account check fails', async () => {
  const h = workflowHarness('/settings/venues');
  fireEvent.click(await screen.findByRole('button', { name: '只读检查' }));
  await screen.findByText(/账户读取已验证/);
  h.fail(401, { code: 'authentication_failed' });
  fireEvent.click(screen.getByRole('button', { name: '只读检查' }));
  await screen.findByRole('alert');
  fireEvent.click(screen.getByRole('link', { name: '初始化检查清单' }));
  expect(await screen.findByRole('button', { name: '激活交易系统' })).toBeDisabled();
  expect(h.writes).toHaveLength(0);
});

it('does not call a persisted active revision activated after application fails and recovers explicitly', async () => {
  const h = workflowHarness('/settings/venues');
  fireEvent.click(await screen.findByRole('button', { name: '只读检查' }));
  await screen.findByText(/账户读取已验证/);
  fireEvent.click(screen.getByRole('link', { name: '初始化检查清单' }));
  h.failApply(true);
  fireEvent.click(await screen.findByRole('button', { name: '激活交易系统' }));
  await screen.findByRole('alert');
  expect(h.saved()).toMatchObject({
    revision: 2,
    apply_status: 'failed',
    applied_revision: 1,
    document: { system: { active: true } },
  });
  expect(screen.queryByText('交易系统已激活。')).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: '重新加载' }));
  await screen.findByText('当前配置激活失败，请检查服务状态后重试。');
  fireEvent.click(screen.getByRole('link', { name: '模型与网关' }));
  await screen.findByRole('heading', { name: '模型与网关' });
  fireEvent.click(screen.getByRole('link', { name: '初始化检查清单' }));
  await screen.findByText('当前配置激活失败，请检查服务状态后重试。');
  expect(screen.queryByText('交易系统已激活。')).not.toBeInTheDocument();
  expect(h.writes).toHaveLength(1);
  h.failApply(false);
  fireEvent.click(screen.getByRole('button', { name: '重试激活当前配置' }));
  await screen.findByText('交易系统已激活。');
  expect(h.writes[1]).toMatchObject({
    expected_revision: 2,
    document: { system: { active: true }, execution: { live_order_execution_enabled: false } },
  });
  expect(h.saved()).toMatchObject({ revision: 3, apply_status: 'applied', applied_revision: 3 });
});

it.each(['failed', 'pending', 'applied'] as const)(
  'requires the current revision to be confirmed on a fresh %s visit',
  async (apply_status) => {
    const config = workflowConfig();
    config.document.system.active = true;
    config.setup_required = false;
    config.apply_status = apply_status;
    config.applied_revision = 0;
    const h = workflowHarness('/setup', config);
    expect(await screen.findByRole('button', { name: '重试激活当前配置' })).toBeDisabled();
    expect(screen.queryByText('交易系统已激活。')).not.toBeInTheDocument();
    expect(h.writes).toHaveLength(0);
  },
);

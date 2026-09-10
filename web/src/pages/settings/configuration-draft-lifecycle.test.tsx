import { act, fireEvent, screen, waitFor, within } from '@testing-library/react';
import { beforeEach, expect, it } from 'vitest';
import i18n from '@/lib/i18n';
import { workflowApprovedChecks, workflowConfig, workflowHarness } from '@/test/configuration-workflow';
import { RUNTIME_CONFIG_QUERY_KEY } from '@/hooks/use-runtime-config';
import { useSettingsStore } from '@/stores/use-settings-store';
beforeEach(async () => {
  await i18n.changeLanguage('zh-CN');
  useSettingsStore.getState().reset();
});

async function openModels() {
  fireEvent.click(screen.getByRole('link', { name: '系统' }));
  await screen.findByRole('heading', { name: '模型与网关' });
}

async function openConnections() {
  fireEvent.click(screen.getByRole('link', { name: '账户' }));
  fireEvent.click(await screen.findByRole('link', { name: '管理连接' }));
  await screen.findByRole('heading', { name: '平台连接' });
}

it('preserves connection text edits while adopting an externally confirmed stop', async () => {
  const initial = workflowConfig();
  const h = workflowHarness('/accounts/connections', initial);
  fireEvent.change(await screen.findByLabelText('名称'), { target: { value: '停用后保留的名称' } });
  const stopped = structuredClone(initial);
  stopped.revision = 2;
  stopped.document.execution.connections[0]!.enabled = false;
  act(() => {
    h.client.setQueryData(RUNTIME_CONFIG_QUERY_KEY, stopped);
  });
  await waitFor(() => expect(screen.getByLabelText('启用连接')).not.toBeChecked());
  expect(screen.getByLabelText('名称')).toHaveValue('停用后保留的名称');
  fireEvent.click(screen.getByRole('button', { name: '保存并检查' }));
  await waitFor(() =>
    expect(
      h.fetchMock.mock.calls.some(
        ([url, init]) => url.endsWith('/api/venue-connections/paper') && init?.method === 'PUT',
      ),
    ).toBe(true),
  );
  const write = h.fetchMock.mock.calls.find(
    ([url, init]) => url.endsWith('/api/venue-connections/paper') && init?.method === 'PUT',
  )!;
  expect(JSON.parse(write[1]!.body as string)).toMatchObject({
    expected_revision: 2,
    enabled: false,
    label: '停用后保留的名称',
  });
});

it('keeps ordinary edits while writing model keys and uses API access rotation for later authenticated saves', async () => {
  const h = workflowHarness('/settings/models');
  fireEvent.change(await screen.findByLabelText('综合分析模型'), { target: { value: 'pending-model' } });
  fireEvent.change(screen.getByLabelText('模型网关密钥'), { target: { value: 'gateway-marker' } });
  fireEvent.click(screen.getByRole('button', { name: '保存网关密钥' }));
  await waitFor(() => expect(screen.queryByLabelText('模型网关密钥')).not.toBeInTheDocument());
  expect(screen.getByLabelText('综合分析模型')).toHaveValue('pending-model');
  fireEvent.click(screen.getByRole('link', { name: '安全与运行设置' }));
  fireEvent.change(await screen.findByLabelText('接口访问密钥'), { target: { value: 'access-marker' } });
  fireEvent.click(screen.getByRole('button', { name: '保存接口访问密钥' }));
  await waitFor(() => expect(screen.queryByLabelText('接口访问密钥')).not.toBeInTheDocument());
  fireEvent.click(screen.getByLabelText('启用接口访问安全'));
  fireEvent.click(screen.getByRole('button', { name: '保存配置' }));
  await waitFor(() => expect(h.writes).toHaveLength(1));
  const put = h.fetchMock.mock.calls.find(([url, init]) => url.endsWith('/api/config') && init?.method === 'PUT')!;
  expect(new Headers(put[1]?.headers).get('X-API-Key')).toBe('access-marker');
  expect(h.writes[0]).toMatchObject({
    expected_revision: 3,
    document: { llm: { models: { analysis: 'analysis' } }, security: { enabled: true } },
  });
  expect(JSON.stringify(h.client.getQueryData(RUNTIME_CONFIG_QUERY_KEY))).not.toContain('access-marker');
  expect(
    JSON.stringify(
      h.client
        .getMutationCache()
        .getAll()
        .map((mutation) => mutation.state.variables),
    ),
  ).not.toContain('gateway-marker');
  act(() => useSettingsStore.getState().reset());
});

it('keeps an unsaved connection edit while navigating through another configuration page', async () => {
  const h = workflowHarness('/accounts/connections');
  fireEvent.change(await screen.findByLabelText('名称'), { target: { value: '保留的连接草稿' } });
  await openModels();
  await openConnections();
  expect(await screen.findByLabelText('名称')).toHaveValue('保留的连接草稿');
  expect(h.writes).toHaveLength(0);
});

it('keeps a new connection ordinary draft while navigating away and back', async () => {
  const h = workflowHarness('/accounts/connections');
  fireEvent.click(await screen.findByRole('button', { name: '新增连接' }));
  const form = within(screen.getByRole('form', { name: '新增平台连接' }));
  fireEvent.change(form.getByRole('combobox', { name: /交易平台/ }), { target: { value: 'sample_venue' } });
  await waitFor(() => expect(form.getByLabelText('触发阈值')).toBeInTheDocument());
  fireEvent.change(form.getByLabelText('名称'), { target: { value: '新建草稿' } });
  fireEvent.change(form.getByLabelText('触发阈值'), { target: { value: '0.8' } });
  await openModels();
  await openConnections();
  const restored = within(await screen.findByRole('form', { name: '新增平台连接' }));
  expect(restored.getByLabelText('名称')).toHaveValue('新建草稿');
  expect(restored.getByLabelText('触发阈值')).toHaveValue(0.8);
  expect(h.writes).toHaveLength(0);
});

it('keeps a connection draft through another configuration save and reload recovery', async () => {
  const h = workflowHarness('/accounts/connections');
  fireEvent.change(await screen.findByLabelText('名称'), { target: { value: '跨页保存草稿' } });
  await openModels();
  fireEvent.change(await screen.findByLabelText('综合分析模型'), { target: { value: 'pending-model' } });
  fireEvent.click(screen.getByRole('button', { name: '保存配置' }));
  await waitFor(() => expect(h.writes).toHaveLength(1));
  await openConnections();
  expect(await screen.findByLabelText('名称')).toHaveValue('跨页保存草稿');
  h.failReload(true);
  fireEvent.click(screen.getByRole('button', { name: '重新加载' }));
  await screen.findByRole('alert');
  expect(screen.getByLabelText('名称')).toHaveValue('跨页保存草稿');
  h.failReload(false);
  const reloadsBefore = h.fetchMock.mock.calls.filter(
    ([url, init]) => url.endsWith('/api/config') && init?.method !== 'PUT',
  ).length;
  const reload = screen.getByRole('button', { name: '重新加载' });
  fireEvent.click(reload);
  await waitFor(() =>
    expect(
      h.fetchMock.mock.calls.filter(([url, init]) => url.endsWith('/api/config') && init?.method !== 'PUT'),
    ).toHaveLength(reloadsBefore + 1),
  );
  expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  await waitFor(() => expect(reload).toBeEnabled());
  expect(screen.getByLabelText('名称')).toHaveValue('跨页保存草稿');
});

it('clears connection draft state after restoring the persisted value', async () => {
  const config = workflowConfig();
  workflowHarness('/accounts/connections', config, undefined, workflowApprovedChecks(config));
  const name = await screen.findByLabelText('名称');
  fireEvent.change(name, { target: { value: '临时名称' } });
  fireEvent.change(name, { target: { value: 'Paper' } });
  await openModels();
  await openConnections();
  expect(await screen.findByLabelText('名称')).toHaveValue('Paper');
});

it('clears restored scheduler rules after a separate explicit pause without retaining a hidden switch draft', async () => {
  const config = workflowConfig();
  config.document.scheduler.automation_enabled = true;
  const h = workflowHarness('/engine#automation', config);
  fireEvent.change(await screen.findByLabelText('分析间隔（分钟）'), { target: { value: '60' } });
  fireEvent.click(await screen.findByRole('button', { name: '暂停自动运行' }));
  await screen.findAllByText('自动运行已暂停');
  expect(await screen.findByLabelText('分析间隔（分钟）')).toHaveValue(60);
  fireEvent.change(screen.getByLabelText('分析间隔（分钟）'), { target: { value: '15' } });
  expect(
    within(screen.getByLabelText('分析间隔（分钟）').closest('form')!).getByRole('button', { name: '保存配置' }),
  ).toBeDisabled();
  expect(h.saved().document.scheduler.automation_enabled).toBe(false);
  expect(h.writes).toHaveLength(0);
});

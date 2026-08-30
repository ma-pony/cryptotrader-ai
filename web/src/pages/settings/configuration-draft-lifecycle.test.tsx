import { act, fireEvent, screen, waitFor } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import i18n from '@/lib/i18n';
import { workflowConfig, workflowHarness } from '@/test/configuration-workflow';
import { RUNTIME_CONFIG_QUERY_KEY } from '@/hooks/use-runtime-config';
import { useSettingsStore } from '@/stores/use-settings-store';
beforeEach(async () => {
  await i18n.changeLanguage('zh-CN');
  useSettingsStore.getState().reset();
});

it('retains venue edits through failed reload, successful reload and section navigation', async () => {
  const h = workflowHarness('/settings/venues');
  fireEvent.change(await screen.findByLabelText('名称'), { target: { value: 'pending-venue' } });
  h.failReload(true);
  fireEvent.click(screen.getByRole('button', { name: '重新加载' }));
  await screen.findByText('重新加载失败，本地修改仍保留，请检查服务状态后重试。');
  expect(screen.getByLabelText('名称')).toHaveValue('pending-venue');
  h.failReload(false);
  fireEvent.click(screen.getByRole('button', { name: '重新加载' }));
  fireEvent.click(screen.getByRole('link', { name: '模型与网关' }));
  await screen.findByLabelText('综合分析模型');
  fireEvent.click(screen.getByRole('link', { name: '平台连接' }));
  expect(await screen.findByLabelText('名称')).toHaveValue('pending-venue');
  vi.spyOn(window, 'confirm').mockReturnValue(true);
  fireEvent.click(screen.getByRole('button', { name: '放弃修改' }));
  expect(screen.getByLabelText('名称')).toHaveValue('Paper');
});
it('clears failed venue credential inputs without caches and recovers a 409 only after reload', async () => {
  const config = workflowConfig();
  config.document.execution.connections = [
    { ...config.document.execution.connections[0]!, adapter_id: 'okx', environment: 'demo' },
  ];
  const h = workflowHarness('/settings/venues', config);
  await screen.findByLabelText('API Key');
  for (const status of [500, 409]) {
    h.fail(status, 'do-not-echo');
    fireEvent.change(screen.getByLabelText('API Key'), { target: { value: 'credential-marker' } });
    fireEvent.change(screen.getByLabelText('API Secret'), { target: { value: 'signing-marker' } });
    fireEvent.change(screen.getByLabelText('OKX Passphrase'), { target: { value: 'phrase-marker' } });
    fireEvent.click(screen.getByRole('button', { name: '保存凭据' }));
    await waitFor(() => expect(screen.getByLabelText('API Key')).toHaveValue(''));
    expect(screen.getByLabelText('API Secret')).toHaveValue('');
    expect(screen.getByLabelText('OKX Passphrase')).toHaveValue('');
    expect(document.body.textContent).not.toContain('do-not-echo');
    expect(JSON.stringify(h.client.getQueryData(RUNTIME_CONFIG_QUERY_KEY))).not.toContain('credential-marker');
    expect(
      JSON.stringify(
        h.client
          .getMutationCache()
          .getAll()
          .map((mutation) => mutation.state.variables),
      ),
    ).not.toContain('signing-marker');
  }
  expect(screen.getByRole('button', { name: '只读检查' })).toBeDisabled();
  h.clearFailure();
  fireEvent.click(screen.getByRole('button', { name: '重新加载' }));
  await waitFor(() => expect(screen.getByRole('button', { name: '只读检查' })).toBeEnabled());
});
it('recovers a successful credential write with failed refresh without keeping a stale local lock', async () => {
  const config = workflowConfig();
  config.document.execution.connections = [
    { ...config.document.execution.connections[0]!, adapter_id: 'okx', environment: 'demo' },
  ];
  const h = workflowHarness('/settings/venues', config);
  await screen.findByLabelText('API Key');
  h.failReload(true);
  for (const [name, value] of [
    ['API Key', 'test-key'],
    ['API Secret', 'test-signing'],
    ['OKX Passphrase', 'test-phrase'],
  ])
    fireEvent.change(screen.getByLabelText(name!), { target: { value } });
  fireEvent.click(screen.getByRole('button', { name: '保存凭据' }));
  await screen.findByText('连接已保存，但配置刷新失败；请重新加载后继续。');
  expect(screen.getByLabelText('API Key')).toHaveValue('');
  h.failReload(false);
  fireEvent.click(screen.getByRole('button', { name: '重新加载' }));
  await waitFor(() =>
    expect(screen.queryByText('连接已保存，但配置刷新失败；请重新加载后继续。')).not.toBeInTheDocument(),
  );
  expect(screen.getByRole('button', { name: '只读检查' })).toBeEnabled();
});
it('keeps ordinary edits while writing model keys and uses API access rotation for later authenticated saves', async () => {
  const h = workflowHarness('/settings/models');
  fireEvent.change(await screen.findByLabelText('综合分析模型'), { target: { value: 'pending-model' } });
  fireEvent.change(screen.getByLabelText('LLM 网关密钥'), { target: { value: 'gateway-marker' } });
  fireEvent.click(screen.getByRole('button', { name: '保存网关密钥' }));
  await waitFor(() => expect(screen.getByLabelText('LLM 网关密钥')).toHaveValue(''));
  expect(screen.getByLabelText('综合分析模型')).toHaveValue('pending-model');
  fireEvent.click(screen.getByRole('link', { name: '系统与通知' }));
  fireEvent.change(await screen.findByLabelText('API 访问密钥'), { target: { value: 'access-marker' } });
  fireEvent.click(screen.getByRole('button', { name: '保存 API 访问密钥' }));
  await waitFor(() => expect(screen.getByLabelText('API 访问密钥')).toHaveValue(''));
  fireEvent.click(screen.getByLabelText('启用 API 访问安全'));
  fireEvent.click(screen.getByRole('button', { name: '保存配置' }));
  await waitFor(() => expect(h.writes).toHaveLength(1));
  const put = h.fetchMock.mock.calls.find(([url, init]) => url.endsWith('/api/config') && init?.method === 'PUT')!;
  expect(new Headers(put[1]?.headers).get('X-API-Key')).toBe('access-marker');
  expect(h.writes[0]).toMatchObject({
    expected_revision: 3,
    document: { llm: { models: { analysis: 'analysis' } }, security: { enabled: true }, system: { active: false } },
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

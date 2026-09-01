import { fireEvent, screen, waitFor, within } from '@testing-library/react';
import { beforeEach, expect, it } from 'vitest';
import i18n from '@/lib/i18n';
import { useSettingsStore } from '@/stores/use-settings-store';
import { workflowHarness } from '@/test/configuration-workflow';

beforeEach(async () => {
  await i18n.changeLanguage('zh-CN');
  useSettingsStore.getState().reset();
});

const runtimeCredentials = [
  { path: '/settings/models', kind: 'llm-gateway', label: '模型网关密钥', save: '保存网关密钥' },
  { path: '/engine', kind: 'news-provider', label: 'CoinDesk 新闻密钥（可选）', save: '保存新闻密钥' },
  { path: '/settings/security', kind: 'api-access', label: '接口访问密钥', save: '保存接口访问密钥' },
];

it.each(runtimeCredentials)(
  'shows durable saved state and explicit replacement on $path',
  async ({ path, kind, label, save }) => {
    const h = workflowHarness(path);
    fireEvent.change(await screen.findByLabelText(label), { target: { value: 'saved-secret-marker' } });
    fireEvent.click(screen.getByRole('button', { name: save }));
    await screen.findByText('凭据已配置');
    await waitFor(() => expect(screen.queryByLabelText(label)).not.toBeInTheDocument());
    expect(screen.queryByText(/未验证连接/)).not.toBeInTheDocument();
    const panel = screen.getByRole('region', { name: `${label}配置` });
    expect(within(panel).getByText(/最近保存/)).not.toHaveTextContent('T13:00:00Z');
    expect(panel.querySelector('time')).toHaveAttribute('dateTime', '2026-08-30T13:00:00Z');
    fireEvent.click(screen.getByRole('button', { name: '更换凭据' }));
    expect(screen.getByLabelText(label)).toHaveValue('');
    fireEvent.change(screen.getByLabelText(label), { target: { value: 'cancelled-secret-marker' } });
    fireEvent.click(screen.getByRole('button', { name: '取消更换' }));
    expect(screen.queryByLabelText(label)).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: '更换凭据' }));
    expect(screen.getByLabelText(label)).toHaveValue('');
    expect(h.fetchMock.mock.calls.filter(([url]) => url.endsWith(`/credentials/${kind}`))).toHaveLength(1);
    expect(
      JSON.stringify(
        h.client
          .getQueryCache()
          .getAll()
          .map((q) => q.state),
      ),
    ).not.toContain('secret-marker');
    expect(h.client.getMutationCache().getAll()).toHaveLength(0);
    const saved = h.saved();
    h.unmount();
    const reloaded = workflowHarness(path, saved, undefined, h.checks);
    await screen.findByText('凭据已配置');
    expect(screen.queryByLabelText(label)).not.toBeInTheDocument();
    expect(reloaded.fetchMock.mock.calls.every(([, init]) => init?.method === 'GET')).toBe(true);
  },
);

it.each(runtimeCredentials)(
  'keeps a failed write editable without claiming it was saved on $path',
  async ({ path, label, save }) => {
    const h = workflowHarness(path);
    fireEvent.change(await screen.findByLabelText(label), { target: { value: 'rejected-secret-marker' } });
    h.fail(422, 'do-not-display-provider-error');
    fireEvent.click(screen.getByRole('button', { name: save }));
    await waitFor(() => expect(screen.getByLabelText(label)).toHaveValue(''));
    expect(screen.getByText(/凭据写入失败/)).toBeInTheDocument();
    expect(screen.getByText('尚未配置凭据')).toBeInTheDocument();
    expect(screen.queryByText('凭据已配置')).not.toBeInTheDocument();
    expect(screen.getByLabelText(label)).toBeEnabled();
    expect(document.body.textContent).not.toContain('do-not-display-provider-error');
    h.clearFailure();
    fireEvent.change(screen.getByLabelText(label), { target: { value: 'retry-secret-marker' } });
    fireEvent.click(screen.getByRole('button', { name: save }));
    await waitFor(() => expect(screen.queryByLabelText(label)).not.toBeInTheDocument());
    expect(screen.getByText('凭据已配置')).toBeInTheDocument();
  },
);

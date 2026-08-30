import { fireEvent, screen, waitFor } from '@testing-library/react';
import { beforeEach, expect, it } from 'vitest';
import i18n from '@/lib/i18n';
import { useSettingsStore } from '@/stores/use-settings-store';
import { workflowHarness } from '@/test/configuration-workflow';

it('reports a committed credential with failed public refresh, clears the input and preserves drafts', async () => {
  const h = workflowHarness('/settings/models');
  fireEvent.change(await screen.findByLabelText('综合分析模型'), { target: { value: 'pending-model' } });
  h.failReload(true);
  fireEvent.change(screen.getByLabelText('LLM 网关密钥'), { target: { value: 'refresh-failure-marker' } });
  fireEvent.click(screen.getByRole('button', { name: '保存网关密钥' }));
  expect(await screen.findByText('凭据已保存，但未能刷新配置状态。请重新加载后再操作。')).toBeInTheDocument();
  expect(screen.getByLabelText('LLM 网关密钥')).toHaveValue('');
  expect(screen.getByLabelText('综合分析模型')).toHaveValue('pending-model');
  expect(screen.getByRole('button', { name: '保存配置' })).toBeDisabled();
  h.failReload(false);
  fireEvent.click(screen.getByRole('button', { name: '重新加载' }));
  await waitFor(() => expect(screen.getByRole('button', { name: '保存配置' })).toBeEnabled());
  expect(screen.queryByText('凭据已保存，但未能刷新配置状态。请重新加载后再操作。')).not.toBeInTheDocument();
  expect(screen.getByLabelText('综合分析模型')).toHaveValue('pending-model');
});

beforeEach(async () => {
  await i18n.changeLanguage('zh-CN');
  useSettingsStore.getState().reset();
});

it('unlocks a fresh settings visit with an existing memory-only key using GET only', async () => {
  const h = workflowHarness('/settings/models', undefined, 'existing-access-marker');
  const field = await screen.findByLabelText('已有 API 访问密钥');
  expect(field).toHaveAttribute('type', 'password');
  fireEvent.change(field, { target: { value: 'existing-access-marker' } });
  fireEvent.click(screen.getByRole('button', { name: '解锁并重新加载' }));
  await screen.findByLabelText('综合分析模型');
  const reads = h.fetchMock.mock.calls.filter(([url]) => url.endsWith('/api/config'));
  expect(new Headers(reads[0]![1]?.headers).get('X-API-Key')).toBeNull();
  expect(new Headers(reads.at(-1)![1]?.headers).get('X-API-Key')).toBe('existing-access-marker');
  expect(h.fetchMock.mock.calls.every(([, init]) => init?.method === 'GET')).toBe(true);
  expect(
    JSON.stringify(
      h.client
        .getQueryCache()
        .getAll()
        .map((query) => query.state),
    ),
  ).not.toContain('existing-access-marker');
  expect(h.client.getMutationCache().getAll()).toHaveLength(0);
  expect(h.writes).toHaveLength(0);
});

it('clears a wrong key and allows a safe retry without rotating stored credentials', async () => {
  const h = workflowHarness('/settings/models', undefined, 'existing-access-marker');
  fireEvent.change(await screen.findByLabelText('已有 API 访问密钥'), { target: { value: 'wrong-access-marker' } });
  fireEvent.click(screen.getByRole('button', { name: '解锁并重新加载' }));
  await waitFor(() => expect(screen.getByLabelText('已有 API 访问密钥')).toHaveValue(''));
  expect(await screen.findByRole('alert')).toHaveTextContent('验证未通过，请检查访问密钥后重试。');
  expect(useSettingsStore.getState().apiKey).toBe('');
  expect(document.body.textContent).not.toContain('wrong-access-marker');
  fireEvent.change(screen.getByLabelText('已有 API 访问密钥'), { target: { value: 'existing-access-marker' } });
  fireEvent.click(screen.getByRole('button', { name: '解锁并重新加载' }));
  await screen.findByLabelText('综合分析模型');
  expect(h.writes).toHaveLength(0);
  expect(
    JSON.stringify(
      h.client
        .getQueryCache()
        .getAll()
        .map((query) => query.state),
    ),
  ).not.toContain('wrong-access-marker');
});

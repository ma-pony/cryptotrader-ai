import { fireEvent, screen, waitFor } from '@testing-library/react';
import { beforeEach, expect, it } from 'vitest';
import i18n from '@/lib/i18n';
import { useSettingsStore } from '@/stores/use-settings-store';
import { workflowHarness } from '@/test/configuration-workflow';

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

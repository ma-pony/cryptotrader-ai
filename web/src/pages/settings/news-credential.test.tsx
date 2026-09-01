import { fireEvent, screen, waitFor } from '@testing-library/react';
import { beforeEach, expect, it } from 'vitest';
import i18n from '@/lib/i18n';
import { workflowHarness } from '@/test/configuration-workflow';
import { RUNTIME_CONFIG_QUERY_KEY } from '@/hooks/use-runtime-config';

beforeEach(() => i18n.changeLanguage('zh-CN'));
it('writes optional news credentials without exposing them or erasing ordinary drafts', async () => {
  const h = workflowHarness('/engine');
  const input = await screen.findByLabelText('CoinDesk 新闻密钥（可选）');
  expect(input).toHaveAttribute('type', 'password');
  fireEvent.change(input, { target: { value: 'fixture-news-secret' } });
  fireEvent.click(screen.getByRole('button', { name: '保存新闻密钥' }));
  await waitFor(() => expect(screen.queryByLabelText('CoinDesk 新闻密钥（可选）')).not.toBeInTheDocument());
  expect(
    h.fetchMock.mock.calls.some(
      ([url, init]) =>
        url.endsWith('/api/config/credentials/news-provider') &&
        JSON.parse(init!.body as string).token === 'fixture-news-secret',
    ),
  ).toBe(true);
  expect(h.client.getQueryData(RUNTIME_CONFIG_QUERY_KEY)).toMatchObject({
    revision: 2,
    document: { market_data: { news_credential_configured: true } },
  });
  expect(
    JSON.stringify(
      h.client
        .getQueryCache()
        .getAll()
        .map((query) => query.state.data),
    ),
  ).not.toContain('fixture-news-secret');
  expect(JSON.stringify(h.client.getMutationCache().getAll())).not.toContain('fixture-news-secret');
  fireEvent.click(screen.getByRole('link', { name: '系统' }));
  fireEvent.change(await screen.findByLabelText('综合分析模型'), { target: { value: 'retained-draft' } });
  fireEvent.click(screen.getByRole('link', { name: '引擎' }));
  fireEvent.click(await screen.findByRole('button', { name: '更换凭据' }));
  fireEvent.change(await screen.findByLabelText('CoinDesk 新闻密钥（可选）'), {
    target: { value: 'rotated-news-secret' },
  });
  fireEvent.click(screen.getByRole('button', { name: '更新新闻密钥' }));
  await waitFor(() => expect(screen.queryByLabelText('CoinDesk 新闻密钥（可选）')).not.toBeInTheDocument());
  fireEvent.click(screen.getByRole('link', { name: '系统' }));
  expect(await screen.findByLabelText('综合分析模型')).toHaveValue('retained-draft');
  fireEvent.click(screen.getByRole('button', { name: '保存配置' }));
  await waitFor(() => expect(h.writes).toHaveLength(1));
  expect(h.writes[0]!.document.market_data).toEqual({ source_id: 'default', timeframe: '1h', parameters: {} });
});

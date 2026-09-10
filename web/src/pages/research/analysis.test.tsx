import { fireEvent, screen, waitFor } from '@testing-library/react';
import { expect, it } from 'vitest';
import { workflowHarness } from '@/test/configuration-workflow';

it('submits analysis through the primary button without requesting trading', async () => {
  const h = workflowHarness('/research/analysis');
  const button = await screen.findByRole('button', { name: '仅分析，不交易' });
  await waitFor(() => expect(button).toBeEnabled());
  fireEvent.click(button);
  await waitFor(() =>
    expect(h.fetchMock.mock.calls.some(([url, init]) => url.endsWith('/api/analyses') && init?.method === 'POST')).toBe(
      true,
    ),
  );
  const posts = h.fetchMock.mock.calls.filter(([, init]) => init?.method === 'POST');
  expect(posts).toHaveLength(1);
  expect(JSON.parse(posts[0]![1]!.body as string)).toEqual({ pair: 'BTC/USDT:USDT', expected_revision: 1 });
});

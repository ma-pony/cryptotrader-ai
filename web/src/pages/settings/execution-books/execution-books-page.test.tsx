import { fireEvent, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, expect, it } from 'vitest';
import i18n from '@/lib/i18n';
import { workflowConfig, workflowHarness } from '@/test/configuration-workflow';
beforeEach(() => i18n.changeLanguage('zh-CN'));
it('keeps saved book identity immutable and new book ID input focused while typing', async () => {
  workflowHarness('/settings/execution-books');
  expect(await screen.findByLabelText('资金池 ID')).toBeDisabled();
  expect(screen.getByLabelText('资金作用域')).toBeDisabled();
  fireEvent.click(screen.getByRole('button', { name: '新增资金池' }));
  const ids = screen.getAllByLabelText('资金池 ID');
  const id = ids[1]!;
  expect(id).not.toHaveValue('');
  const user = userEvent.setup();
  await user.clear(id);
  await user.type(id, 'new-book');
  expect(id).toHaveFocus();
  expect(id).toHaveValue('new-book');
  fireEvent.click(screen.getAllByRole('button', { name: '移除' })[1]!);
  expect(screen.getAllByLabelText('资金池 ID')).toHaveLength(1);
});
it('excludes disabled, canary and wrong-scope connections and shows exact percentage remainder', async () => {
  const config = workflowConfig();
  const paper = config.document.execution.connections[0]!;
  config.document.execution.connections.push(
    { ...paper, id: 'disabled', label: 'Disabled', enabled: false },
    { ...paper, id: 'canary', label: 'Canary', canary_only: true },
    { ...paper, id: 'live', label: 'Live', adapter_id: 'okx', environment: 'live' },
  );
  workflowHarness('/settings/execution-books', config);
  const weight = await screen.findByLabelText('Paper 权重（%）');
  expect(screen.queryByRole('checkbox', { name: '启用 Disabled' })).not.toBeInTheDocument();
  expect(screen.queryByRole('checkbox', { name: '启用 Canary' })).not.toBeInTheDocument();
  expect(screen.queryByRole('checkbox', { name: '启用 Live' })).not.toBeInTheDocument();
  fireEvent.change(weight, { target: { value: '40' } });
  expect(screen.getByText('已分配 40% · 剩余 60%')).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: '保存配置' }));
  expect(weight).toHaveAttribute('aria-invalid', 'true');
  expect(screen.getByRole('checkbox', { name: '允许实盘下单' })).not.toBeChecked();
});
it('saves dirty books over latest venue and credential writes without reverting other drafts', async () => {
  const h = workflowHarness('/settings/models');
  fireEvent.change(await screen.findByLabelText('综合分析模型'), { target: { value: 'pending-model' } });
  fireEvent.click(screen.getByRole('link', { name: '执行资金池' }));
  fireEvent.change(await screen.findByLabelText('名称'), { target: { value: 'book-draft' } });
  fireEvent.click(screen.getByRole('link', { name: '平台连接' }));
  await screen.findByRole('heading', { name: '平台连接' });
  fireEvent.change(await screen.findByLabelText('名称'), { target: { value: 'fresh-connection' } });
  fireEvent.click(screen.getByRole('button', { name: '保存连接' }));
  await waitFor(() => expect(h.saved().revision).toBe(2));
  fireEvent.click(screen.getByRole('button', { name: '新增连接' }));
  const form = screen.getByRole('form', { name: '新增平台连接' });
  fireEvent.change(within(form).getByLabelText('名称'), { target: { value: 'OKX' } });
  fireEvent.change(within(form).getByLabelText('交易平台'), { target: { value: 'okx' } });
  fireEvent.click(within(form).getByRole('button', { name: '创建连接' }));
  await waitFor(() => expect(h.saved().revision).toBe(3));
  await waitFor(() => expect(screen.queryByRole('form', { name: '新增平台连接' })).not.toBeInTheDocument());
  fireEvent.change(screen.getByLabelText('API Key'), { target: { value: 'fixture-key' } });
  fireEvent.change(screen.getByLabelText('API Secret'), { target: { value: 'fixture-signing' } });
  fireEvent.change(screen.getByLabelText('OKX Passphrase'), { target: { value: 'fixture-phrase' } });
  fireEvent.click(screen.getByRole('button', { name: '保存凭据' }));
  await waitFor(() => expect(h.saved().revision).toBe(4));
  fireEvent.click(screen.getByRole('link', { name: '执行资金池' }));
  await screen.findByRole('heading', { name: '执行资金池' });
  expect(screen.getByLabelText('名称')).toHaveValue('book-draft');
  fireEvent.click(screen.getByRole('button', { name: '保存配置' }));
  await waitFor(() => expect(h.writes).toHaveLength(1));
  expect(h.writes[0]).toMatchObject({
    expected_revision: 4,
    document: {
      execution: { books: [{ label: 'book-draft' }], connections: [{ label: 'fresh-connection' }, { label: 'OKX' }] },
      llm: { models: { analysis: 'analysis' } },
    },
  });
  expect(h.saved().document.execution.connections[1]!.credential_updated_at).toBe('2026-08-30T13:00:00Z');
  fireEvent.click(screen.getByRole('link', { name: '模型与网关' }));
  expect(await screen.findByLabelText('综合分析模型')).toHaveValue('pending-model');
});

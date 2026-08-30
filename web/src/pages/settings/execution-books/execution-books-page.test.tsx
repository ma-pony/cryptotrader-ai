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
it('keeps a new book editable through an existing ID prefix and duplicate validation, then locks it after save', async () => {
  const h = workflowHarness('/settings/execution-books');
  fireEvent.click(await screen.findByRole('button', { name: '新增资金池' }));
  const id = screen.getAllByLabelText('资金池 ID')[1]!;
  const user = userEvent.setup();
  await user.clear(id);
  await user.type(id, 'sim-two');
  expect(id).toHaveValue('sim-two');
  expect(id).toHaveFocus();
  const scope = screen.getAllByLabelText('资金作用域')[1]!;
  expect(scope).toBeEnabled();
  await user.clear(id);
  await user.type(id, 'sim');
  fireEvent.change(screen.getAllByLabelText('名称')[1]!, { target: { value: 'Second book' } });
  fireEvent.click(screen.getAllByLabelText('启用资金池')[1]!);
  fireEvent.click(screen.getByRole('button', { name: '保存配置' }));
  expect(id).toBeEnabled();
  expect(id).toHaveAttribute('aria-invalid', 'true');
  expect(h.writes).toHaveLength(0);
  await user.type(id, '-two');
  fireEvent.click(screen.getByRole('button', { name: '保存配置' }));
  await waitFor(() => expect(h.saved().document.execution.books).toHaveLength(2));
  await waitFor(() => expect(screen.getAllByLabelText('资金池 ID')[1]!).toHaveAttribute('disabled'));
  expect(screen.getAllByLabelText('资金作用域')[1]!).toHaveAttribute('disabled');
});

it('maps unallocated weights to unique stable controls and focuses indexed server errors', async () => {
  const config = workflowConfig();
  const paper = config.document.execution.connections[0]!;
  config.document.execution.connections.push(
    { ...paper, id: 'second', label: 'Second' },
    { ...paper, id: 'third', label: 'Third' },
  );
  const h = workflowHarness('/settings/execution-books', config);
  const second = await screen.findByLabelText<HTMLInputElement>('Second 权重（%）');
  const third = screen.getByLabelText<HTMLInputElement>('Third 权重（%）');
  expect(second).not.toBe(third);
  expect(second.id).not.toBe(third.id);
  expect(second.name).not.toBe(third.name);
  const secondId = second.id;
  fireEvent.click(screen.getByRole('checkbox', { name: '启用 Paper' }));
  fireEvent.click(screen.getByRole('checkbox', { name: '启用 Second' }));
  expect(screen.getByLabelText('Second 权重（%）')).toBe(second);
  expect(second.id).toBe(secondId);
  h.fail(422, [
    {
      loc: ['body', 'document', 'execution', 'books', 0, 'allocations', 1, 'weight'],
      msg: 'invalid',
      type: 'value_error',
    },
  ]);
  fireEvent.click(screen.getByRole('button', { name: '保存配置' }));
  await waitFor(() => expect(second).toHaveAttribute('aria-invalid', 'true'));
  expect(second).toHaveFocus();
  expect(second.labels?.[0]?.control).toBe(second);
  expect(third.labels?.[0]?.control).toBe(third);
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

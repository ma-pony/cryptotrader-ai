import { act, fireEvent, screen, waitFor, within } from '@testing-library/react';
import { beforeEach, expect, it } from 'vitest';
import i18n from '@/lib/i18n';
import { workflowConfig, workflowHarness } from '@/test/configuration-workflow';
import { RUNTIME_CONFIG_QUERY_KEY } from '@/hooks/use-runtime-config';
beforeEach(() => i18n.changeLanguage('zh-CN'));
it('rejects zero Paper capital locally with an associated error and accepts a small positive amount', async () => {
  const h = workflowHarness('/settings/venues');
  const amount = await screen.findByLabelText('模拟初始资金（USDT）');
  fireEvent.change(amount, { target: { value: '0' } });
  fireEvent.click(screen.getByRole('button', { name: '保存连接' }));
  expect(amount).toHaveAttribute('aria-invalid', 'true');
  expect(amount).toHaveAccessibleDescription(/数字/);
  expect(amount).toHaveFocus();
  expect(h.fetchMock.mock.calls.filter(([, init]) => init?.method === 'PUT')).toHaveLength(0);
  fireEvent.change(amount, { target: { value: '0.00001' } });
  fireEvent.click(screen.getByRole('button', { name: '保存连接' }));
  await waitFor(() => expect(h.saved().revision).toBe(2));
  expect(h.saved().document.execution.connections[0]!.parameters[0]!.value.number_value).toBe('0.00001');
});

it('saves an edited connection through the strict update contract and then enables its read-only check', async () => {
  const config = workflowConfig();
  config.document.execution.connections[0] = {
    ...config.document.execution.connections[0]!,
    parameters: [{
      key: 'initial_equity',
      value: {
        kind: 'number',
        boolean_value: null,
        number_value: '0.00001',
        string_value: null,
        datetime_value: null,
        pair_value: null,
        items: [],
        entries: [],
      },
    }],
  };
  const h = workflowHarness('/settings/venues', config);
  const amount = await screen.findByLabelText('模拟初始资金（USDT）');
  fireEvent.change(amount, { target: { value: '10000' } });
  fireEvent.click(screen.getByRole('button', { name: '保存连接' }));

  await waitFor(() => expect(h.saved().revision).toBe(2));
  expect(h.saved().document.execution.connections[0]).toMatchObject({
    id: 'paper',
    parameters: [{ key: 'initial_equity', value: { number_value: '10000' } }],
  });
  const update = h.fetchMock.mock.calls.find(
    ([url, init]) => url.endsWith('/api/venue-connections/paper') && init?.method === 'PUT',
  );
  expect(JSON.parse((update![1] as RequestInit).body as string)).toEqual({
    expected_revision: 1,
    label: 'Paper',
    adapter_id: 'paper',
    environment: 'paper',
    enabled: true,
    leverage: 1,
    margin_mode: 'cross',
    canary_only: false,
    parameters: { initial_equity: 10000 },
  });
  expect(screen.getByRole('button', { name: '保存连接' })).toBeDisabled();
  const check = screen.getByRole('button', { name: '只读检查' });
  expect(check).toBeEnabled();
  fireEvent.click(check);
  expect(await screen.findByText(/账户读取已验证/)).toBeInTheDocument();
});

it('describes fixed shared margin for Paper while keeping leverage and external mode choices', async () => {
  workflowHarness('/settings/venues');
  fireEvent.click(await screen.findByRole('button', { name: '新增连接' }));
  const form = screen.getByRole('form', { name: '新增平台连接' });
  expect(within(form).queryByLabelText('保证金模式')).not.toBeInTheDocument();
  expect(within(form).getByText(/全仓.*共享账户权益/)).toBeInTheDocument();
  expect(within(form).getByLabelText('杠杆（倍）')).toHaveValue(1);
  for (const adapter of ['okx', 'bybit']) {
    fireEvent.change(within(form).getByLabelText('交易平台'), { target: { value: adapter } });
    const mode = within(form).getByLabelText('保证金模式');
    expect(within(mode).getAllByRole('option').map((option) => (option as HTMLOptionElement).value)).toEqual(['cross', 'isolated']);
    fireEvent.change(mode, { target: { value: 'isolated' } });
    expect(mode).toHaveValue('isolated');
  }
});
it('starts Paper at editable 10000 USDT with generated identity and no credentials', async () => {
  const h = workflowHarness('/settings/venues');
  fireEvent.click(await screen.findByRole('button', { name: '新增连接' }, { timeout: 5000 }));
  const form = screen.getByRole('form', { name: '新增平台连接' });
  expect(within(form).getByLabelText('模拟初始资金（USDT）')).toHaveValue(10000);
  expect(within(form).queryByLabelText('API Key')).not.toBeInTheDocument();
  expect(within(form).getByLabelText('连接 ID')).not.toHaveValue('');
  fireEvent.change(within(form).getByLabelText('名称'), { target: { value: '新模拟账户' } });
  fireEvent.change(within(form).getByLabelText('模拟初始资金（USDT）'), { target: { value: '25000' } });
  fireEvent.click(within(form).getByRole('button', { name: '创建连接' }));
  await waitFor(() => expect(h.saved().document.execution.connections).toHaveLength(2));
  await waitFor(() => expect(screen.queryByRole('form', { name: '新增平台连接' })).not.toBeInTheDocument());
  const added = h.saved().document.execution.connections[1]!;
  expect(added.parameters[0]).toMatchObject({ key: 'initial_equity', value: { number_value: '25000' } });
  expect(h.saved().document.system.active).toBe(false);
});
it('uses catalog environments and required OKX passphrase, with no check on unsaved connection', async () => {
  workflowHarness('/settings/venues');
  fireEvent.click(await screen.findByRole('button', { name: '新增连接' }));
  const form = screen.getByRole('form', { name: '新增平台连接' });
  fireEvent.change(within(form).getByLabelText('交易平台'), { target: { value: 'okx' } });
  expect(
    within(within(form).getByLabelText('环境'))
      .getAllByRole('option')
      .map((option) => (option as HTMLOptionElement).value),
  ).toEqual(['demo', 'live']);
  expect(within(form).getByLabelText('OKX Passphrase')).toBeRequired();
  expect(within(form).getByRole('button', { name: '保存凭据' })).toBeDisabled();
  expect(within(form).queryByRole('button', { name: '只读检查' })).not.toBeInTheDocument();
});
it('hides old check success on draft changes and credential timestamp refresh', async () => {
  const config = workflowConfig();
  config.document.execution.connections = [
    {
      ...config.document.execution.connections[0]!,
      id: 'okx',
      adapter_id: 'okx',
      environment: 'demo',
      credential_configured: true,
      credential_updated_at: '2026-08-30T00:00:00Z',
    },
  ];
  const h = workflowHarness('/settings/venues', config);
  fireEvent.click(await screen.findByRole('button', { name: '只读检查' }));
  const verified = await screen.findByText(/账户读取已验证/);
  expect(verified.querySelector('time')).toHaveAttribute('dateTime', '2026-08-30T12:00:00Z');
  expect(verified).not.toHaveTextContent('T12:00:00Z');
  fireEvent.change(screen.getByLabelText('名称'), { target: { value: 'changed' } });
  expect(screen.queryByText(/账户读取已验证/)).not.toBeInTheDocument();
  expect(screen.getByRole('button', { name: '只读检查' })).toBeDisabled();
  fireEvent.change(screen.getByLabelText('名称'), { target: { value: 'Paper' } });
  const saved = h.saved();
  act(() => {
    h.client.setQueryData(RUNTIME_CONFIG_QUERY_KEY, {
      ...saved,
      revision: 2,
      document: {
        ...saved.document,
        execution: {
          ...saved.document.execution,
          connections: saved.document.execution.connections.map((item) => ({
            ...item,
            credential_updated_at: '2026-08-30T14:00:00Z',
          })),
        },
      },
    });
  });
  await waitFor(() => expect(screen.queryByText(/账户读取已验证/)).not.toBeInTheDocument());
});
it('renders safe read-only check failures from the real FastAPI envelope', async () => {
  const h = workflowHarness('/settings/venues');
  h.fail(401, { code: 'authentication_failed' });
  fireEvent.click(await screen.findByRole('button', { name: '只读检查' }));
  expect(await screen.findByRole('alert')).toHaveTextContent(/验证失败/);
  expect(screen.queryByText(/账户读取已验证/)).not.toBeInTheDocument();
});

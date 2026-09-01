import { fireEvent, screen, waitFor, within } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import i18n from '@/lib/i18n';
import { useSettingsStore } from '@/stores/use-settings-store';
import {
  workflowApprovedChecks,
  workflowCatalog,
  workflowConfig,
  workflowHarness,
} from '@/test/configuration-workflow';
import { pluginFields } from '@/test/configuration-catalog-fixture';

beforeEach(async () => {
  await i18n.changeLanguage('zh-CN');
  useSettingsStore.getState().reset();
});

it('uses the server definition to save arbitrary credentials then performs one read-only check', async () => {
  const h = workflowHarness('/accounts/connections');

  fireEvent.click(await screen.findByRole('button', { name: '新增连接' }));
  const form = within(screen.getByRole('form', { name: '新增平台连接' }));
  fireEvent.change(form.getByRole('combobox', { name: /交易平台/ }), { target: { value: 'sample_venue' } });
  await waitFor(() => expect(form.getByRole('button', { name: '保存并检查' })).toBeEnabled());
  fireEvent.change(form.getByLabelText('名称'), { target: { value: '沙盒账户' } });
  fireEvent.change(form.getByLabelText('账户编码'), { target: { value: 'sandbox-account' } });
  fireEvent.change(form.getByLabelText('访问令牌'), { target: { value: ' fixture-token ' } });
  fireEvent.click(form.getByRole('button', { name: '保存并检查' }));

  await screen.findByText('凭据已配置');
  expect(screen.queryByDisplayValue(' fixture-token ')).not.toBeInTheDocument();
  const credentialWrites = h.fetchMock.mock.calls.filter(
    ([url, init]) => url.includes('/credentials') && init?.method === 'PUT',
  );
  expect(credentialWrites).toHaveLength(1);
  const requestBody = credentialWrites[0]![1]!.body;
  expect(typeof requestBody).toBe('string');
  const credentialPayload = JSON.parse(requestBody as string) as {
    expected_revision: number;
    values: Record<string, string>;
  };
  expect(credentialPayload).toEqual({
    expected_revision: 2,
    values: { account_code: 'sandbox-account', access_token: ' fixture-token ' },
  });
  await waitFor(() => expect(h.fetchMock.mock.calls.filter(([url]) => url.endsWith('/test'))).toHaveLength(1));
  expect(
    JSON.stringify(
      h.client
        .getQueryCache()
        .getAll()
        .map((query) => query.state),
    ),
  ).not.toContain('fixture-token');
  expect(JSON.stringify(h.client.getMutationCache().getAll())).not.toContain('fixture-token');
});

it('does not let a late default-venue definition clear a user-edited sample parameter', async () => {
  const sample = workflowCatalog.venues.find((venue) => venue.id === 'sample_venue')!;
  const bybit = {
    ...sample,
    id: 'bybit',
    label: { zh_CN: 'Bybit', en_US: 'Bybit' },
    environments: [{ id: 'testnet', label: { zh_CN: '测试网', en_US: 'Testnet' }, capital_scope: 'simulated' as const }],
    fields: [],
    credential_fields: [],
  };
  const catalog = { ...workflowCatalog, venues: [bybit, ...workflowCatalog.venues] };
  let resolveBybit!: (response: Response) => void;
  const lateBybit = new Promise<Response>((resolve) => {
    resolveBybit = resolve;
  });
  const response = (value: unknown) => Promise.resolve(new Response(JSON.stringify(value), { status: 200 }));
  workflowHarness('/accounts/connections', workflowConfig(), undefined, new Map(), (url) => {
    if (url.endsWith('/api/config/catalog')) return response(catalog);
    if (url.includes('/api/config/catalog/venues/bybit')) return lateBybit;
    return undefined;
  });

  fireEvent.click(await screen.findByRole('button', { name: '新增连接' }));
  const form = within(screen.getByRole('form', { name: '新增平台连接' }));
  fireEvent.change(form.getByRole('combobox', { name: /交易平台/ }), { target: { value: 'sample_venue' } });
  await waitFor(() => expect(form.getByLabelText('触发阈值')).toBeInTheDocument());
  fireEvent.change(form.getByLabelText('触发阈值'), { target: { value: '0.8' } });

  resolveBybit(
    new Response(
      JSON.stringify({
        id: bybit.id,
        label: bybit.label,
        description: bybit.description,
        environment: bybit.environments[0],
        fields: [],
        credential_fields: [],
        margin_modes: ['cross'],
        leverage_minimum: 1,
        leverage_maximum: null,
        account_read: true,
        capabilities: null,
      }),
      { status: 200 },
    ),
  );

  await waitFor(() => expect(form.getByLabelText('触发阈值')).toHaveValue(0.8));
});

it('saves basic connection details and lists missing required credentials', async () => {
  const h = workflowHarness('/accounts/connections');
  fireEvent.click(await screen.findByRole('button', { name: '新增连接' }));
  const form = within(screen.getByRole('form', { name: '新增平台连接' }));
  fireEvent.change(form.getByRole('combobox', { name: /交易平台/ }), { target: { value: 'sample_venue' } });
  await waitFor(() => expect(form.getByRole('button', { name: '保存并检查' })).toBeEnabled());
  fireEvent.change(form.getByLabelText('名称'), { target: { value: '仅保存连接' } });
  fireEvent.click(form.getByRole('button', { name: '保存并检查' }));

  await screen.findByText('连接已保存，仍缺少必填凭据：账户编码、访问令牌。');
  expect(
    h.fetchMock.mock.calls.filter(([url, init]) => url.includes('/api/venue-connections') && init?.method === 'POST'),
  ).toHaveLength(1);
  expect(
    h.fetchMock.mock.calls.filter(([url, init]) => url.includes('/credentials') && init?.method === 'PUT'),
  ).toHaveLength(0);
});

it('associates environment validation errors and focuses the invalid ordinary field before any write', async () => {
  const h = workflowHarness('/accounts/connections');
  fireEvent.click(await screen.findByRole('button', { name: '新增连接' }));
  const form = within(screen.getByRole('form', { name: '新增平台连接' }));
  fireEvent.change(form.getByRole('combobox', { name: /交易平台/ }), { target: { value: 'sample_venue' } });
  await waitFor(() => expect(form.getByRole('button', { name: '保存并检查' })).toBeEnabled());
  fireEvent.change(form.getByLabelText('名称'), { target: { value: '无效杠杆' } });
  fireEvent.change(form.getByLabelText('杠杆（倍）'), { target: { value: '0' } });
  fireEvent.click(form.getByRole('button', { name: '保存并检查' }));
  expect(form.getByLabelText('杠杆（倍）')).toHaveAttribute('aria-invalid', 'true');
  expect(form.getByLabelText('杠杆（倍）')).toHaveFocus();
  expect(
    h.fetchMock.mock.calls.filter(([url, init]) => url.includes('/api/venue-connections') && init?.method === 'POST'),
  ).toHaveLength(0);
});

it('rejects a non-multiple environment parameter step before saving, then accepts a valid multiple', async () => {
  const h = workflowHarness('/accounts/connections');
  fireEvent.click(await screen.findByRole('button', { name: '新增连接' }));
  const form = within(screen.getByRole('form', { name: '新增平台连接' }));
  fireEvent.change(form.getByRole('combobox', { name: /交易平台/ }), { target: { value: 'sample_venue' } });
  await waitFor(() => expect(form.getByRole('button', { name: '保存并检查' })).toBeEnabled());
  fireEvent.change(form.getByLabelText('名称'), { target: { value: '步长校验' } });
  fireEvent.change(form.getByLabelText('触发阈值'), { target: { value: '0.55' } });
  fireEvent.click(form.getByRole('button', { name: '保存并检查' }));
  expect(form.getByLabelText('触发阈值')).toHaveAttribute('aria-invalid', 'true');
  expect(form.getByLabelText('触发阈值')).toHaveFocus();
  expect(
    h.fetchMock.mock.calls.filter(([url, init]) => url.includes('/api/venue-connections') && init?.method === 'POST'),
  ).toHaveLength(0);
  fireEvent.change(form.getByLabelText('触发阈值'), { target: { value: '0.8' } });
  fireEvent.click(form.getByRole('button', { name: '保存并检查' }));
  await screen.findByText('连接已保存，仍缺少必填凭据：账户编码、访问令牌。');
  expect(
    h.fetchMock.mock.calls.filter(([url, init]) => url.includes('/api/venue-connections') && init?.method === 'POST'),
  ).toHaveLength(1);
});

it('submits environment-definition defaults rather than platform-directory defaults', async () => {
  const h = workflowHarness('/accounts/connections');
  fireEvent.click(await screen.findByRole('button', { name: '新增连接' }));
  const form = within(screen.getByRole('form', { name: '新增平台连接' }));
  fireEvent.change(form.getByRole('combobox', { name: /交易平台/ }), { target: { value: 'sample_venue' } });
  await waitFor(() => expect(form.getByRole('button', { name: '保存并检查' })).toBeEnabled());
  fireEvent.change(form.getByLabelText('名称'), { target: { value: '环境默认值' } });
  fireEvent.click(form.getByRole('button', { name: '保存并检查' }));
  await screen.findByText('连接已保存，仍缺少必填凭据：账户编码、访问令牌。');
  const request = h.fetchMock.mock.calls.find(
    ([url, init]) => url.includes('/api/venue-connections') && init?.method === 'POST',
  )!;
  expect(JSON.parse(request[1]!.body as string)).toMatchObject({
    margin_mode: 'isolated',
    parameters: { threshold: 0.5 },
  });
});

it('stops after a confirmed connection write when its refresh needs recovery', async () => {
  const h = workflowHarness('/accounts/connections');
  fireEvent.click(await screen.findByRole('button', { name: '新增连接' }));
  const form = within(screen.getByRole('form', { name: '新增平台连接' }));
  fireEvent.change(form.getByRole('combobox', { name: /交易平台/ }), { target: { value: 'sample_venue' } });
  await waitFor(() => expect(form.getByRole('button', { name: '保存并检查' })).toBeEnabled());
  fireEvent.change(form.getByLabelText('名称'), { target: { value: '待刷新连接' } });
  fireEvent.change(form.getByLabelText('账户编码'), { target: { value: 'account' } });
  fireEvent.change(form.getByLabelText('访问令牌'), { target: { value: 'token' } });
  h.failReload(true);
  fireEvent.click(form.getByRole('button', { name: '保存并检查' }));

  await screen.findByText('连接已保存，但配置刷新失败；请重新加载后继续。');
  expect(
    h.fetchMock.mock.calls.filter(([url, init]) => url.includes('/credentials') && init?.method === 'PUT'),
  ).toHaveLength(0);
  expect(h.fetchMock.mock.calls.filter(([url]) => url.endsWith('/test'))).toHaveLength(0);
});

it('continues with credentials after recovering a newly created connection refresh', async () => {
  const h = workflowHarness('/accounts/connections');
  fireEvent.click(await screen.findByRole('button', { name: '新增连接' }));
  const form = within(screen.getByRole('form', { name: '新增平台连接' }));
  fireEvent.change(form.getByRole('combobox', { name: /交易平台/ }), { target: { value: 'sample_venue' } });
  await waitFor(() => expect(form.getByRole('button', { name: '保存并检查' })).toBeEnabled());
  fireEvent.change(form.getByLabelText('名称'), { target: { value: '恢复后继续' } });
  fireEvent.change(form.getByLabelText('账户编码'), { target: { value: 'account' } });
  fireEvent.change(form.getByLabelText('访问令牌'), { target: { value: 'token' } });
  h.failReload(true);
  fireEvent.click(form.getByRole('button', { name: '保存并检查' }));
  await screen.findByText('连接已保存，但配置刷新失败；请重新加载后继续。');
  h.failReload(false);
  fireEvent.click(screen.getByRole('button', { name: '重新加载' }));
  await waitFor(() =>
    expect(within(screen.getAllByRole('form').at(-1)!).getByRole('button', { name: '保存并检查' })).toBeEnabled(),
  );
  const recovered = within(screen.getAllByRole('form').at(-1)!);
  fireEvent.click(recovered.getByRole('button', { name: '保存并检查' }));
  await screen.findByText('凭据已配置');
  expect(
    h.fetchMock.mock.calls.filter(([url, init]) => url.endsWith('/api/venue-connections') && init?.method === 'POST'),
  ).toHaveLength(1);
  expect(
    h.fetchMock.mock.calls.filter(([url, init]) => url.includes('/credentials') && init?.method === 'PUT'),
  ).toHaveLength(1);
});

it('keeps credential input visible when the connection save succeeded but credential save failed', async () => {
  const h = workflowHarness('/accounts/connections');

  fireEvent.click(await screen.findByRole('button', { name: '新增连接' }));
  const form = within(screen.getByRole('form', { name: '新增平台连接' }));
  fireEvent.change(form.getByRole('combobox', { name: /交易平台/ }), { target: { value: 'sample_venue' } });
  await waitFor(() => expect(form.getByRole('button', { name: '保存并检查' })).toBeEnabled());
  fireEvent.change(form.getByLabelText('名称'), { target: { value: '失败后保留' } });
  fireEvent.change(form.getByLabelText('账户编码'), { target: { value: 'retain-account' } });
  fireEvent.change(form.getByLabelText('访问令牌'), { target: { value: 'retain-token' } });
  h.failCredentials(422, 'provider-detail-must-not-render');
  fireEvent.click(form.getByRole('button', { name: '保存并检查' }));

  await screen.findByText('连接已保存，凭据保存失败。本次输入仍保留，请检查后重试。');
  expect(screen.getByLabelText('账户编码')).toHaveValue('retain-account');
  expect(screen.getByLabelText('访问令牌')).toHaveValue('retain-token');
  expect(h.saved().document.execution.connections).toHaveLength(2);
  expect(screen.queryByText('凭据已配置')).not.toBeInTheDocument();
  expect(document.body.textContent).not.toContain('provider-detail-must-not-render');
});

it('hides a prior successful check while normal fields or replacement credentials are pending', async () => {
  const h = workflowHarness('/accounts/connections');
  const form = await screen.findByRole('form', { name: 'Paper' });
  fireEvent.click(within(form).getByRole('button', { name: '保存并检查' }));
  await screen.findByText(/账户读取已验证/);
  fireEvent.change(within(form).getByLabelText('名称'), { target: { value: '待保存名称' } });
  expect(screen.queryByText(/账户读取已验证/)).not.toBeInTheDocument();
  expect(h.fetchMock.mock.calls.filter(([url]) => url.endsWith('/test'))).toHaveLength(1);
});

it('does not reuse an old green check while replacing configured credentials', async () => {
  const h = workflowHarness('/accounts/connections');
  fireEvent.click(await screen.findByRole('button', { name: '新增连接' }));
  const form = within(screen.getByRole('form', { name: '新增平台连接' }));
  fireEvent.change(form.getByRole('combobox', { name: /交易平台/ }), { target: { value: 'sample_venue' } });
  await waitFor(() => expect(form.getByRole('button', { name: '保存并检查' })).toBeEnabled());
  fireEvent.change(form.getByLabelText('名称'), { target: { value: '替换凭据' } });
  fireEvent.change(form.getByLabelText('账户编码'), { target: { value: 'account' } });
  fireEvent.change(form.getByLabelText('访问令牌'), { target: { value: 'token' } });
  fireEvent.click(form.getByRole('button', { name: '保存并检查' }));
  await screen.findByText(/账户读取已验证/);
  fireEvent.click(screen.getByRole('button', { name: '更换凭据' }));
  expect(screen.queryByText(/账户读取已验证/)).not.toBeInTheDocument();
  expect(screen.getByLabelText('访问令牌')).toHaveValue('');
  expect(h.fetchMock.mock.calls.filter(([url]) => url.endsWith('/test'))).toHaveLength(1);
});

it('rotates configured credentials without a normal connection update and performs one new check', async () => {
  const h = workflowHarness('/accounts/connections');
  fireEvent.click(await screen.findByRole('button', { name: '新增连接' }));
  const newForm = within(screen.getByRole('form', { name: '新增平台连接' }));
  fireEvent.change(newForm.getByRole('combobox', { name: /交易平台/ }), { target: { value: 'sample_venue' } });
  await waitFor(() => expect(newForm.getByRole('button', { name: '保存并检查' })).toBeEnabled());
  fireEvent.change(newForm.getByLabelText('名称'), { target: { value: '轮换凭据' } });
  fireEvent.change(newForm.getByLabelText('账户编码'), { target: { value: 'old-account' } });
  fireEvent.change(newForm.getByLabelText('访问令牌'), { target: { value: 'old-token' } });
  fireEvent.click(newForm.getByRole('button', { name: '保存并检查' }));
  await screen.findByText(/账户读取已验证/);
  const savedForm = within(screen.getAllByRole('form').at(-1)!);
  const checksBefore = h.fetchMock.mock.calls.filter(([url]) => url.endsWith('/test')).length;
  fireEvent.click(savedForm.getByRole('button', { name: '更换凭据' }));
  fireEvent.change(savedForm.getByLabelText('账户编码'), { target: { value: 'new-account' } });
  fireEvent.change(savedForm.getByLabelText('访问令牌'), { target: { value: 'new-token' } });
  fireEvent.click(savedForm.getByRole('button', { name: '保存并检查' }));
  await screen.findByText(/账户读取已验证/);
  expect(
    h.fetchMock.mock.calls.filter(
      ([url, init]) => /\/api\/venue-connections\/[^/]+$/.test(url) && init?.method === 'PUT',
    ),
  ).toHaveLength(0);
  const writes = h.fetchMock.mock.calls.filter(([url, init]) => url.includes('/credentials') && init?.method === 'PUT');
  expect(writes).toHaveLength(2);
  expect(JSON.parse(writes[1]![1]!.body as string)).toEqual({
    expected_revision: 3,
    values: { account_code: 'new-account', access_token: 'new-token' },
  });
  expect(h.fetchMock.mock.calls.filter(([url]) => url.endsWith('/test'))).toHaveLength(checksBefore + 1);
});

it('cancels credential replacement without retaining typed values or writing credentials', async () => {
  const h = workflowHarness('/accounts/connections');
  fireEvent.click(await screen.findByRole('button', { name: '新增连接' }));
  const newForm = within(screen.getByRole('form', { name: '新增平台连接' }));
  fireEvent.change(newForm.getByRole('combobox', { name: /交易平台/ }), { target: { value: 'sample_venue' } });
  await waitFor(() => expect(newForm.getByRole('button', { name: '保存并检查' })).toBeEnabled());
  fireEvent.change(newForm.getByLabelText('名称'), { target: { value: '取消轮换' } });
  fireEvent.change(newForm.getByLabelText('账户编码'), { target: { value: 'old-account' } });
  fireEvent.change(newForm.getByLabelText('访问令牌'), { target: { value: 'old-token' } });
  fireEvent.click(newForm.getByRole('button', { name: '保存并检查' }));
  await screen.findByText('凭据已配置');
  const savedForm = within(screen.getAllByRole('form').at(-1)!);
  fireEvent.click(savedForm.getByRole('button', { name: '更换凭据' }));
  fireEvent.change(savedForm.getByLabelText('账户编码'), { target: { value: 'discard-account' } });
  fireEvent.change(savedForm.getByLabelText('访问令牌'), { target: { value: 'discard-token' } });
  fireEvent.click(savedForm.getByRole('button', { name: '取消更换' }));
  fireEvent.click(savedForm.getByRole('button', { name: '更换凭据' }));
  expect(savedForm.getByLabelText('账户编码')).toHaveValue('');
  expect(savedForm.getByLabelText('访问令牌')).toHaveValue('');
  expect(
    h.fetchMock.mock.calls.filter(
      ([url, init]) => url.includes('/credentials') && ['PUT', 'DELETE'].includes(String(init?.method)),
    ),
  ).toHaveLength(1);
  expect(screen.getByText('凭据已配置')).toBeInTheDocument();
});

it.each([0.8, 4.3, 123.1])(
  'saves an ordinary parameter change to %s without credential writes and replaces the old check',
  async (value) => {
    const config = workflowConfig();
    config.document.execution.connections[0] = {
      ...config.document.execution.connections[0]!,
      label: '样本已保存',
      adapter_id: 'sample_venue',
      environment: 'sandbox',
      margin_mode: 'isolated',
      credential_configured: true,
      credential_updated_at: '2026-08-30T00:00:00Z',
      parameters: [{ key: 'threshold', value: pluginFields[0]!.default_value }],
    };
    const h = workflowHarness('/accounts/connections', config, undefined, workflowApprovedChecks(config));
    if (value > 1) {
      const fetchFixture = h.fetchMock.getMockImplementation()!;
      h.fetchMock.mockImplementation(async (url, init) => {
        const response = await fetchFixture(url, init);
        if (!url.includes('/api/config/catalog/venues/sample_venue?')) return response;
        const definition = (await response.json()) as Record<string, unknown>;
        return new Response(
          JSON.stringify({
            ...definition,
            fields: [{ ...pluginFields[0]!, maximum: 200 }],
          }),
          { status: 200 },
        );
      });
    }
    const form = within(await screen.findByRole('form', { name: '样本已保存' }));
    expect(await screen.findByText(/账户读取已验证/)).toBeInTheDocument();
    await waitFor(() => expect(form.getByLabelText('触发阈值')).toBeInTheDocument());
    fireEvent.change(form.getByLabelText('触发阈值'), { target: { value: String(value) } });
    expect(screen.queryByText(/账户读取已验证/)).not.toBeInTheDocument();
    fireEvent.click(form.getByRole('button', { name: '保存并检查' }));
    expect(form.getByLabelText('触发阈值')).toHaveAttribute('aria-invalid', 'false');
    await screen.findByText(/账户读取已验证/);
    const updates = h.fetchMock.mock.calls.filter(
      ([url, init]) => url.endsWith('/api/venue-connections/paper') && init?.method === 'PUT',
    );
    expect(updates).toHaveLength(1);
    expect(JSON.parse(updates[0]![1]!.body as string)).toMatchObject({ parameters: { threshold: value } });
    expect(
      h.fetchMock.mock.calls.filter(([url, init]) => url.includes('/credentials') && init?.method === 'PUT'),
    ).toHaveLength(0);
    expect(h.fetchMock.mock.calls.filter(([url]) => url.endsWith('/test'))).toHaveLength(1);
  },
);

it('stops after a confirmed credential write when refresh needs recovery', async () => {
  const h = workflowHarness('/accounts/connections');
  fireEvent.click(await screen.findByRole('button', { name: '新增连接' }));
  const form = within(screen.getByRole('form', { name: '新增平台连接' }));
  fireEvent.change(form.getByRole('combobox', { name: /交易平台/ }), { target: { value: 'sample_venue' } });
  await waitFor(() => expect(form.getByRole('button', { name: '保存并检查' })).toBeEnabled());
  fireEvent.change(form.getByLabelText('名称'), { target: { value: '先保存再配置凭据' } });
  fireEvent.click(form.getByRole('button', { name: '保存并检查' }));
  await screen.findByText('连接已保存，仍缺少必填凭据：账户编码、访问令牌。');
  const savedForm = within(screen.getAllByRole('form').at(-1)!);
  fireEvent.change(savedForm.getByLabelText('账户编码'), { target: { value: 'account' } });
  fireEvent.change(savedForm.getByLabelText('访问令牌'), { target: { value: 'token' } });
  h.failReload(true);
  fireEvent.click(savedForm.getByRole('button', { name: '保存并检查' }));
  await screen.findByText('凭据已保存，但配置刷新失败；请重新加载后继续。');
  expect(screen.queryByLabelText('访问令牌')).not.toBeInTheDocument();
  expect(h.fetchMock.mock.calls.filter(([url]) => url.endsWith('/test'))).toHaveLength(0);
});

it('deletes configured credentials only after explicit confirmation', async () => {
  const h = workflowHarness('/accounts/connections');
  fireEvent.click(await screen.findByRole('button', { name: '新增连接' }));
  const form = within(screen.getByRole('form', { name: '新增平台连接' }));
  fireEvent.change(form.getByRole('combobox', { name: /交易平台/ }), { target: { value: 'sample_venue' } });
  await waitFor(() => expect(form.getByRole('button', { name: '保存并检查' })).toBeEnabled());
  fireEvent.change(form.getByLabelText('名称'), { target: { value: '待删除凭据' } });
  fireEvent.change(form.getByLabelText('账户编码'), { target: { value: 'account' } });
  fireEvent.change(form.getByLabelText('访问令牌'), { target: { value: 'token' } });
  fireEvent.click(form.getByRole('button', { name: '保存并检查' }));
  await screen.findByText(/账户读取已验证/);
  vi.stubGlobal('confirm', () => true);
  fireEvent.click(screen.getByRole('button', { name: '删除凭据' }));
  await waitFor(() =>
    expect(
      h.fetchMock.mock.calls.filter(([url, init]) => url.includes('/credentials?') && init?.method === 'DELETE'),
    ).toHaveLength(1),
  );
  await waitFor(() => expect(screen.queryByText('凭据已配置')).not.toBeInTheDocument());
  expect(screen.queryByText(/账户读取已验证/)).not.toBeInTheDocument();
  expect(h.saved().document.execution.connections.find((item) => item.id !== 'paper')).toMatchObject({ enabled: true });
  expect(h.saved().document.execution.books[0]).toMatchObject({
    enabled: true,
    allocations: [{ connection_id: 'paper', enabled: true }],
  });
  expect(h.saved().document).toMatchObject({ execution: { live_order_execution_enabled: false } });
});

it('cancels credential deletion without a write', async () => {
  const h = workflowHarness('/accounts/connections');
  fireEvent.click(await screen.findByRole('button', { name: '新增连接' }));
  const form = within(screen.getByRole('form', { name: '新增平台连接' }));
  fireEvent.change(form.getByRole('combobox', { name: /交易平台/ }), { target: { value: 'sample_venue' } });
  await waitFor(() => expect(form.getByRole('button', { name: '保存并检查' })).toBeEnabled());
  fireEvent.change(form.getByLabelText('名称'), { target: { value: '取消删除' } });
  fireEvent.change(form.getByLabelText('账户编码'), { target: { value: 'account' } });
  fireEvent.change(form.getByLabelText('访问令牌'), { target: { value: 'token' } });
  fireEvent.click(form.getByRole('button', { name: '保存并检查' }));
  await screen.findByText('凭据已配置');
  vi.stubGlobal('confirm', () => false);
  fireEvent.click(screen.getByRole('button', { name: '删除凭据' }));
  expect(
    h.fetchMock.mock.calls.filter(([url, init]) => url.includes('/credentials?') && init?.method === 'DELETE'),
  ).toHaveLength(0);
  expect(screen.getByText('凭据已配置')).toBeInTheDocument();
});

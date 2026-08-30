import { useState } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import i18n from '@/lib/i18n';
import { toRuntimeDocument } from '@/hooks/use-runtime-config';
import { runtimeConfigFixture } from '@/test/runtime-config-fixture';
import { configurationCatalogFixture, pluginFields } from '@/test/configuration-catalog-fixture';
import type { ConfigurationDraft, RuntimeDocument, RuntimeJsonObject } from '@/types/api';
import { ParameterFields } from './parameter-fields';
import { ModelSettings } from '@/pages/settings/forms/model-settings';
import { SignalSettings } from '@/pages/settings/forms/signal-settings';
import { MarketSettings } from '@/pages/settings/forms/market-settings';
import { RiskSettings } from '@/pages/settings/forms/risk-settings';
import { SchedulerSettings } from '@/pages/settings/forms/scheduler-settings';
import { SystemSettings } from '@/pages/settings/forms/system-settings';

afterEach(() => vi.unstubAllGlobals());

it('edits installed plugin number, choice, boolean and nested debate fields as typed values', () => {
  function Harness() {
    const [value, setValue] = useState<RuntimeJsonObject>({ debate: { max_rounds: 3 } });
    return (
      <>
        <ParameterFields fields={pluginFields} value={value} onChange={setValue} idPrefix="fixture" />
        <output data-testid="value">{JSON.stringify(value)}</output>
      </>
    );
  }
  render(<Harness />);
  fireEvent.change(screen.getByRole('spinbutton', { name: '触发阈值' }), { target: { value: '0.7' } });
  fireEvent.change(screen.getByRole('combobox', { name: '运行模式' }), { target: { value: 'relaxed' } });
  fireEvent.click(screen.getByRole('checkbox', { name: '启用交叉检查' }));
  fireEvent.click(screen.getByText('高级设置'));
  fireEvent.change(screen.getByRole('spinbutton', { name: '最大辩论轮数' }), { target: { value: '4' } });
  expect(JSON.parse(screen.getByTestId('value').textContent)).toEqual({
    debate: { max_rounds: 4 },
    threshold: 0.7,
    mode: 'relaxed',
    enabled: true,
  });
  expect(screen.queryByRole('textbox', { name: /JSON/i })).not.toBeInTheDocument();
});

it('associates a field error and preserves a cleared numeric draft through disclosure toggles', () => {
  function Harness() {
    const [value, setValue] = useState<RuntimeJsonObject>({ debate: { max_rounds: 3 } });
    return (
      <ParameterFields
        fields={pluginFields}
        value={value}
        onChange={setValue}
        errors={{ 'fixture.debate.max_rounds': '请填写轮数' }}
        idPrefix="fixture"
      />
    );
  }
  render(<Harness />);
  fireEvent.click(screen.getByText('高级设置'));
  const input = screen.getByRole('spinbutton', { name: '最大辩论轮数' });
  expect(input).toHaveAccessibleDescription('请填写轮数');
  expect(input).toHaveAttribute('aria-invalid', 'true');
  fireEvent.change(input, { target: { value: '' } });
  fireEvent.click(screen.getByText('高级设置'));
  fireEvent.click(screen.getByText('高级设置'));
  expect(screen.getByRole('spinbutton', { name: '最大辩论轮数' })).toHaveValue(null);
});

it('renders catalog ratio units as percentages and stores their inverse conversion', () => {
  function Harness() {
    const [value, setValue] = useState<RuntimeJsonObject>({ threshold: 0.04 });
    return (
      <>
        <ParameterFields
          fields={[{ ...pluginFields[0]!, unit: 'ratio', step: 0.01 }]}
          value={value}
          onChange={setValue}
          idPrefix="ratio"
        />
        <output data-testid="ratio">{JSON.stringify(value)}</output>
      </>
    );
  }
  render(<Harness />);
  const input = screen.getByLabelText('触发阈值（%）');
  expect(input).toHaveValue(4);
  fireEvent.change(input, { target: { value: '5' } });
  expect(JSON.parse(screen.getByTestId('ratio').textContent)).toEqual({ threshold: 0.05 });
});

function FormsHarness() {
  const [value, setValue] = useState<ConfigurationDraft<RuntimeDocument>>(
    toRuntimeDocument(runtimeConfigFixture().document),
  );
  return (
    <>
      <ModelSettings value={value.llm} onChange={(llm) => setValue({ ...value, llm })} />
      <SignalSettings
        value={value.signals}
        onChange={(signals) => setValue({ ...value, signals })}
        catalog={configurationCatalogFixture}
      />
      <MarketSettings
        value={value.market_data}
        onChange={(market_data) => setValue({ ...value, market_data })}
        catalog={configurationCatalogFixture}
      />
      <RiskSettings
        value={value.risk}
        onChange={(risk) => setValue({ ...value, risk })}
        hitl={value.hitl}
        onHitlChange={(hitl) => setValue({ ...value, hitl })}
      />
      <SchedulerSettings
        value={value.scheduler}
        onChange={(scheduler) => setValue({ ...value, scheduler })}
        triggers={value.triggers}
        onTriggersChange={(triggers) => setValue({ ...value, triggers })}
      />
      <SystemSettings value={value} onChange={(patch) => setValue({ ...value, ...patch })} />
      <output data-testid="document">{JSON.stringify(value)}</output>
    </>
  );
}

it('renders all six forms directly and keeps percentages, list edits, role models and real system settings typed', () => {
  render(<FormsHarness />);
  fireEvent.change(screen.getByLabelText('技术分析模型'), { target: { value: 'gateway/custom-model' } });
  fireEvent.change(screen.getByLabelText('最大回撤（%）'), { target: { value: '5' } });
  fireEvent.change(screen.getByLabelText('审批有效期（分钟）'), { target: { value: '45' } });
  fireEvent.change(screen.getByLabelText('交易对'), { target: { value: 'ETH/USDT' } });
  fireEvent.click(screen.getByRole('button', { name: '添加交易对' }));
  fireEvent.change(screen.getByLabelText('行情来源'), { target: { value: 'default' } });
  fireEvent.change(screen.getByLabelText('添加信号组件'), { target: { value: 'fixture-plugin' } });
  fireEvent.change(screen.getByLabelText('测试信号组件权重（%）'), { target: { value: '100' } });
  fireEvent.click(screen.getByLabelText('启用日报通知'));
  const result = JSON.parse(screen.getByTestId('document').textContent);
  expect(result.llm.models.tech_agent).toBe('gateway/custom-model');
  expect(result.risk.loss.max_drawdown_pct).toBe(0.05);
  expect(result.hitl.approval_ttl_minutes).toBe(45);
  expect(result.scheduler.pairs).toEqual(['BTC/USDT', 'ETH/USDT']);
  expect(result.signals.components[0]).toMatchObject({ component_id: 'fixture-plugin', weight: 1 });
  expect(result.market_data.source_id).toBe('default');
  expect(result.notifications.enabled).toBe(true);
  expect(screen.getByText(/重启服务后生效/)).toBeInTheDocument();
  expect(screen.queryByLabelText(/Telegram|JSON|每日亏损|冷却|图片/)).not.toBeInTheDocument();
});

it('keeps a model cost row focused while its editable name changes', () => {
  render(<FormsHarness />);
  fireEvent.click(screen.getByText('模型高级设置'));
  fireEvent.click(screen.getByRole('button', { name: '添加模型价格' }));
  const name = screen.getByLabelText('模型名称 1');
  name.focus();
  fireEvent.change(name, { target: { value: 'custom-one' } });
  expect(screen.getByLabelText('模型名称 1')).toHaveFocus();
  fireEvent.change(screen.getByLabelText('输入价格 1（美元 / 百万 Token）'), { target: { value: '2.5' } });
  expect(JSON.parse(screen.getByTestId('document').textContent).llm.model_costs[0]).toEqual({
    name: 'custom-one',
    input_usd_per_mtok: 2.5,
    output_usd_per_mtok: 0,
  });
});

it('writes a gateway key outside the ordinary document and clears its input', async () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  vi.stubGlobal(
    'fetch',
    vi
      .fn()
      .mockResolvedValue(
        new Response(JSON.stringify({ revision: 2, configured: true, updated_at: '2026-08-30T00:00:00Z' }), {
          status: 200,
        }),
      ),
  );
  const llm = toRuntimeDocument(runtimeConfigFixture().document).llm;
  render(
    <QueryClientProvider client={client}>
      <ModelSettings
        value={llm}
        onChange={() => undefined}
        secrets={{ revision: 1, configured: false, updatedAt: null }}
      />
    </QueryClientProvider>,
  );
  fireEvent.change(screen.getByLabelText('LLM 网关密钥'), { target: { value: 'fixture-only-key' } });
  fireEvent.click(screen.getByRole('button', { name: '保存网关密钥' }));
  await waitFor(() => expect(screen.getByLabelText('LLM 网关密钥')).toHaveValue(''));
  expect(JSON.stringify(client.getMutationCache().getAll())).not.toContain('fixture-only-key');
});

it('uses English labels when the current locale is English', async () => {
  await i18n.changeLanguage('en-US');
  render(<FormsHarness />);
  expect(screen.getByLabelText('Maximum drawdown (%)')).toBeInTheDocument();
  expect(screen.getByLabelText('Technical analysis model')).toBeInTheDocument();
  await i18n.changeLanguage('zh-CN');
});

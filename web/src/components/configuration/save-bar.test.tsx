import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { expect, it, vi } from 'vitest';
import '@/lib/i18n';
import { SaveBar } from './save-bar';
import { ModelSettings } from '@/pages/settings/forms/model-settings';
import { useConfigurationDraft } from '@/hooks/use-configuration-draft';
import { RUNTIME_CONFIG_QUERY_KEY } from '@/hooks/use-runtime-config';
import { runtimeConfigFixture } from '@/test/runtime-config-fixture';
import { SignalSettings } from '@/pages/settings/forms/signal-settings';
import { configurationCatalogFixture } from '@/test/configuration-catalog-fixture';

it('shows an active retry instead of the previous save failure', () => {
  const view = render(
    <SaveBar dirty status="failed" failure="保存失败" onSave={() => undefined} onDiscard={() => undefined} />,
  );
  expect(screen.getByRole('status')).toHaveTextContent('保存失败');
  view.rerender(
    <SaveBar dirty status="saving" failure="保存失败" onSave={() => undefined} onDiscard={() => undefined} />,
  );
  expect(screen.getByRole('status')).toHaveTextContent('保存中');
  expect(screen.getByRole('status')).not.toHaveTextContent('保存失败');
});

it('submits the owning configuration form from the keyboard', async () => {
  const saved = vi.fn();
  render(
    <form
      onSubmit={(event) => {
        event.preventDefault();
        saved();
      }}
    >
      <label>
        周期
        <input />
      </label>
      <SaveBar dirty submit onSave={() => undefined} onDiscard={() => undefined} />
    </form>,
  );

  await userEvent.type(screen.getByLabelText('周期'), '2h{Enter}');
  expect(saved).toHaveBeenCalledOnce();
});

it('opens an advanced group and focuses the first invalid field on save', () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
  client.setQueryData(RUNTIME_CONFIG_QUERY_KEY, runtimeConfigFixture());
  function Harness() {
    const draft = useConfigurationDraft();
    return (
      <form
        noValidate
        onSubmit={(event) => {
          event.preventDefault();
          void draft.save('models', event.currentTarget);
        }}
      >
        <ModelSettings value={draft.document!.llm} errors={draft.errors} onChange={(llm) => draft.update('llm', llm)} />
        <button type="submit">Save models</button>
      </form>
    );
  }
  render(
    <QueryClientProvider client={client}>
      <Harness />
    </QueryClientProvider>,
  );
  fireEvent.click(screen.getByText('模型高级设置'));
  fireEvent.change(screen.getByLabelText('默认请求超时（秒）'), { target: { value: '' } });
  fireEvent.click(screen.getByText('模型高级设置'));
  fireEvent.click(screen.getByRole('button', { name: 'Save models' }));
  const input = screen.getByLabelText('默认请求超时（秒）');
  expect(input).toHaveFocus();
  expect(input).toHaveAttribute('aria-invalid', 'true');
  expect(input).toHaveValue(null);
  expect(input).toHaveAccessibleDescription('请输入允许范围内的整数。');
});

it('requires confirmation before discarding and distinguishes persistence from application', () => {
  const discard = vi.fn();
  const confirm = vi.spyOn(window, 'confirm').mockReturnValueOnce(false).mockReturnValueOnce(true);
  const view = render(<SaveBar dirty onSave={() => undefined} onDiscard={discard} />);
  fireEvent.click(screen.getByRole('button', { name: '放弃修改' }));
  expect(discard).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button', { name: '放弃修改' }));
  expect(discard).toHaveBeenCalledOnce();
  view.rerender(
    <SaveBar dirty={false} status="saved" applyStatus="pending" onSave={() => undefined} onDiscard={discard} />,
  );
  expect(screen.getByRole('status')).toHaveTextContent('配置已保存，尚未实际生效。');
  expect(screen.getByRole('button', { name: '保存配置' })).toBeDisabled();
  confirm.mockRestore();
});

it('focuses an enabled weight when the signal total is invalid', () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
  const base = runtimeConfigFixture();
  client.setQueryData(
    RUNTIME_CONFIG_QUERY_KEY,
    runtimeConfigFixture({
      document: {
        ...base.document,
        signals: {
          ...base.document.signals,
          components: [{ component_id: 'fixture-plugin', enabled: true, weight: 0.4, parameters: [] }],
        },
      },
    }),
  );
  function Harness() {
    const draft = useConfigurationDraft(configurationCatalogFixture);
    return (
      <form
        noValidate
        onSubmit={(event) => {
          event.preventDefault();
          void draft.save('signals', event.currentTarget);
        }}
      >
        <SignalSettings
          value={draft.document!.signals}
          onChange={(signals) => draft.update('signals', signals)}
          errors={draft.errors}
          catalog={configurationCatalogFixture}
        />
        <button type="submit">Save signals</button>
      </form>
    );
  }
  render(
    <QueryClientProvider client={client}>
      <Harness />
    </QueryClientProvider>,
  );
  fireEvent.click(screen.getByRole('button', { name: 'Save signals' }));
  const weight = screen.getByLabelText('测试信号组件权重（%）');
  expect(weight).toHaveFocus();
  expect(weight).toHaveAttribute('aria-invalid', 'true');
});

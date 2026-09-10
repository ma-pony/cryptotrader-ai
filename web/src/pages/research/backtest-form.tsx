import { useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import {
  ChoiceField,
  Field,
  NumberField,
  TextField,
  focusFirstError,
  type FieldErrors,
} from '@/components/configuration/field';
import { useBacktestRuns, useLoadBacktestRun, useStartBacktest } from '@/hooks/use-backtest';
import { useRuntimeConfig } from '@/hooks/use-runtime-config';
import type { BacktestParams, BacktestRunStatus } from '@/types/api';
import { SnapshotSummary } from './presentation';

const defaults: BacktestParams = {
  pair: 'BTC/USDT',
  start: '',
  end: '',
  interval: '1h',
  initial_equity: '10000',
  fee_rate: '0.001',
  slippage_bps: '0',
  funding_assumption: 'available_only',
  name: null,
  snapshot_run_id: null,
};

export function BacktestForm({
  onRunStarted,
  initialRun,
}: {
  onRunStarted: (id: string) => void;
  initialRun?: BacktestRunStatus | undefined;
}) {
  const runtime = useRuntimeConfig();
  const history = useBacktestRuns();
  const load = useLoadBacktestRun();
  const start = useStartBacktest();
  const form = useRef<HTMLFormElement>(null);
  const [values, setValues] = useState<BacktestParams>(() =>
    initialRun ? { ...initialRun.params, name: null, snapshot_run_id: initialRun.run_id } : defaults,
  );
  const [selected, setSelected] = useState(initialRun?.run_id ?? '');
  const [selectedRun, setSelectedRun] = useState(initialRun);
  const [errors, setErrors] = useState<FieldErrors>({});
  const today = new Date().toISOString().slice(0, 10);
  const busy = start.isPending || load.isPending;
  const snapshotReady = selected
    ? selectedRun?.run_id === selected && selectedRun.config_snapshot !== null
    : !!runtime.document;
  const update = (key: keyof BacktestParams, value: string | null) => {
    setValues((old) => ({ ...old, [key]: value }));
    start.reset();
  };
  const number = (value: string | null) => (value === null || value === '' ? '' : Number(value));
  const items = history.data?.items ?? [];
  const options =
    initialRun && !items.some((item) => item.run_id === initialRun.run_id) ? [initialRun, ...items] : items;

  return (
    <form
      ref={form}
      noValidate
      className="space-y-4 rounded-lg border border-border bg-card p-4"
      onSubmit={(event) => {
        event.preventDefault();
        if (busy) return;
        const next: FieldErrors = {};
        if (!values.start) next.start = '请填写起始日期';
        if (!values.end || values.start >= values.end) next.end = '结束日期须晚于起始日期';
        else if (values.end > today) next.end = '结束日期不能晚于今天';
        if (!values.pair.trim()) next.pair = '请填写交易对';
        if (!values.initial_equity || Number(values.initial_equity) < 100) next.initial_equity = '初始资金至少为100';
        if (
          values.fee_rate === null ||
          values.fee_rate === '' ||
          Number(values.fee_rate) < 0 ||
          Number(values.fee_rate) >= 1
        )
          next.fee_rate = '手续费须为0%至100%之间（不含100%）';
        if (
          values.slippage_bps === null ||
          values.slippage_bps === '' ||
          Number(values.slippage_bps) < 0 ||
          Number(values.slippage_bps) >= 10000
        )
          next.slippage_bps = '滑点须为0至10000 bps之间（不含10000）';
        if (!values.interval) next.interval = '请选择运行周期';
        setErrors(next);
        if (Object.keys(next).length) {
          focusFirstError(next, form.current ?? undefined);
          return;
        }
        if (!snapshotReady) return;
        start.mutate(
          { ...values, snapshot_run_id: selected || null },
          { onSuccess: (run) => onRunStarted(run.run_id) },
        );
      }}
    >
      <h2 className="font-semibold">新建回测实验</h2>
      <Field
        name="snapshot"
        label="配置来源与历史复用"
        help="使用当前已保存配置，或复用历史完整快照；选择不会启动回测。"
      >
        <select
          id="snapshot"
          className="configuration-control"
          value={selected}
          disabled={busy}
          onChange={(event) => {
            const id = event.target.value;
            setSelected(id);
            start.reset();
            if (!id) {
              setSelectedRun(undefined);
              load.reset();
              return;
            }
            load.mutate(id, {
              onSuccess: (run) => {
                setSelectedRun(run);
                setValues({ ...run.params, name: null, snapshot_run_id: run.run_id });
                setErrors({});
              },
            });
          }}
        >
          <option value="">当前已保存配置{runtime.revision !== undefined ? ` · R${runtime.revision}` : ''}</option>
          {options.map((run) => (
            <option key={run.run_id} value={run.run_id} disabled={run.config_snapshot === null}>
              {run.params.name ?? run.params.pair} · {run.params.start} · {run.run_id}
              {run.config_snapshot !== null ? '' : '（快照缺失，不可复用）'}
            </option>
          ))}
        </select>
      </Field>
      {selectedRun && selectedRun.run_id === selected ? (
        <details>
          <summary className="min-h-10 cursor-pointer">查看将使用的历史配置</summary>
          <SnapshotSummary run={selectedRun} />
        </details>
      ) : (
        <p>
          使用已保存版本；未保存草稿不参与。本次原行情源：{runtime.document?.market_data.source_id ?? '读取中'}，组件：
          {runtime.document?.signals.components
            .filter((item) => item.enabled)
            .map((item) => `${item.component_id}（权重${item.weight}）`)
            .join('、') ?? '读取中'}
          。
        </p>
      )}
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <TextField
          name="pair"
          label="币对"
          value={values.pair}
          onChange={(value) => update('pair', value)}
          disabled={busy}
          error={errors.pair}
        />
        {(['start', 'end'] as const).map((key) => (
          <Field key={key} name={key} label={key === 'start' ? '起始日期' : '结束日期'} error={errors[key]}>
            <input
              className="configuration-control"
              id={key}
              name={key}
              type="date"
              max={today}
              value={values[key]}
              disabled={busy}
              aria-invalid={!!errors[key]}
              aria-describedby={`${key}-help`}
              onChange={(event) => update(key, event.target.value)}
            />
          </Field>
        ))}
        <ChoiceField
          name="interval"
          label="运行周期"
          value={values.interval ?? ''}
          options={['1m', '5m', '15m', '1h', '4h', '1d'].map((value) => ({ value, label: value }))}
          onChange={(value) => update('interval', value)}
          error={errors.interval}
          disabled={busy}
        />
        <NumberField
          name="initial_equity"
          label="初始资金（USDT）"
          min={100}
          value={number(values.initial_equity)}
          onChange={(value) => update('initial_equity', String(value))}
          error={errors.initial_equity}
          disabled={busy}
        />
        <NumberField
          name="fee_rate"
          label="手续费（%）"
          help="每笔实际成交额计费；0.1%表示费率0.001。"
          percent
          min={0}
          max={99.99}
          value={number(values.fee_rate)}
          onChange={(value) => update('fee_rate', String(value))}
          error={errors.fee_rate}
          disabled={busy}
        />
        <NumberField
          name="slippage_bps"
          label="滑点（bps）"
          help="1 bps = 0.01%；买入加价，卖出减价。"
          min={0}
          max={9999}
          value={number(values.slippage_bps)}
          onChange={(value) => update('slippage_bps', String(value))}
          error={errors.slippage_bps}
          disabled={busy}
        />
        <ChoiceField
          name="funding_assumption"
          label="资金费处理"
          value={values.funding_assumption ?? ''}
          options={[
            { value: 'available_only', label: '仅已提供的历史结算点' },
            { value: 'disabled', label: '明确不计资金费' },
          ]}
          onChange={(value) => update('funding_assumption', value)}
          disabled={busy}
        />
      </div>
      <TextField
        name="name"
        label="实验名称（可选）"
        value={values.name ?? ''}
        onChange={(value) => update('name', value || null)}
        disabled={busy}
      />
      <aside className="space-y-2 rounded border border-border p-3" aria-label="启动前说明">
        <p>启动后会读取所选源的历史已收盘 K 线。实际数据覆盖在结果中列出；目前未读取，不能预先保证完整。</p>
        <p>
          缺项：历史新闻、盘口与部分宏观数据不可用；没有提供的资金费结算点不估造。OHLCV 模拟不涵盖市场冲击及 bar
          内真实路径。
        </p>
        <p>
          模型费用：启用 LLM
          组件会产生模型请求费用，长区间可能较高；未知价格不视为免费。预训练模型可能含历史截止时间之后的信息。
        </p>
        <p>仅使用本地模拟账本，不读取交易账户、不发送真实或官方模拟订单。无名称运行也会保存。</p>
      </aside>
      {Object.keys(errors).length ? (
        <p role="alert" className="configuration-error">
          {Object.values(errors)[0]}
        </p>
      ) : null}
      {start.isError ? (
        <p role="alert" className="configuration-error">
          回测启动失败，输入已保留。请核对配置与访问权限后重试。
        </p>
      ) : null}
      {load.isError ? <p role="alert">历史配置读取失败，原输入已保留；请重新选择或使用当前配置。</p> : null}
      {history.isError ? (
        <p role="alert">
          历史列表读取失败。
          <button type="button" className="configuration-button" onClick={() => void history.refetch()}>
            重新读取历史
          </button>
        </p>
      ) : null}
      {runtime.isError && !selected ? <p role="alert">当前配置读取失败，请重新加载配置。</p> : null}
      {load.isPending ? <p role="status">正在读取历史快照…</p> : null}
      <Button className="min-h-10" type="submit" disabled={busy || !snapshotReady}>
        {start.isPending ? '正在排入任务…' : '运行回测'}
      </Button>
    </form>
  );
}

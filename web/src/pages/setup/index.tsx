import { ArrowRight, CheckCircle2, Plus } from 'lucide-react';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';

import { Button } from '@/components/ui/button';
import { decodeEntries, useRuntimeConfig } from '@/hooks/use-runtime-config';
import type { RuntimeConfig, RuntimeDocument, RuntimeJsonObject } from '@/types/api';
import { BookForm, newBook, validateBooks } from '@/pages/settings/execution-books/book-form';
import { VenueForm } from '@/pages/settings/venues/venue-form';

const STEPS = ['LLM', '信号组件', '行情来源', '平台连接', '执行资金池', '风控与审批', '调度器', '测试并激活'];
type DraftConnection = RuntimeDocument['execution']['connections'][number];
type ResponseConnection = RuntimeConfig['document']['execution']['connections'][number];

const testFingerprint = (connection: DraftConnection, credentialUpdatedAt?: string | null) =>
  JSON.stringify({
    id: connection.id,
    label: connection.label,
    adapter: connection.adapter_id,
    environment: connection.environment,
    enabled: connection.enabled,
    leverage: connection.leverage,
    margin: connection.margin_mode,
    parameters: connection.parameters,
    credentialUpdatedAt: credentialUpdatedAt ?? null,
  });

const asDraftConnection = (connection: ResponseConnection): DraftConnection => {
  const { credential_configured: _configured, credential_updated_at: _updatedAt, parameters, ...rest } = connection;
  return { ...rest, parameters: decodeEntries(parameters) } as DraftConnection;
};

const SetupEditor = ({
  initialDocument,
  onReload,
}: {
  initialDocument: RuntimeDocument;
  onReload: () => Promise<void>;
}) => {
  const { t } = useTranslation('configuration');
  const runtime = useRuntimeConfig();
  const [step, setStep] = useState(0);
  const [draft, setDraft] = useState(initialDocument);
  const [tested, setTested] = useState<Record<string, string>>({});
  const [marketParameters, setMarketParameters] = useState(
    JSON.stringify(initialDocument.market_data.parameters, null, 2),
  );
  const [marketError, setMarketError] = useState('');
  const [marketDirty, setMarketDirty] = useState(false);
  const [customId, setCustomId] = useState('');
  const [customParameters, setCustomParameters] = useState('{}');
  const [signalError, setSignalError] = useState('');
  const [riskText, setRiskText] = useState(JSON.stringify(initialDocument.risk, null, 2));
  const [riskError, setRiskError] = useState('');
  const [riskDirty, setRiskDirty] = useState(false);
  const [llmAdvanced, setLlmAdvanced] = useState(JSON.stringify({ streaming_models: initialDocument.llm.streaming_models, retry: initialDocument.llm.retry, model_costs: initialDocument.llm.model_costs, timeout_seconds: initialDocument.llm.models.timeout_seconds }, null, 2));
  const [llmError, setLlmError] = useState('');
  const [activationError, setActivationError] = useState('');

  const enabledConnections = draft.execution.connections.filter((connection) => connection.enabled);
  const bookErrors = validateBooks(draft.execution.books, draft.execution.connections);
  const enabled = draft.signals.components.filter((component) => component.enabled);
  const signalValid =
    enabled.length > 0 &&
    Math.abs(enabled.reduce((sum, component) => sum + component.weight, 0) - 1) < 1e-9 &&
    draft.signals.neutral_threshold >= 0 &&
    draft.signals.neutral_threshold < 1 &&
    draft.signals.max_target_ratio > 0 &&
    draft.signals.max_target_ratio <= 1 &&
    draft.signals.atr_stop_multiplier > 0 &&
    draft.signals.reward_ratio > 0;
  const connectionsTested =
    enabledConnections.length > 0 &&
    enabledConnections.every(
      (connection) =>
        tested[connection.id] === testFingerprint(connection, runtime.credentialStates[connection.id]?.updatedAt),
    );
  const ready =
    signalValid &&
    draft.execution.books.some((book) => book.enabled) &&
    bookErrors.length === 0 &&
    connectionsTested &&
    !marketDirty &&
    !marketError &&
    !riskDirty &&
    !riskError;

  const applyMarketParameters = () => {
    try {
      const parameters = JSON.parse(marketParameters) as RuntimeJsonObject;
      if (!parameters || Array.isArray(parameters) || typeof parameters !== 'object')
        throw new Error('object expected');
      setDraft((current) => ({ ...current, market_data: { ...current.market_data, parameters } }));
      setMarketError('');
      setMarketDirty(false);
    } catch {
      setMarketError('行情参数必须是 JSON 对象。');
    }
  };
  const activate = async () => {
    if (!ready) return;
    try {
      setActivationError('');
      await runtime.replace({ ...draft, system: { ...draft.system, active: true } });
    } catch {
      setActivationError('激活失败，请重新加载后重试。');
    }
  };
  const replaceConnection = (saved: ResponseConnection) =>
    setDraft((current) => ({
      ...current,
      execution: {
        ...current.execution,
        connections: current.execution.connections.some((connection) => connection.id === saved.id)
          ? current.execution.connections.map((connection) =>
              connection.id === saved.id ? asDraftConnection(saved) : connection,
            )
          : [...current.execution.connections, asDraftConnection(saved)],
      },
    }));

  const content = (() => {
    if (step === 0) {
      const keys = [
        'analysis',
        'fallback',
        'tech_agent',
        'chain_agent',
        'news_agent',
        'macro_agent',
        'debate',
        'committee_summary',
      ] as const;
      return (
        <div className="grid gap-3 md:grid-cols-2">
          {keys.map((key) => (
            <label key={key} className="text-sm text-muted-foreground">
              {key}
              <input
                aria-label={`LLM ${key}`}
                value={draft.llm.models[key]}
                onChange={(event) =>
                  setDraft((current) => ({
                    ...current,
                    llm: { ...current.llm, models: { ...current.llm.models, [key]: event.target.value } },
                  }))
                }
                className="mt-1 h-10 w-full rounded border bg-background px-3 font-mono"
              />
            </label>
          ))}
          <label>Base URL<input aria-label="LLM base URL" value={draft.llm.base_url} onChange={(event) => setDraft((current) => ({ ...current, llm: { ...current.llm, base_url: event.target.value } }))} className="mt-1 h-10 w-full rounded border bg-background px-3" /></label>
          <label>默认温度<input aria-label="LLM 默认温度" type="number" value={draft.llm.default_temperature} onChange={(event) => setDraft((current) => ({ ...current, llm: { ...current.llm, default_temperature: Number(event.target.value) } }))} className="mt-1 h-10 w-full rounded border bg-background px-3" /></label>
          <label>超时<input aria-label="LLM 超时" type="number" value={draft.llm.timeout} onChange={(event) => setDraft((current) => ({ ...current, llm: { ...current.llm, timeout: Number(event.target.value) } }))} className="mt-1 h-10 w-full rounded border bg-background px-3" /></label>
          <label className="flex items-center gap-2"><input aria-label="LLM prompt caching" type="checkbox" checked={draft.llm.prompt_caching} onChange={(event) => setDraft((current) => ({ ...current, llm: { ...current.llm, prompt_caching: event.target.checked } }))} />Prompt caching</label>
          <label className="md:col-span-2">Streaming / retry / model costs JSON<textarea aria-label="LLM 高级 JSON" value={llmAdvanced} onChange={(event) => setLlmAdvanced(event.target.value)} className="mt-1 min-h-32 w-full rounded border bg-background p-3 font-mono text-xs" /></label>
          <Button type="button" variant="outline" onClick={() => { try { const value = JSON.parse(llmAdvanced) as { streaming_models: string[]; retry: RuntimeDocument['llm']['retry']; model_costs: RuntimeDocument['llm']['model_costs']; timeout_seconds: number }; if (!Array.isArray(value.streaming_models) || !value.retry || !Array.isArray(value.model_costs) || !Number.isInteger(value.timeout_seconds)) throw new Error(); setDraft((current) => ({ ...current, llm: { ...current.llm, streaming_models: value.streaming_models, retry: value.retry, model_costs: value.model_costs, models: { ...current.llm.models, timeout_seconds: value.timeout_seconds } } })); setLlmError(''); } catch { setLlmError('LLM 高级配置格式无效。'); } }}>应用高级配置</Button>
          {llmError ? <p role="alert" className="text-sm text-trade-short">{llmError}</p> : null}
        </div>
      );
    }
    if (step === 1)
      return (
        <div className="space-y-3">
          <p className="text-sm text-muted-foreground">Kronos、LLM 委员会及自定义组件的权重总和必须为 100%。</p>
          {draft.signals.components.map((component, index) => (
            <div key={component.component_id} className="grid grid-cols-[1fr_auto_auto] items-center gap-3">
              <span>{component.component_id}</span>
              <input
                aria-label={`${component.component_id} 启用`}
                type="checkbox"
                checked={component.enabled}
                onChange={(event) =>
                  setDraft((current) => ({
                    ...current,
                    signals: {
                      ...current.signals,
                      components: current.signals.components.map((item, itemIndex) =>
                        itemIndex === index ? { ...item, enabled: event.target.checked } : item,
                      ),
                    },
                  }))
                }
              />
              <input
                aria-label={`${component.component_id} 权重`}
                className="h-9 w-24 rounded border bg-background px-2"
                type="number"
                value={component.weight * 100}
                onChange={(event) =>
                  setDraft((current) => ({
                    ...current,
                    signals: {
                      ...current.signals,
                      components: current.signals.components.map((item, itemIndex) =>
                        itemIndex === index ? { ...item, weight: Number(event.target.value) / 100 } : item,
                      ),
                    },
                  }))
                }
              />
            </div>
          ))}
          <div className="grid gap-2 md:grid-cols-[1fr_2fr_auto]">
            <input aria-label="自定义 component ID" value={customId} onChange={(event) => setCustomId(event.target.value)} placeholder="自定义 component ID" className="h-10 rounded border bg-background px-3" />
            <input aria-label="自定义 component 参数" value={customParameters} onChange={(event) => setCustomParameters(event.target.value)} placeholder="{}" className="h-10 rounded border bg-background px-3 font-mono" />
            <Button type="button" variant="outline" onClick={() => {
              try {
                const id = customId.trim(); const parameters = JSON.parse(customParameters) as RuntimeJsonObject;
                if (!id || draft.signals.components.some((component) => component.component_id === id) || !parameters || Array.isArray(parameters)) throw new Error();
                setDraft((current) => ({ ...current, signals: { ...current.signals, components: [...current.signals.components, { component_id: id, enabled: false, weight: 0, parameters }] } }));
                setCustomId(''); setCustomParameters('{}'); setSignalError('');
              } catch { setSignalError('自定义组件 ID 必须唯一，参数必须是 JSON 对象。'); }
            }}><Plus className="h-4 w-4" />添加自定义组件</Button>
          </div>
          {signalError ? <p role="alert" className="text-sm text-trade-short">{signalError}</p> : null}
          <label>
            中性阈值
            <input
              aria-label="中性阈值"
              type="number"
              value={draft.signals.neutral_threshold}
              onChange={(event) =>
                setDraft((current) => ({
                  ...current,
                  signals: { ...current.signals, neutral_threshold: Number(event.target.value) },
                }))
              }
              className="ml-2 h-9 rounded border bg-background px-2"
            />
          </label>
          <label>最大目标比例<input aria-label="最大目标比例" type="number" value={draft.signals.max_target_ratio} onChange={(event) => setDraft((current) => ({ ...current, signals: { ...current.signals, max_target_ratio: Number(event.target.value) } }))} className="ml-2 h-9 rounded border bg-background px-2" /></label>
          <label>ATR 止损<input aria-label="ATR 止损" type="number" value={draft.signals.atr_stop_multiplier} onChange={(event) => setDraft((current) => ({ ...current, signals: { ...current.signals, atr_stop_multiplier: Number(event.target.value) } }))} className="ml-2 h-9 rounded border bg-background px-2" /></label>
          <label>盈亏比<input aria-label="盈亏比" type="number" value={draft.signals.reward_ratio} onChange={(event) => setDraft((current) => ({ ...current, signals: { ...current.signals, reward_ratio: Number(event.target.value) } }))} className="ml-2 h-9 rounded border bg-background px-2" /></label>
          <label className="flex items-center gap-2"><input aria-label="信号 HITL" type="checkbox" checked={draft.signals.hitl_required} onChange={(event) => setDraft((current) => ({ ...current, signals: { ...current.signals, hitl_required: event.target.checked } }))} />信号 HITL</label>
        </div>
      );
    if (step === 2)
      return (
        <div className="space-y-3">
          <label className="block text-sm">
            行情 source ID
            <input
              aria-label="行情 source ID"
              value={draft.market_data.source_id}
              onChange={(event) =>
                setDraft((current) => ({
                  ...current,
                  market_data: { ...current.market_data, source_id: event.target.value },
                }))
              }
              className="mt-1 h-10 w-full rounded border bg-background px-3"
            />
          </label>
          <label className="block text-sm">
            JSON 参数
            <textarea
              aria-label="行情 JSON 参数"
              value={marketParameters}
              onChange={(event) => { setMarketParameters(event.target.value); setMarketDirty(true); }}
              className="mt-1 min-h-32 w-full rounded border bg-background p-3 font-mono text-xs"
            />
          </label>
          <Button type="button" variant="outline" onClick={applyMarketParameters}>
            应用行情参数
          </Button>
          {marketError ? (
            <p role="alert" className="text-sm text-trade-short">
              {marketError}
            </p>
          ) : null}
        </div>
      );
    if (step === 3)
      return (
        <div className="space-y-4">
          <p className="text-sm text-muted-foreground">
            保存连接后会合并进当前草稿；测试只对当前连接 fingerprint 有效。
          </p>
          {draft.execution.connections.map((connection) => (
            <VenueForm
              key={connection.id}
              revision={runtime.revision ?? 0}
              connection={connection}
              onSaved={replaceConnection}
              tested={(id) => {
                const current = draft.execution.connections.find((item) => item.id === id);
                if (current)
                  setTested((old) => ({
                    ...old,
                    [id]: testFingerprint(current, runtime.credentialStates[id]?.updatedAt),
                  }));
              }}
            />
          ))}
          <VenueForm revision={runtime.revision ?? 0} onSaved={replaceConnection} />
        </div>
      );
    if (step === 4)
      return (
        <div className="space-y-3">
          {draft.execution.books.map((book, index) => (
            <BookForm
              key={`${book.id}-${index}`}
              book={book}
              connections={draft.execution.connections}
              onChange={(next) =>
                setDraft((current) => ({
                  ...current,
                  execution: {
                    ...current.execution,
                    books: current.execution.books.map((item, itemIndex) => (itemIndex === index ? next : item)),
                  },
                }))
              }
              onRemove={() =>
                setDraft((current) => ({
                  ...current,
                  execution: {
                    ...current.execution,
                    books: current.execution.books.filter((_, itemIndex) => itemIndex !== index),
                  },
                }))
              }
            />
          ))}
          <Button
            type="button"
            variant="outline"
            onClick={() =>
              setDraft((current) => ({
                ...current,
                execution: { ...current.execution, books: [...current.execution.books, newBook()] },
              }))
            }
          >
            新增资金池
          </Button>
          {bookErrors.map((error) => (
            <p key={error} role="alert" className="text-sm text-trade-short">
              {error}
            </p>
          ))}
        </div>
      );
    if (step === 5)
      return (
        <div className="grid gap-3 md:grid-cols-2">
          <label>
            最大止损比例
            <input
              aria-label="最大止损比例"
              type="number"
              value={draft.risk.max_stop_loss_pct}
              onChange={(event) =>
                setDraft((current) => ({
                  ...current,
                  risk: { ...current.risk, max_stop_loss_pct: Number(event.target.value) },
                }))
              }
              className="ml-2 h-9 rounded border bg-background px-2"
            />
          </label>
          <label>
            审批 TTL（分钟）
            <input
              aria-label="审批 TTL"
              type="number"
              value={draft.hitl.approval_ttl_minutes}
              onChange={(event) =>
                setDraft((current) => ({
                  ...current,
                  hitl: { ...current.hitl, approval_ttl_minutes: Number(event.target.value) },
                }))
              }
              className="ml-2 h-9 rounded border bg-background px-2"
            />
          </label>
          <label className="md:col-span-2">完整风控 JSON（position/loss/cooldown/volatility/exchange/rate_limit）<textarea aria-label="完整风控 JSON" value={riskText} onChange={(event) => { setRiskText(event.target.value); setRiskDirty(true); }} className="mt-1 min-h-36 w-full rounded border bg-background p-3 font-mono text-xs" /></label>
          <Button type="button" variant="outline" onClick={() => { try { const risk = JSON.parse(riskText) as RuntimeDocument['risk']; if (!risk || Array.isArray(risk)) throw new Error(); setDraft((current) => ({ ...current, risk })); setRiskError(''); setRiskDirty(false); } catch { setRiskError('风控配置必须是 JSON 对象。'); } }}>应用完整风控配置</Button>
          {riskError ? <p role="alert" className="text-sm text-trade-short">{riskError}</p> : null}
        </div>
      );
    if (step === 6)
      return (
        <div className="grid gap-3 md:grid-cols-2">
          <label className="flex items-center gap-2">
            <input
              aria-label="启用调度器"
              type="checkbox"
              checked={draft.scheduler.enabled}
              onChange={(event) =>
                setDraft((current) => ({
                  ...current,
                  scheduler: { ...current.scheduler, enabled: event.target.checked },
                }))
              }
            />
            启用调度器
          </label>
          <label>
            间隔（分钟）
            <input
              aria-label="调度间隔"
              type="number"
              value={draft.scheduler.interval_minutes}
              onChange={(event) =>
                setDraft((current) => ({
                  ...current,
                  scheduler: { ...current.scheduler, interval_minutes: Number(event.target.value) },
                }))
              }
              className="ml-2 h-9 rounded border bg-background px-2"
            />
          </label>
          <label>交易对（逗号分隔）<input aria-label="调度交易对" value={draft.scheduler.pairs.join(',')} onChange={(event) => setDraft((current) => ({ ...current, scheduler: { ...current.scheduler, pairs: event.target.value.split(',').map((pair) => pair.trim()).filter(Boolean) } }))} className="ml-2 h-9 rounded border bg-background px-2" /></label>
          <label>日报小时<input aria-label="日报小时" type="number" value={draft.scheduler.daily_summary_hour} onChange={(event) => setDraft((current) => ({ ...current, scheduler: { ...current.scheduler, daily_summary_hour: Number(event.target.value) } }))} className="ml-2 h-9 rounded border bg-background px-2" /></label>
        </div>
      );
    return (
      <div className="space-y-3">
        <p className={ready ? 'text-trade-long' : 'text-trade-short'}>
          {ready
            ? '所有激活条件已满足。'
            : '需有启用组件、合法 100% 资金池，以及至少一个已启用并在本向导测试成功的连接。'}
        </p>
        <Button disabled={!ready || runtime.isSaving} onClick={() => void activate()}>
          <CheckCircle2 className="h-4 w-4" />
          测试并激活
        </Button>
        {activationError ? (
          <p role="alert" className="text-sm text-trade-short">
            {activationError}
          </p>
        ) : null}
      </div>
    );
  })();

  return (
    <main className="min-h-screen bg-background p-6 text-foreground">
      <div className="mx-auto max-w-5xl">
        <header className="border-b border-amber-500/30 pb-6">
          <p className="font-mono text-xs tracking-[.24em] text-amber-500">COMMISSIONING / REV {runtime.revision}</p>
          <h1 className="mt-3 text-3xl font-semibold">{t('commissioning')}</h1>
          <p className="mt-2 text-muted-foreground">依次完成八个 commissioning 阶段；激活后才进入操作台。</p>
        </header>
        <div className="mt-8 grid gap-6 lg:grid-cols-[230px_1fr]">
          <ol className="border-l border-amber-500/30">
            {STEPS.map((label, index) => (
              <li
                key={label}
                className={`relative py-3 pl-5 text-sm ${index === step ? 'font-semibold text-amber-500' : index < step ? 'text-trade-long' : 'text-muted-foreground'}`}
              >
                <span className="absolute -left-1.5 top-4 h-3 w-3 rounded-full bg-current" />
                {index + 1}. {label}
              </li>
            ))}
          </ol>
          <section className="rounded-2xl border border-border bg-card p-6">
            <p className="font-mono text-xs text-amber-500">STAGE 0{step + 1}</p>
            <h2 className="mt-2 text-xl font-semibold">{STEPS[step]}</h2>
            <div className="mt-4">{content}</div>
            {step < 7 ? (
              <Button className="mt-6" onClick={() => setStep((current) => Math.min(current + 1, 7))}>
                下一阶段 <ArrowRight className="h-4 w-4" />
              </Button>
            ) : null}
            {runtime.conflict ? (
              <div className="mt-4 flex gap-2">
                <p role="alert" className="text-sm text-trade-short">
                  配置已被其他操作更新，请重新加载
                </p>
                <Button size="sm" variant="outline" onClick={() => void onReload()}>
                  重新加载
                </Button>
              </div>
            ) : null}
          </section>
        </div>
      </div>
    </main>
  );
};

const SetupPage = () => {
  const { t } = useTranslation('configuration');
  const runtime = useRuntimeConfig();
  const [editorVersion, setEditorVersion] = useState(0);
  const reloadEditor = async () => {
    const result = await runtime.reload();
    if (result.isSuccess && !result.error) setEditorVersion((version) => version + 1);
  };
  if (runtime.isLoading)
    return <div className="grid min-h-screen place-items-center text-amber-500">LOADING CONFIG…</div>;
  if (runtime.isError || !runtime.document || runtime.revision === undefined)
    return (
      <main className="grid min-h-screen place-items-center">
        <div>
          <h1>{t('loadError')}</h1>
          <Button onClick={() => void reloadEditor()}>{t('retry')}</Button>
        </div>
      </main>
    );
  return <SetupEditor key={editorVersion} initialDocument={runtime.document} onReload={reloadEditor} />;
};

export default SetupPage;

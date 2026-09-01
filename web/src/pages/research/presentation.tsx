import { Link } from 'react-router';
import type { BacktestRunStatus } from '@/types/api';

export const runLabels: Record<BacktestRunStatus['status'], string> = {
  queued: '已排队',
  running: '运行中',
  completed: '已完成',
  canceled: '已取消',
  failed: '运行失败',
  interrupted: '运行已中断',
};

const labels: Record<string, string> = {
  pair: '交易对',
  start: '开始日期',
  end: '结束日期',
  interval: '运行周期',
  initial_equity: '初始本金',
  fee_rate: '手续费率',
  slippage_bps: '滑点（bps）',
  funding_assumption: '资金费假设',
  market_data: '行情来源',
  source_id: '来源标识',
  timeframe: '行情周期',
  market_adapter_id: '市场平台',
  limit: '历史回看数量',
  evaluation_interval: '评估周期',
  signals: '信号配置',
  risk: '风险设置',
  llm: '模型配置',
  model_evidence: '模型与提示词证据',
  data_coverage: '数据覆盖',
  unmodeled_costs: '未建模成本',
  incomplete_fields: '缺失证据',
  revision: '配置版本',
  updated_at: '配置保存时间',
  parameters: '参数',
  max_single_pct: '单品种最大占比',
  max_total_exposure_pct: '最大总敞口',
  max_margin_used_pct: '最大保证金占比',
  max_drawdown_pct: '最大回撤限制',
  position: '仓位限制',
  loss: '亏损限制',
  default_temperature: '采样温度',
  input_usd_per_mtok: '每百万输入单价（美元）',
  output_usd_per_mtok: '每百万输出单价（美元）',
  historical_news: '历史新闻',
  candles: 'K线覆盖',
  expected: '预期数量',
  available: '可用数量',
  missing: '缺失数量',
  market_source_id: '原行情源',
  market_parameters: '原源参数',
  as_of: '历史截止时间',
  market_type: '市场类型',
  unavailable_context: '缺失上下文',
  model_limitations: '模型局限',
  first_open: '首根开盘',
  last_close: '末根收盘',
  funding: '资金费',
  status: '状态',
  provided_settlements: '提供的结算点',
  applied_settlements: '已使用结算点',
  source_ids: '来源',
  fee: '手续费',
  execution: '执行假设',
  protection: '保护触发规则',
};
export const fieldLabel = (key: string) => labels[key] ?? key;

export function ReadableValues({ value }: { value: unknown }) {
  if (value === null || value === undefined) return <span>未知</span>;
  if (typeof value === 'boolean') return <span>{value ? '是' : '否'}</span>;
  if (typeof value === 'string' || typeof value === 'number') {
    const names: Record<string, string> = {
      unavailable: '不可用',
      available: '可用',
      disabled: '明确关闭',
      partial: '部分可用',
      not_applicable: '不适用',
      available_only: '仅采用已提供的结算点',
      'market impact and intrabar path': '市场冲击与 bar 内路径未建模',
    };
    return <span className="break-words">{names[String(value)] ?? String(value)}</span>;
  }
  if (Array.isArray(value))
    return value.length ? (
      <ul className="space-y-1">
        {(value as unknown[]).map((item, index) => (
          <li key={index}>
            <ReadableValues value={item} />
          </li>
        ))}
      </ul>
    ) : (
      <span>暂无记录</span>
    );
  if (typeof value !== 'object') return <span>未知</span>;
  return (
    <dl className="space-y-1">
      {Object.entries(value as Record<string, unknown>).map(([key, item]) => (
        <div className="grid gap-1 sm:grid-cols-[minmax(8rem,1fr)_2fr]" key={key}>
          <dt className="text-muted-foreground">{fieldLabel(key)}</dt>
          <dd className="min-w-0">
            <ReadableValues value={item} />
          </dd>
        </div>
      ))}
    </dl>
  );
}

export function SnapshotSummary({ run }: { run: BacktestRunStatus }) {
  const snapshot = run.config_snapshot;
  if (snapshot === null) return <p>旧记录没有完整安全配置，无法复用。请使用当前已保存配置发起新实验。</p>;
  const { components: _components, ...fusion } = snapshot.signals;
  return (
    <section className="space-y-3" aria-label="冻结配置">
      <h2 className="font-semibold">冻结配置 · R{snapshot.revision}</h2>
      <p>
        原行情源 {snapshot.market_data.source_id} · 参考周期 {snapshot.market_data.timeframe} · 评估周期{' '}
        {snapshot.signals.evaluation_interval ?? snapshot.market_data.timeframe}
      </p>
      <ReadableValues value={snapshot.market_data.parameters} />
      {snapshot.signals.components.map((component) => (
        <details key={component.component_id} className="rounded border border-border p-3">
          <summary className="min-h-10 cursor-pointer">
            {component.component_id} · {component.enabled ? '已启用' : '已停用'} · 信任权重 {component.weight}
          </summary>
          <ReadableValues value={component.parameters} />
          <ReadableValues value={component.model_identity} />
        </details>
      ))}
      <details className="rounded border border-border p-3">
        <summary className="min-h-10 cursor-pointer">风险、融合与模型请求设置</summary>
        <ReadableValues value={snapshot.risk} />
        <ReadableValues value={fusion} />
        <ReadableValues value={snapshot.llm} />
      </details>
      <p className="text-muted-foreground">
        快照不含交易连接、凭据或真实资金授权。复用不会立即运行；相同参数不保证模型产生相同输出。
      </p>
    </section>
  );
}

export function RunMetrics({ run }: { run: BacktestRunStatus }) {
  const result = run.result;
  if (!result) return <p>尚无完成结果。已保存当前状态，可稍后重新打开。</p>;
  const metrics = result.metrics;
  return (
    <dl className="grid grid-cols-2 gap-4 rounded-lg border border-border bg-card p-4 lg:grid-cols-5">
      <div>
        <dt className="text-muted-foreground">总收益率</dt>
        <dd className="mt-1 text-lg tabular-nums">{(metrics.total_return_pct * 100).toFixed(2)}%</dd>
      </div>
      <div>
        <dt className="text-muted-foreground">最大回撤</dt>
        <dd className="mt-1 text-lg tabular-nums">{(metrics.max_drawdown_pct * 100).toFixed(2)}%</dd>
      </div>
      <div>
        <dt className="text-muted-foreground">夏普比率</dt>
        <dd className="mt-1 text-lg tabular-nums">{metrics.sharpe.toFixed(2)}</dd>
      </div>
      <div>
        <dt className="text-muted-foreground">实际成交 / 平仓回合</dt>
        <dd className="mt-1 text-lg tabular-nums">
          {metrics.fill_count} / {metrics.closed_trade_count}
        </dd>
      </div>
      <div>
        <dt className="text-muted-foreground">平仓胜率</dt>
        <dd className="mt-1">
          {metrics.win_rate === null || !metrics.closed_trade_count
            ? '尚无平仓样本'
            : `${(metrics.win_rate * 100).toFixed(1)}%`}
        </dd>
      </div>
    </dl>
  );
}

export function ResearchNav() {
  return (
    <nav aria-label="研究导航" className="flex flex-wrap gap-4 border-b border-border">
      <Link className="inline-flex min-h-11 items-center text-primary" to="/research">
        回测与历史
      </Link>
      <Link className="inline-flex min-h-11 items-center text-primary" to="/research/market">
        市场观察
      </Link>
      <Link className="inline-flex min-h-11 items-center text-primary" to="/research/analysis">
        仅分析
      </Link>
    </nav>
  );
}

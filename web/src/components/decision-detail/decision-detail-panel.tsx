import { ExternalLink } from 'lucide-react';
import { useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { Link } from 'react-router';

import { PairBadge } from '@/components/PairBadge';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { useDecisionDetail } from '@/hooks/use-decision-detail';
import { formatCurrency, formatDateTime } from '@/lib/format';

import { AgentAnalysisGrid } from './agent-analysis-grid';
import { DebateSection } from './debate-section';
import { ExecutionSection } from './execution-section';
import { MarketContextSection } from './market-context-section';
import { NodeTimelinePipeline } from './node-timeline-pipeline';
import { RiskGateSection } from './risk-gate-section';
import { SectionHeader } from './section-header';
import { SectionNav, type SectionDef } from './section-nav';
import { VerdictCard } from './verdict-card';

interface Props {
  commitHash: string | undefined;
}

const SECTIONS: SectionDef[] = [
  { id: 'sec-summary', label: '1 · 摘要' },
  { id: 'sec-context', label: '2 · 市场上下文' },
  { id: 'sec-agents', label: '3 · 四方分析' },
  { id: 'sec-debate', label: '4 · 辩论' },
  { id: 'sec-verdict', label: '5 · AI 裁决' },
  { id: 'sec-risk', label: '6 · 风控审计' },
  { id: 'sec-exec', label: '7 · 执行' },
  { id: 'sec-meta', label: '8 · 元数据' },
];

export const DecisionDetailPanel = ({ commitHash }: Props) => {
  const { t } = useTranslation('decisions');
  const { data, isLoading, isError } = useDecisionDetail(commitHash);
  const scrollRef = useRef<HTMLDivElement>(null);

  const verdictTone = useMemo(() => {
    if (!data) return 'secondary' as const;
    const a = data.verdict.action?.toLowerCase();
    if (a === 'long' || a === 'buy') return 'success' as const;
    if (a === 'short' || a === 'sell') return 'destructive' as const;
    return 'secondary' as const;
  }, [data]);

  if (!commitHash) {
    return (
      <div className="flex items-center justify-center h-full text-sm text-muted-foreground">
        {t('detail.select_hint', { defaultValue: '选择一条决策查看详情' })}
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="space-y-3 p-4">
        {Array.from({ length: 5 }).map((_, i) => (
          <Skeleton key={i} className="h-24 w-full" />
        ))}
      </div>
    );
  }

  if (isError || !data) {
    return (
      <div className="p-4 text-sm text-destructive">
        {t('detail.load_error', { defaultValue: '加载失败' })}
      </div>
    );
  }

  return (
    <div ref={scrollRef} className="relative h-full overflow-y-auto">
      <SectionNav sections={SECTIONS} containerRef={scrollRef} />

      <div className="space-y-5 p-4">
        {/* Section 1 — Summary (hero) */}
        <section aria-label={t('detail.summary', { defaultValue: '摘要' })}>
          <SectionHeader
            id="sec-summary"
            index={1}
            title={t('detail.summary', { defaultValue: '摘要' })}
            right={(() => {
              const traceUiTpl = import.meta.env.VITE_TRACE_UI_URL as string | undefined;
              if (!data.trace_id || !traceUiTpl) return null;
              const href = traceUiTpl.includes('{trace_id}')
                ? traceUiTpl.replace('{trace_id}', data.trace_id)
                : `${traceUiTpl}${data.trace_id}`;
              return (
                <a
                  href={href}
                  className="inline-flex items-center gap-1 text-xs text-amber-500 hover:underline"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <ExternalLink className="h-3 w-3" aria-hidden />
                  OTel
                </a>
              );
            })()}
          />
          <div
            className="rounded-xl border p-4 flex items-start gap-4"
            style={{
              background: 'linear-gradient(135deg, var(--amber-glow) -20%, hsl(var(--card)) 60%)',
              borderColor: 'var(--amber-500)',
              borderWidth: 1,
            }}
          >
            <div
              className="w-12 h-12 rounded-xl flex items-center justify-center text-primary-foreground shrink-0"
              style={{
                background: 'linear-gradient(135deg, var(--amber-500), var(--amber-600))',
                boxShadow: 'var(--amber-glow) 0px 0px 24px',
              }}
            >
              <ExternalLink className="h-5 w-5" aria-hidden />
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 flex-wrap text-xs">
                <Badge variant={verdictTone}>{data.verdict.action?.toUpperCase()}</Badge>
                <span className="font-mono text-muted-foreground">
                  {data.commit_hash.slice(0, 10)}
                </span>
                <span className="text-muted-foreground">·</span>
                <span className="text-muted-foreground">{formatDateTime(data.ts)}</span>
              </div>
              <h2 className="text-lg font-semibold mt-1.5 flex flex-wrap items-center gap-2">
                <PairBadge
                  pair={data.pair}
                  pairDisplay={data.pair_display}
                  marketType={data.market_type}
                />
                <span>· {data.verdict.action}</span>
                <span className="text-amber-500 font-mono">
                  {(data.verdict.size * 100).toFixed(0)}%
                </span>
                <span>仓位</span>
              </h2>
              {data.verdict.reasoning ? (
                <p className="text-sm text-muted-foreground mt-1.5 leading-relaxed">
                  {data.verdict.reasoning}
                </p>
              ) : null}
            </div>
            <div className="text-right shrink-0">
              <div className="text-[10px] uppercase tracking-wider text-muted-foreground font-medium">
                价格
              </div>
              <div className="font-mono font-semibold tabular-nums">
                {formatCurrency(data.price)}
              </div>
            </div>
          </div>
        </section>

        {/* Section 2 — Market Context */}
        <MarketContextSection
          pair={data.pair}
          price={data.price}
          ts={data.ts}
          traceId={data.trace_id}
          consensus={data.consensus_metrics}
        />

        {/* Section 3 — Agents */}
        <section aria-label={t('detail.agents', { defaultValue: '四方分析' })}>
          <SectionHeader
            id="sec-agents"
            index={3}
            title={t('detail.agents', { defaultValue: '四方分析' })}
            right={
              <span className="text-[11px] text-muted-foreground">
                {data.agent_analyses.length} Agent 并行
              </span>
            }
          />
          <AgentAnalysisGrid analyses={data.agent_analyses} />
        </section>

        {/* Section 4 — Debate */}
        <section aria-label={t('detail.debate', { defaultValue: '辩论' })}>
          <SectionHeader
            id="sec-debate"
            index={4}
            title={t('detail.debate', { defaultValue: '辩论' })}
            right={
              data.debate_turns.length === 0 ? (
                <Badge variant="secondary">跳过</Badge>
              ) : (
                <Badge variant="default">
                  {Math.max(...data.debate_turns.map((t) => t.round))} 轮 · {data.debate_turns.length}{' '}
                  turn
                </Badge>
              )
            }
          />
          {data.debate_turns.length === 0 ? (
            <div className="rounded-lg border border-border bg-muted/40 p-4 text-xs text-muted-foreground">
              {data.debate_gate?.reason || 'debate_gate 跳过辩论（强共识或共同困惑）—— 无分歧需要调解'}
            </div>
          ) : (
            <div className="space-y-2">
              {data.debate_turns.slice(0, 6).map((turn, i) => (
                <div
                  key={`${turn.round}-${turn.from}-${i}`}
                  className="rounded-md border border-border bg-card p-3 text-xs"
                >
                  <div className="mb-1 flex items-center gap-2">
                    <span className="font-mono text-muted-foreground">R{turn.round}</span>
                    <span className="font-medium text-violet-500">{turn.from}</span>
                    {turn.to ? (
                      <>
                        <span className="text-muted-foreground">→</span>
                        <span className="text-muted-foreground">{turn.to}</span>
                      </>
                    ) : null}
                    <span className="ml-auto font-mono text-muted-foreground">
                      {turn.before_direction}({turn.before_confidence.toFixed(2)}) →{' '}
                      {turn.after_direction}({turn.after_confidence.toFixed(2)})
                    </span>
                    <Badge variant="secondary">{turn.move}</Badge>
                  </div>
                  <div className="text-muted-foreground">{turn.reasoning}</div>
                </div>
              ))}
              {data.debate_turns.length > 6 ? (
                <div className="text-center text-xs text-muted-foreground">
                  还有 {data.debate_turns.length - 6} 条 turn —{' '}
                  <Link className="text-amber-500 hover:underline" to={`/debate/${data.commit_hash}`}>
                    查看完整辩论 →
                  </Link>
                </div>
              ) : (
                <div className="text-center">
                  <Link
                    className="text-xs text-amber-500 hover:underline"
                    to={`/debate/${data.commit_hash}`}
                  >
                    在辩论页中查看完整可视化 →
                  </Link>
                </div>
              )}
            </div>
          )}
          {data.debate_rounds.length > 0 ? (
            <div className="mt-3">
              <div className="mb-1 text-[10px] uppercase tracking-wider text-muted-foreground">
                Legacy bull/bear rounds
              </div>
              <DebateSection rounds={data.debate_rounds} />
            </div>
          ) : null}
        </section>

        {/* Section 5 — Verdict */}
        <section aria-label={t('detail.verdict', { defaultValue: 'AI 裁决' })}>
          <SectionHeader
            id="sec-verdict"
            index={5}
            title={t('detail.verdict', { defaultValue: 'AI 裁决' })}
          />
          <div className="space-y-3">
            <VerdictCard verdict={data.verdict} />
          </div>
        </section>

        {/* Section 6 — Risk audit */}
        <section aria-label={t('detail.risk_gate', { defaultValue: '风控审计' })}>
          <SectionHeader
            id="sec-risk"
            index={6}
            title={t('detail.risk_gate', { defaultValue: '风控审计' })}
            right={
              <Badge variant={data.risk_gate.passed ? 'success' : 'destructive'}>
                {data.risk_gate.passed ? 'PASS' : 'REJECT'}{' '}
                {data.risk_gate.checks.length > 0
                  ? `${data.risk_gate.checks.filter((c) => c.passed).length}/${data.risk_gate.checks.length}`
                  : ''}
              </Badge>
            }
          />
          <RiskGateSection gate={data.risk_gate} />
        </section>

        {/* Section 7 — Execution */}
        <section aria-label={t('detail.execution', { defaultValue: '执行' })}>
          <SectionHeader
            id="sec-exec"
            index={7}
            title={t('detail.execution', { defaultValue: '执行' })}
          />
          {data.execution ? (
            <ExecutionSection execution={data.execution} />
          ) : (
            <div className="rounded-lg border border-border bg-muted/40 p-4 text-xs text-muted-foreground">
              未执行（观望 / 被风控拒绝 / 回测模式）
            </div>
          )}
        </section>

        {/* Section 8 — Meta (latency + tokens + node timeline) */}
        <section aria-label={t('detail.meta', { defaultValue: '元数据' })}>
          <SectionHeader
            id="sec-meta"
            index={8}
            title={t('detail.meta', { defaultValue: '元数据' })}
          />
          <div className="grid grid-cols-2 gap-3">
            <div className="rounded-lg border border-border bg-card p-4">
              <div className="mb-2 text-[10px] uppercase tracking-wider font-medium text-muted-foreground">
                延迟分解 · 总 {data.latency_breakdown.total_ms.toFixed(0)}ms
              </div>
              <div className="space-y-2">
                {(
                  [
                    ['data', data.latency_breakdown.data_ms, 'var(--cyan-500)'],
                    ['agents', data.latency_breakdown.agents_ms, 'var(--cyan-500)'],
                    ['debate', data.latency_breakdown.debate_ms, 'var(--violet-500)'],
                    ['verdict', data.latency_breakdown.verdict_ms, 'var(--amber-500)'],
                    ['risk', data.latency_breakdown.risk_ms, 'var(--violet-500)'],
                    ['execute', data.latency_breakdown.execute_ms, 'var(--trade-long)'],
                  ] as const
                ).map(([label, ms, color]) => {
                  const total = data.latency_breakdown.total_ms || 1;
                  return (
                    <div key={label} className="flex items-center gap-2 text-xs">
                      <div className="w-16 text-muted-foreground">{label}</div>
                      <div className="relative h-1.5 flex-1 overflow-hidden rounded bg-muted">
                        <div
                          className="h-full"
                          style={{ width: `${(ms / total) * 100}%`, background: color }}
                        />
                      </div>
                      <div className="w-14 text-right font-mono tabular-nums">
                        {ms.toFixed(0)}ms
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
            <div className="rounded-lg border border-border bg-card p-4">
              <div className="mb-2 text-[10px] uppercase tracking-wider font-medium text-muted-foreground">
                Token · 成本 ·{' '}
                <span className="text-amber-500">${data.token_usage.cost_usd.toFixed(4)}</span>
              </div>
              <div className="grid grid-cols-3 gap-3">
                <div>
                  <div className="text-[10px] text-muted-foreground">输入</div>
                  <div className="font-mono text-lg tabular-nums">
                    {data.token_usage.input_tokens.toLocaleString()}
                  </div>
                </div>
                <div>
                  <div className="text-[10px] text-muted-foreground">输出</div>
                  <div className="font-mono text-lg tabular-nums">
                    {data.token_usage.output_tokens.toLocaleString()}
                  </div>
                </div>
                <div>
                  <div className="text-[10px] text-muted-foreground">
                    调用 · 缓存命中
                  </div>
                  <div className="font-mono text-lg tabular-nums">
                    {data.token_usage.calls}
                    <span className="text-sm text-muted-foreground ml-1">
                      / {data.token_usage.cache_hits}
                    </span>
                  </div>
                </div>
              </div>
              {Object.keys(data.token_usage.by_model).length > 0 ? (
                <div className="mt-3 space-y-1 border-t border-border pt-2 text-[11px]">
                  {Object.entries(data.token_usage.by_model).map(([model, stats]) => (
                    <div key={model} className="flex justify-between text-muted-foreground">
                      <span className="font-mono truncate">{model}</span>
                      <span className="font-mono tabular-nums">
                        ${stats.cost_usd?.toFixed(4) ?? '0'} ({stats.calls ?? 0} calls)
                      </span>
                    </div>
                  ))}
                </div>
              ) : null}
            </div>
          </div>
          <div className="mt-3 space-y-3">
            <NodeTimelinePipeline entries={data.node_timeline} />
          </div>
        </section>
      </div>
    </div>
  );
};

/**
 * spec 021 — CaseChainCard
 *
 * 单个 cycle 的"完整因果链条"展示，分 5 段：
 *   1. 市场环境（regime + pair + timestamp）
 *   2. 4 agent analyses（tech / chain / news / macro）
 *   3. verdict（action + reasoning + applied_patterns）
 *   4. risk gate
 *   5. execution（fill_status / SL / TP / hit_sl / actual_exit_price） + IVE 分类
 *
 * 节点：lazy fetch — 仅在 expand 时拉 `/api/memory/cases/<cycle_id>`。
 */

import { ChevronDown, ChevronRight } from 'lucide-react';
import { useState } from 'react';

import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { cn } from '@/lib/cn';
import { formatCurrency, formatDateTime } from '@/lib/format';

import { useCaseDetail, type CaseDetail } from '../queries';

const Section = ({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) => (
  <div className="space-y-1.5">
    <div className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
      {label}
    </div>
    <div className="text-xs text-foreground/90">{children}</div>
  </div>
);

const AgentBlock = ({ agentKey, text }: { agentKey: string; text: string }) => {
  const label = agentKey.replace(/_agent$/, '');
  return (
    <div className="rounded-md border border-border bg-background/40 p-2">
      <div className="mb-1 inline-flex items-center gap-1">
        <Badge variant="outline" className="text-[10px] px-1.5 py-0">
          {label}
        </Badge>
      </div>
      <p className="whitespace-pre-wrap text-[11px] leading-relaxed text-foreground/85">
        {text}
      </p>
    </div>
  );
};

const Body = ({ data }: { data: CaseDetail }) => {
  const verdictColor =
    data.verdict_action === 'long'
      ? 'text-trade-long'
      : data.verdict_action === 'short'
        ? 'text-trade-short'
        : 'text-amber-500';
  const fillStatus = data.trade_execution?.fill_status ?? '—';
  const hitSl = data.trade_execution?.hit_sl;
  const entry = data.trade_execution?.entry_price;
  const sl = data.trade_execution?.stop_loss;
  const tp = data.trade_execution?.take_profit;
  const exit = data.trade_execution?.actual_exit_price;

  return (
    <div className="space-y-4 border-t border-border bg-muted/10 px-4 py-3">
      {/* 1. 市场环境 */}
      <Section label="① 市场环境">
        <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] tabular-nums">
          <span>
            <span className="text-muted-foreground">pair</span>{' '}
            <span className="font-mono">{data.pair}</span>
          </span>
          <span>
            <span className="text-muted-foreground">at</span>{' '}
            <span className="font-mono">{formatDateTime(data.timestamp)}</span>
          </span>
          {Object.entries(data.snapshot_summary).length > 0 ? (
            <span className="text-[10px] text-muted-foreground">
              snapshot: {Object.keys(data.snapshot_summary).length} 字段
            </span>
          ) : null}
        </div>
      </Section>

      {/* 2. agent analyses */}
      <Section label={`② 4 Agent 分析（${Object.keys(data.agent_analyses).length}）`}>
        {Object.keys(data.agent_analyses).length === 0 ? (
          <span className="text-[10px] text-muted-foreground">无</span>
        ) : (
          <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
            {Object.entries(data.agent_analyses).map(([k, v]) => (
              <AgentBlock key={k} agentKey={k} text={v} />
            ))}
          </div>
        )}
      </Section>

      {/* 3. verdict */}
      <Section label="③ 决议（verdict）">
        <div className="space-y-1.5">
          <div className="flex flex-wrap items-center gap-2 text-[11px]">
            <span className="text-muted-foreground">action</span>
            <span className={cn('font-mono font-semibold uppercase', verdictColor)}>
              {data.verdict_action}
            </span>
            {data.applied_patterns.length > 0 ? (
              <>
                <span className="text-muted-foreground">·</span>
                <span className="text-muted-foreground">applied</span>
                {data.applied_patterns.map((p) => (
                  <Badge key={p} variant="outline" className="text-[10px] px-1.5 py-0">
                    {p}
                  </Badge>
                ))}
              </>
            ) : null}
          </div>
          {data.verdict_reasoning ? (
            <p className="whitespace-pre-wrap text-[11px] leading-relaxed text-foreground/85">
              {data.verdict_reasoning}
            </p>
          ) : (
            <span className="text-[10px] text-muted-foreground">无 reasoning</span>
          )}
        </div>
      </Section>

      {/* 4. risk gate */}
      <Section label="④ 风控（risk gate）">
        <span
          className={cn(
            'inline-flex items-center gap-1.5 font-mono text-[11px]',
            data.risk_gate_passed ? 'text-trade-long' : 'text-trade-short',
          )}
        >
          {data.risk_gate_passed ? 'PASS' : 'REJECT'}
        </span>
      </Section>

      {/* 5. execution + IVE */}
      <Section label="⑤ 撮合 / 结算 + IVE 失败归因">
        <div className="space-y-2">
          <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-[11px] tabular-nums md:grid-cols-3">
            <div>
              <span className="text-muted-foreground">fill</span>{' '}
              <span className="font-mono">{fillStatus}</span>
            </div>
            <div>
              <span className="text-muted-foreground">entry</span>{' '}
              <span className="font-mono">{entry !== undefined && entry !== null ? entry : '—'}</span>
            </div>
            <div>
              <span className="text-muted-foreground">SL</span>{' '}
              <span className="font-mono">{sl !== undefined && sl !== null ? sl : '—'}</span>
            </div>
            <div>
              <span className="text-muted-foreground">TP</span>{' '}
              <span className="font-mono">{tp !== undefined && tp !== null ? tp : '—'}</span>
            </div>
            <div>
              <span className="text-muted-foreground">exit</span>{' '}
              <span className="font-mono">{exit !== undefined && exit !== null ? exit : '—'}</span>
            </div>
            <div>
              <span className="text-muted-foreground">hit_sl</span>{' '}
              <span
                className={cn(
                  'font-mono',
                  hitSl === true
                    ? 'text-trade-short'
                    : hitSl === false
                      ? 'text-trade-long'
                      : 'text-muted-foreground',
                )}
              >
                {hitSl === true ? '是' : hitSl === false ? '否' : '—'}
              </span>
            </div>
          </div>
          <div className="flex items-center gap-2 text-[11px]">
            <span className="text-muted-foreground">final_pnl</span>
            {data.final_pnl === null || data.final_pnl === undefined ? (
              <span className="text-muted-foreground">— (未结算)</span>
            ) : (
              <span
                className={cn(
                  'font-mono font-semibold tabular-nums',
                  data.final_pnl > 0
                    ? 'text-trade-long'
                    : data.final_pnl < 0
                      ? 'text-trade-short'
                      : 'text-muted-foreground',
                )}
              >
                {formatCurrency(data.final_pnl)}
              </span>
            )}
          </div>
          {data.ive_classification ? (
            <div className="rounded-md border border-border/50 bg-background/40 p-2">
              <div className="mb-1 flex items-center gap-2">
                <Badge variant="outline" className="text-[10px] px-1.5 py-0">
                  IVE: {data.ive_classification.failure_type}
                </Badge>
                <span className="text-[10px] text-muted-foreground">
                  confidence {(data.ive_classification.confidence * 100).toFixed(0)}%
                </span>
              </div>
              <p className="text-[11px] leading-relaxed text-foreground/80">
                {data.ive_classification.reasoning}
              </p>
            </div>
          ) : null}
        </div>
      </Section>
    </div>
  );
};

export const CaseChainCard = ({
  cycleId,
  defaultOpen = false,
}: {
  cycleId: string;
  /** Auto-expand on mount (used to surface the most representative cycle for a pattern). */
  defaultOpen?: boolean;
}) => {
  const [open, setOpen] = useState(defaultOpen);
  const { data, isLoading, isError } = useCaseDetail(open ? cycleId : null);

  return (
    <div className="overflow-hidden rounded-md border border-border">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className={cn(
          'flex w-full items-center gap-2 px-3 py-2 text-left text-xs',
          'transition-colors hover:bg-muted/30 focus-visible:bg-muted/30 focus-visible:outline-none',
        )}
        aria-expanded={open}
      >
        {open ? (
          <ChevronDown className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
        ) : (
          <ChevronRight className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
        )}
        <span className="font-mono text-[11px] text-foreground">{cycleId}</span>
        {data ? (
          <>
            <span className="text-muted-foreground">·</span>
            <span className="font-mono text-[11px] text-muted-foreground">{data.pair}</span>
            <span className="ml-auto flex items-center gap-2 text-[11px]">
              <span
                className={cn(
                  'font-mono uppercase',
                  data.verdict_action === 'long'
                    ? 'text-trade-long'
                    : data.verdict_action === 'short'
                      ? 'text-trade-short'
                      : 'text-amber-500',
                )}
              >
                {data.verdict_action}
              </span>
              {data.final_pnl !== null && data.final_pnl !== undefined ? (
                <span
                  className={cn(
                    'tabular-nums',
                    data.final_pnl > 0 ? 'text-trade-long' : 'text-trade-short',
                  )}
                >
                  {formatCurrency(data.final_pnl)}
                </span>
              ) : (
                <span className="text-muted-foreground">未结算</span>
              )}
            </span>
          </>
        ) : (
          <span className="ml-auto text-[10px] text-muted-foreground">
            {open ? (isLoading ? '加载中…' : isError ? '加载失败' : '') : '点击展开'}
          </span>
        )}
      </button>
      {open ? (
        isLoading ? (
          <div className="px-4 py-3">
            <Skeleton className="h-32 w-full" />
          </div>
        ) : isError || !data ? (
          <div className="border-t border-border bg-muted/10 px-4 py-3 text-[11px] text-trade-short">
            无法加载 case 详情（cycle_id={cycleId}）
          </div>
        ) : (
          <Body data={data} />
        )
      ) : null}
    </div>
  );
};

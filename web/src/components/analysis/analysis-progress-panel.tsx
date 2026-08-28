import { CheckCircle2, GitMerge, LoaderCircle, XCircle } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { Link } from 'react-router';

import type { AnalysisProgressState } from '@/hooks/use-analysis-progress';
import { cn } from '@/lib/cn';

import { AgentCard } from './agent-card';

interface AnalysisProgressPanelProps {
  progress: AnalysisProgressState;
  onInterrupt?: (() => void) | undefined;
}

const componentLabel = (id: string) => id === 'kronos' ? 'Kronos' : id === 'llm_committee' ? 'LLM 四智能体委员会' : id;

export function AnalysisProgressPanel({
  progress,
  onInterrupt,
}: AnalysisProgressPanelProps) {
  const { t } = useTranslation('chat');
  const components = Object.entries(progress.components);
  const agents = Object.entries(progress.agents);
  const hasActivity = progress.status !== 'idle' || components.length > 0 || agents.length > 0;

  if (!hasActivity) return null;

  if (progress.cancelled) {
    return (
      <div className="m-4 rounded-lg border border-border bg-muted/40 px-4 py-3 text-sm text-muted-foreground">
        <div className="flex items-center gap-2"><XCircle className="h-4 w-4" />本轮分析已取消</div>
      </div>
    );
  }

  return (
    <div className="m-4 space-y-3 rounded-xl border border-border bg-muted/20 p-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold">信号融合周期</h3>
          {progress.cycleId ? <p className="mt-0.5 font-mono text-[10px] text-muted-foreground">{progress.cycleId}</p> : null}
        </div>
        <div className="flex items-center gap-2">
          <span className={cn('rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase', progress.status === 'running' ? 'border-amber-500/40 text-amber-500' : progress.status === 'completed' ? 'border-success/40 text-success' : progress.status === 'failed' ? 'border-destructive/40 text-destructive' : 'border-border text-muted-foreground')}>
            {progress.status}
          </span>
          {progress.status === 'running' && onInterrupt ? (
            <button type="button" onClick={onInterrupt} className="rounded border border-border px-2 py-1 text-xs hover:bg-muted" aria-label={t('interrupt', { defaultValue: 'Interrupt' })}>
              {t('interrupt', { defaultValue: 'Interrupt' })}
            </button>
          ) : null}
        </div>
      </div>

      {components.length > 0 ? (
        <div className="grid gap-2 sm:grid-cols-2">
          {components.map(([id, component]) => (
            <div key={id} className="rounded-lg border border-border bg-card p-3">
              <div className="flex items-center justify-between gap-2 text-xs">
                <span className="font-semibold">{componentLabel(id)}</span>
                {component.status === 'running' ? <LoaderCircle className="h-4 w-4 animate-spin text-amber-500" /> : component.status === 'done' ? <CheckCircle2 className="h-4 w-4 text-success" /> : <XCircle className="h-4 w-4 text-destructive" />}
              </div>
              {component.signal ? (
                <div className="mt-2 flex items-center justify-between font-mono text-xs text-muted-foreground">
                  <span>{component.signal.direction}</span><span>{(component.signal.confidence * 100).toFixed(0)}%</span>
                </div>
              ) : null}
              {component.error ? <p className="mt-2 text-xs text-destructive">{component.error}</p> : null}
            </div>
          ))}
        </div>
      ) : null}

      {agents.length > 0 ? (
        <div>
          <div className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">委员会内部分析</div>
          <div className="grid grid-cols-2 gap-2">
            {agents.map(([agentId, agent]) => (
              <AgentCard key={agentId} agentId={agentId} agent={agent} />
            ))}
          </div>
        </div>
      ) : null}

      {progress.debateRound > 0 ? <div className="text-xs text-muted-foreground">内部辩论第 {progress.debateRound} 轮</div> : null}

      {progress.fusion ? (
        <div className="flex items-center justify-between rounded-lg border border-amber-500/30 bg-amber-500/5 px-3 py-2">
          <span className="flex items-center gap-2 text-xs font-semibold"><GitMerge className="h-4 w-4 text-amber-500" />确定性融合完成</span>
          <span className="font-mono text-sm font-semibold">{progress.fusion.score >= 0 ? '+' : ''}{progress.fusion.score.toFixed(2)}</span>
        </div>
      ) : null}

      {progress.cycleId && progress.status !== 'running' ? <div className="text-right"><Link className="text-xs font-medium text-amber-500 hover:underline" to={`/decisions/${progress.cycleId}`}>查看周期决策 →</Link></div> : null}
    </div>
  );
}

import { ExternalLink } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import { Skeleton } from '@/components/ui/skeleton';
import { useDecisionDetail } from '@/hooks/use-decision-detail';
import { formatCurrency, formatDateTime } from '@/lib/format';

import { AgentAnalysisGrid } from './agent-analysis-grid';
import { DebateSection } from './debate-section';
import { ExecutionSection } from './execution-section';
import { ExperienceMemorySection } from './experience-memory-section';
import { NodeTimelinePipeline } from './node-timeline-pipeline';
import { RiskGateSection } from './risk-gate-section';
import { VerdictCard } from './verdict-card';

interface Props {
  commitHash: string | undefined;
}

export const DecisionDetailPanel = ({ commitHash }: Props) => {
  const { t } = useTranslation('decisions');
  const { data, isLoading, isError } = useDecisionDetail(commitHash);

  if (!commitHash) {
    return (
      <div className="flex items-center justify-center h-full text-sm text-muted-foreground">
        {t('detail.select_hint', { defaultValue: '选择一条决策查看详情' })}
      </div>
    );
  }

  if (isLoading) {
    return <div className="space-y-3 p-4">{Array.from({ length: 5 }).map((_, i) => <Skeleton key={i} className="h-24 w-full" />)}</div>;
  }

  if (isError || !data) {
    return <div className="p-4 text-sm text-destructive">{t('detail.load_error', { defaultValue: '加载失败' })}</div>;
  }

  return (
    <div className="space-y-4 p-4 overflow-y-auto">
      <header className="space-y-1">
        <div className="flex items-center gap-2 text-sm">
          <span className="font-mono text-xs text-muted-foreground">{data.commit_hash.slice(0, 8)}</span>
          <span className="font-semibold">{data.pair}</span>
          <span className="tabular-nums">{formatCurrency(data.price)}</span>
          <span className="text-muted-foreground">{formatDateTime(data.ts)}</span>
          {data.trace_id && (
            <a
              href={`/traces/${data.trace_id}`}
              className="ml-auto inline-flex items-center gap-1 text-xs text-primary hover:underline"
              target="_blank"
              rel="noopener noreferrer"
            >
              <ExternalLink className="h-3 w-3" aria-hidden />
              OTel
            </a>
          )}
        </div>
      </header>

      <NodeTimelinePipeline entries={data.node_timeline} />
      <AgentAnalysisGrid analyses={data.agent_analyses} />
      <ExperienceMemorySection memory={data.experience_memory_ref} />
      <DebateSection rounds={data.debate_rounds} />
      <VerdictCard verdict={data.verdict} />
      <RiskGateSection gate={data.risk_gate} />
      <ExecutionSection execution={data.execution} />
    </div>
  );
};

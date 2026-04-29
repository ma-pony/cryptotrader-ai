import { useTranslation } from 'react-i18next';

import type { AnalysisProgressState } from '@/hooks/use-analysis-progress';

import { AgentCard } from './agent-card';
import { NodeProgressBar } from './node-progress-bar';
import { VerdictCard } from './verdict-card';

interface AnalysisProgressPanelProps {
  progress: AnalysisProgressState;
  sessionId: string | null;
  onSteer?: ((target: string, instruction: string) => void) | undefined;
  onInterrupt?: (() => void) | undefined;
}

export function AnalysisProgressPanel({
  progress,
  sessionId,
  onSteer,
  onInterrupt,
}: AnalysisProgressPanelProps) {
  const { t } = useTranslation('chat');
  const agentEntries = Object.entries(progress.agents);
  const hasActivity = Object.keys(progress.nodes).length > 0 || agentEntries.length > 0;

  if (!hasActivity && !progress.verdict) return null;

  return (
    <div className="space-y-3 rounded-lg border p-4 bg-muted/30">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold">
          {t('analysis_progress', { defaultValue: 'Analysis Progress' })}
        </h3>
        {sessionId && !progress.interrupted && onInterrupt && (
          <button
            type="button"
            onClick={onInterrupt}
            className="text-xs px-2 py-1 rounded border hover:bg-muted"
            aria-label={t('interrupt', { defaultValue: 'Interrupt' })}
          >
            {t('interrupt', { defaultValue: 'Interrupt' })}
          </button>
        )}
      </div>

      <NodeProgressBar nodes={progress.nodes} />

      {agentEntries.length > 0 && (
        <div className="grid grid-cols-2 gap-2">
          {agentEntries.map(([agentId, agent]) => (
            <AgentCard
              key={agentId}
              agentId={agentId}
              agent={agent}
              onSteer={
                agent.status === 'thinking' && onSteer
                  ? (instruction) => onSteer(agentId, instruction)
                  : undefined
              }
            />
          ))}
        </div>
      )}

      {progress.debateRound > 0 && (
        <div className="text-xs text-muted-foreground">
          {t('debate_round', { defaultValue: 'Debate round' })}: {progress.debateRound}
        </div>
      )}

      {progress.verdict && <VerdictCard verdict={progress.verdict} />}

      {progress.riskCheck && (
        <div className={`text-xs px-3 py-1 rounded ${
          progress.riskCheck.allowed
            ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300'
            : 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300'
        }`}>
          {progress.riskCheck.allowed
            ? t('risk_passed', { defaultValue: 'Risk check passed' })
            : `${t('risk_rejected', { defaultValue: 'Risk rejected' })}: ${progress.riskCheck.reason}`}
        </div>
      )}
    </div>
  );
}

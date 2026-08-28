import type { AgentProgress } from '@/hooks/use-analysis-progress';

interface AgentCardProps {
  agentId: string;
  agent: AgentProgress;
}

const DIRECTION_COLORS: Record<string, string> = {
  bullish: 'text-green-600 dark:text-green-400',
  bearish: 'text-red-600 dark:text-red-400',
  neutral: 'text-yellow-600 dark:text-yellow-400',
};

export function AgentCard({ agentId, agent }: AgentCardProps) {
  return (
    <div className="rounded-lg border p-3 bg-card text-card-foreground">
      <div className="flex items-center justify-between mb-2">
        <span className="font-medium text-sm">{agentId.replace(/_/g, ' ')}</span>
        <div className="flex items-center gap-2">
          <span className={`text-xs ${agent.status === 'thinking' ? 'animate-pulse text-blue-500' : 'text-muted-foreground'}`}>
            {agent.status}
          </span>
        </div>
      </div>

      {agent.status === 'done' && (
        <div className="text-sm">
          <span className={DIRECTION_COLORS[agent.direction] ?? 'text-muted-foreground'}>
            {agent.direction}
          </span>
          <span className="ml-2 text-muted-foreground">
            {(agent.confidence * 100).toFixed(0)}%
          </span>
        </div>
      )}
    </div>
  );
}

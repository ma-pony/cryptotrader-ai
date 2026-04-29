import { useState } from 'react';
import { useTranslation } from 'react-i18next';

import type { AgentProgress } from '@/hooks/use-analysis-progress';

interface AgentCardProps {
  agentId: string;
  agent: AgentProgress;
  onSteer?: ((instruction: string) => void) | undefined;
}

const DIRECTION_COLORS: Record<string, string> = {
  bullish: 'text-green-600 dark:text-green-400',
  bearish: 'text-red-600 dark:text-red-400',
  neutral: 'text-yellow-600 dark:text-yellow-400',
};

export function AgentCard({ agentId, agent, onSteer }: AgentCardProps) {
  const { t } = useTranslation('chat');
  const [steerInput, setSteerInput] = useState('');

  const handleSubmitSteer = () => {
    if (steerInput.trim() && onSteer) {
      onSteer(steerInput.trim());
      setSteerInput('');
    }
  };

  return (
    <div className="rounded-lg border p-3 bg-card text-card-foreground">
      <div className="flex items-center justify-between mb-2">
        <span className="font-medium text-sm">{agentId.replace(/_/g, ' ')}</span>
        <div className="flex items-center gap-2">
          {agent.steered && (
            <span className="text-xs bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300 px-1.5 py-0.5 rounded">
              {t('steered', { defaultValue: 'steered' })}
            </span>
          )}
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

      {agent.status === 'thinking' && onSteer && (
        <div className="mt-2 flex gap-1">
          <input
            type="text"
            value={steerInput}
            onChange={(e) => setSteerInput(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') handleSubmitSteer(); }}
            className="flex-1 text-xs px-2 py-1 rounded border bg-background"
            placeholder={t('steer_placeholder', { defaultValue: 'Steering instruction...' })}
            aria-label={t('steer_input', { defaultValue: 'Steering input' })}
          />
          <button
            type="button"
            onClick={handleSubmitSteer}
            className="text-xs px-2 py-1 rounded bg-primary text-primary-foreground"
          >
            {t('send', { defaultValue: 'Send' })}
          </button>
        </div>
      )}
    </div>
  );
}

import { useTranslation } from 'react-i18next';

import type { NodeProgress } from '@/hooks/use-analysis-progress';

interface NodeProgressBarProps {
  nodes: Record<string, NodeProgress>;
}

const STATUS_COLORS = {
  pending: 'bg-gray-300 dark:bg-gray-600',
  running: 'bg-blue-500 animate-pulse',
  done: 'bg-green-500',
} as const;

export function NodeProgressBar({ nodes }: NodeProgressBarProps) {
  const { t } = useTranslation('chat');
  const entries = Object.entries(nodes);

  if (entries.length === 0) return null;

  return (
    <div className="flex items-center gap-1 overflow-x-auto py-2">
      {entries.map(([name, node]) => (
        <div key={name} className="flex flex-col items-center min-w-[60px]">
          <div className={`h-2 w-full rounded ${STATUS_COLORS[node.status]}`} />
          <span className="mt-1 text-xs text-muted-foreground truncate max-w-[80px]">
            {name.replace(/_/g, ' ')}
          </span>
          {node.status === 'done' && node.duration_ms > 0 && (
            <span className="text-[10px] text-muted-foreground">
              {node.duration_ms}{t('ms', { defaultValue: 'ms' })}
            </span>
          )}
        </div>
      ))}
    </div>
  );
}

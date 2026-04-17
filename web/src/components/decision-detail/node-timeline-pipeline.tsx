import { useTranslation } from 'react-i18next';

import { cn } from '@/lib/cn';
import type { NodeTimelineEntry } from '@/types/api';

interface Props {
  entries: NodeTimelineEntry[];
}

export const NodeTimelinePipeline = ({ entries }: Props) => {
  const { t } = useTranslation('decisions');
  if (entries.length === 0) return null;

  const totalMs = entries.reduce((s, e) => s + e.duration_ms, 0);

  return (
    <section aria-label={t('detail.node_timeline', { defaultValue: '节点序列' })}>
      <h3 className="text-sm font-medium mb-2">{t('detail.node_timeline', { defaultValue: '节点序列' })}</h3>
      <div className="flex items-center gap-1 overflow-x-auto py-1">
        {entries.map((entry, i) => {
          const widthPct = totalMs > 0 ? Math.max(8, (entry.duration_ms / totalMs) * 100) : 100 / entries.length;
          return (
            <div key={i} className="flex flex-col items-center" style={{ width: `${widthPct}%`, minWidth: 60 }}>
              <div
                className={cn(
                  'w-full h-6 rounded text-[10px] flex items-center justify-center truncate',
                  'bg-primary/20 text-primary-foreground',
                )}
              >
                {entry.node}
              </div>
              <span className="text-[10px] text-muted-foreground tabular-nums mt-0.5">{entry.duration_ms}ms</span>
            </div>
          );
        })}
      </div>
    </section>
  );
};

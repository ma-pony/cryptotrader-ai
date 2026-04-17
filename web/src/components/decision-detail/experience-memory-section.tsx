import { useState } from 'react';
import { useTranslation } from 'react-i18next';

import type { ExperienceMemoryRef } from '@/types/api';

interface Props {
  memory: ExperienceMemoryRef;
}

const CollapsibleGroup = ({ title, items, renderItem }: { title: string; items: unknown[]; renderItem: (item: unknown, i: number) => React.ReactNode }) => {
  const [open, setOpen] = useState(false);
  if (items.length === 0) return null;

  return (
    <div className="border rounded-md">
      <button
        className="w-full text-left px-3 py-2 text-xs font-medium flex items-center justify-between hover:bg-muted/50"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
      >
        <span>{title} ({items.length})</span>
        <span className="text-muted-foreground">{open ? '−' : '+'}</span>
      </button>
      {open && <div className="px-3 pb-2 space-y-1">{items.map(renderItem)}</div>}
    </div>
  );
};

export const ExperienceMemorySection = ({ memory }: Props) => {
  const { t } = useTranslation('decisions');

  const hasContent = memory.success_patterns.length > 0 || memory.forbidden_zones.length > 0 || memory.strategic_insights.length > 0;
  if (!hasContent) return null;

  return (
    <section aria-label={t('detail.experience', { defaultValue: '经验记忆' })}>
      <h3 className="text-sm font-medium mb-2">{t('detail.experience', { defaultValue: '经验记忆' })}</h3>
      <div className="space-y-2">
        <CollapsibleGroup
          title={t('detail.success_patterns', { defaultValue: '成功模式' })}
          items={memory.success_patterns}
          renderItem={(item, i) => (
            <pre key={i} className="text-[10px] bg-muted rounded p-1 overflow-x-auto">{JSON.stringify(item, null, 2)}</pre>
          )}
        />
        <CollapsibleGroup
          title={t('detail.forbidden_zones', { defaultValue: '禁止区域' })}
          items={memory.forbidden_zones}
          renderItem={(item, i) => (
            <pre key={i} className="text-[10px] bg-muted rounded p-1 overflow-x-auto">{JSON.stringify(item, null, 2)}</pre>
          )}
        />
        <CollapsibleGroup
          title={t('detail.strategic_insights', { defaultValue: '战略洞察' })}
          items={memory.strategic_insights}
          renderItem={(item, i) => (
            <p key={i} className="text-xs text-muted-foreground">{String(item)}</p>
          )}
        />
      </div>
    </section>
  );
};

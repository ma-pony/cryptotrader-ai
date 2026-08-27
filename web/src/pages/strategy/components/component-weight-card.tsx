import { Power, Radio } from 'lucide-react';
import { type CSSProperties } from 'react';
import { useTranslation } from 'react-i18next';

import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/cn';

export interface ComponentWeightDraft {
  component_id: string;
  enabled: boolean;
  weight: number;
}

interface Props {
  component: ComponentWeightDraft;
  displayName: string;
  description: string;
  accent: string;
  onChange: (next: ComponentWeightDraft) => void;
}

export const ComponentWeightCard = ({
  component,
  displayName,
  description,
  accent,
  onChange,
}: Props) => {
  const { t } = useTranslation('strategy');
  const percent = Math.round(component.weight * 100);
  const style = { '--signal-accent': accent } as CSSProperties;

  return (
    <article
      className={cn(
        'group relative overflow-hidden rounded-xl border bg-card p-5 transition-[border-color,opacity,transform] duration-200',
        component.enabled
          ? 'border-[color:var(--signal-accent)]/50 shadow-sm hover:-translate-y-0.5'
          : 'border-border opacity-60',
      )}
      style={style}
    >
      <div className="absolute inset-y-0 left-0 w-1 bg-[var(--signal-accent)]" aria-hidden />
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-[color:var(--signal-accent)]/10 text-[var(--signal-accent)]">
              <Radio className="h-4 w-4" aria-hidden="true" />
            </span>
            <div>
              <h3 className="text-sm font-semibold text-foreground">{displayName}</h3>
              <code className="text-[10px] text-muted-foreground">{component.component_id}</code>
            </div>
          </div>
          <p className="mt-3 max-w-xl text-xs leading-5 text-muted-foreground">{description}</p>
        </div>

        <label className="relative inline-flex cursor-pointer items-center gap-2">
          <span className="sr-only">{t('components.toggle', { name: displayName })}</span>
          <input
            type="checkbox"
            className="peer sr-only"
            checked={component.enabled}
            onChange={(event) =>
              onChange({
                ...component,
                enabled: event.target.checked,
                weight: event.target.checked ? component.weight : 0,
              })
            }
          />
          <span className="relative h-6 w-11 rounded-full bg-muted transition-colors after:absolute after:left-1 after:top-1 after:h-4 after:w-4 after:rounded-full after:bg-muted-foreground after:transition-transform peer-checked:bg-[var(--signal-accent)] peer-checked:after:translate-x-5 peer-checked:after:bg-white peer-focus-visible:ring-2 peer-focus-visible:ring-ring peer-focus-visible:ring-offset-2" />
        </label>
      </div>

      <div className="mt-5 flex items-end justify-between gap-4 border-t border-border/70 pt-4">
        <Badge variant={component.enabled ? 'outline' : 'secondary'}>
          <Power className="mr-1 h-3 w-3" />
          {component.enabled ? t('components.enabled') : t('components.disabled')}
        </Badge>
        <label className="flex items-end gap-2">
          <span className="pb-2 text-[11px] font-medium text-muted-foreground">
            {t('components.trust')}
          </span>
          <span className="relative">
            <input
              type="number"
              min={0}
              max={100}
              step={1}
              value={percent}
              disabled={!component.enabled}
              aria-label={t('components.weight_label', { name: displayName })}
              onChange={(event) => {
                const value = Math.max(0, Math.min(100, Number(event.target.value)));
                onChange({ ...component, weight: value / 100 });
              }}
              className="h-10 w-24 rounded-lg border border-input bg-background px-3 pr-8 text-right font-mono text-base font-semibold tabular-nums outline-none transition focus:border-[var(--signal-accent)] focus:ring-2 focus:ring-[color:var(--signal-accent)]/20 disabled:cursor-not-allowed disabled:opacity-50"
            />
            <span className="pointer-events-none absolute right-3 top-2.5 text-sm text-muted-foreground">%</span>
          </span>
        </label>
      </div>
    </article>
  );
};

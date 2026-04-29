import * as SwitchPrimitive from '@radix-ui/react-switch';
import { Pencil, Trash2 } from 'lucide-react';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { ConfirmDialog } from '@/components/ui/dialog';
import { cn } from '@/lib/cn';
import type { ScheduleRule, TriggerType } from '@/types/api';

import { useDeleteRule, useToggleRule } from '../hooks/use-rules';

type BadgeVariant = 'default' | 'warning' | 'secondary' | 'destructive';

const TYPE_VARIANT: Record<TriggerType, BadgeVariant> = {
  price_threshold: 'default',
  pct_change: 'warning',
  candle_pattern: 'secondary',
  funding_rate: 'destructive',
};

interface RowProps {
  rule: ScheduleRule;
  onEdit: (rule: ScheduleRule) => void;
}

const RuleRow = ({ rule, onEdit }: RowProps) => {
  const { t } = useTranslation('scheduler');
  const toggleMutation = useToggleRule();
  const deleteMutation = useDeleteRule();
  const [deleteOpen, setDeleteOpen] = useState(false);

  const lastTriggered = rule.last_triggered_at
    ? new Date(rule.last_triggered_at).toLocaleString()
    : t('table.never');

  return (
    <tr className="border-b border-border last:border-0">
      <td className="py-2.5 pl-3 pr-3 text-sm font-medium text-foreground">{rule.name}</td>
      <td className="py-2.5 pr-3">
        <Badge variant={TYPE_VARIANT[rule.trigger_type]}>{t(`trigger_type.${rule.trigger_type}`)}</Badge>
      </td>
      <td className="py-2.5 pr-3 font-mono text-xs text-muted-foreground">{rule.pair}</td>
      <td className="py-2.5 pr-3">
        <SwitchPrimitive.Root
          checked={rule.enabled}
          onCheckedChange={() => toggleMutation.mutate(rule.id)}
          disabled={toggleMutation.isPending}
          className={cn(
            'relative inline-flex h-5 w-9 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors',
            'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
            'disabled:cursor-not-allowed disabled:opacity-50',
            rule.enabled ? 'bg-primary' : 'bg-input',
          )}
          aria-label={t('table.enabled')}
        >
          <SwitchPrimitive.Thumb
            className={cn(
              'pointer-events-none block h-4 w-4 rounded-full bg-background shadow-lg ring-0 transition-transform',
              rule.enabled ? 'translate-x-4' : 'translate-x-0',
            )}
          />
        </SwitchPrimitive.Root>
      </td>
      <td className="py-2.5 pr-3">
        <span className={cn('text-xs', rule.in_cooldown ? 'text-warning' : 'text-success')}>
          {rule.in_cooldown ? t('table.in_cooldown') : t('table.ready')}
        </span>
      </td>
      <td className="py-2.5 pr-3 text-xs text-muted-foreground">{lastTriggered}</td>
      <td className="py-2.5 pr-3">
        <div className="flex items-center gap-1">
          <Button variant="ghost" size="icon" onClick={() => onEdit(rule)} aria-label={t('actions.edit')}>
            <Pencil className="h-3.5 w-3.5" />
          </Button>
          <Button variant="ghost" size="icon" onClick={() => setDeleteOpen(true)} aria-label={t('actions.delete')}>
            <Trash2 className="h-3.5 w-3.5 text-destructive" />
          </Button>
        </div>
        <ConfirmDialog
          open={deleteOpen}
          onOpenChange={setDeleteOpen}
          title={t('delete.confirm_title')}
          body={t('delete.confirm_body')}
          confirmLabel={t('delete.confirm_action')}
          destructive
          onConfirm={() => { void deleteMutation.mutateAsync(rule.id); }}
        />
      </td>
    </tr>
  );
};

interface Props {
  rules: ScheduleRule[];
  onEdit: (rule: ScheduleRule) => void;
}

export const RuleTable = ({ rules, onEdit }: Props) => {
  const { t } = useTranslation('scheduler');

  if (rules.length === 0) {
    return <p className="py-10 text-center text-sm text-muted-foreground">{t('table.empty')}</p>;
  }

  return (
    <div className="overflow-x-auto rounded-md border border-border">
      <table className="min-w-full">
        <thead>
          <tr className="border-b border-border bg-muted/40 text-xs text-muted-foreground">
            <th className="py-2.5 pl-3 pr-3 text-left font-medium">{t('table.name')}</th>
            <th className="py-2.5 pr-3 text-left font-medium">{t('table.type')}</th>
            <th className="py-2.5 pr-3 text-left font-medium">{t('table.pair')}</th>
            <th className="py-2.5 pr-3 text-left font-medium">{t('table.enabled')}</th>
            <th className="py-2.5 pr-3 text-left font-medium">{t('table.cooldown')}</th>
            <th className="py-2.5 pr-3 text-left font-medium">{t('table.last_triggered')}</th>
            <th className="py-2.5 pr-3 text-left font-medium">{t('table.actions')}</th>
          </tr>
        </thead>
        <tbody>
          {rules.map((rule) => (
            <RuleRow key={rule.id} rule={rule} onEdit={onEdit} />
          ))}
        </tbody>
      </table>
    </div>
  );
};

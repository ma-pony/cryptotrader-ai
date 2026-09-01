import * as Collapsible from '@radix-ui/react-collapsible';
import { ChevronDown, ChevronRight } from 'lucide-react';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';

import { Button } from '@/components/ui/button';
import type { PaginatedTriggerEvents, ScheduleRule, TriggerEvent } from '@/types/api';

interface EventRowProps {
  event: TriggerEvent;
  rule: ScheduleRule | undefined;
}

const EventRow = ({ event, rule }: EventRowProps) => {
  const { t } = useTranslation('scheduler');
  const [open, setOpen] = useState(false);

  return (
    <Collapsible.Root open={open} onOpenChange={setOpen}>
      <tr className="border-b border-border last:border-0">
        <td className="py-2.5 pl-3 pr-3 text-xs text-muted-foreground">
          {new Date(event.triggered_at).toLocaleString()}
        </td>
        <td className="py-2.5 pr-3 text-sm font-medium text-foreground">
          {rule?.name ?? event.rule_id}
        </td>
        <td className="py-2.5 pr-3 max-w-xs truncate text-xs text-muted-foreground" title={event.trigger_reason}>
          {event.trigger_reason}
        </td>
        <td className="py-2.5 pr-3 tabular-nums text-xs text-muted-foreground">{event.schedule_depth}</td>
        <td className="py-2.5 pr-3 font-mono text-xs text-muted-foreground">
          {event.analysis_commit_id ? event.analysis_commit_id.slice(0, 8) : '—'}
        </td>
        <td className="py-2.5 pr-3">
          <Collapsible.Trigger asChild>
            <Button variant="ghost" size="icon" aria-label={t('history.expand')}>
              {open ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
            </Button>
          </Collapsible.Trigger>
        </td>
      </tr>
      <Collapsible.Content asChild>
        <tr className="bg-muted/30">
          <td colSpan={6} className="px-3 py-3">
            <div className="space-y-1">
              <p className="text-xs font-medium text-muted-foreground">{t('history.price_snapshot')}</p>
              <pre className="overflow-x-auto rounded bg-muted p-2 text-xs">
                {JSON.stringify(event.price_snapshot, null, 2)}
              </pre>
            </div>
          </td>
        </tr>
      </Collapsible.Content>
    </Collapsible.Root>
  );
};

interface Props {
  data: PaginatedTriggerEvents | undefined;
  rules: ScheduleRule[];
  page: number;
  onPageChange: (page: number) => void;
}

export const TriggerHistoryTable = ({ data, rules, page, onPageChange }: Props) => {
  const { t } = useTranslation('scheduler');
  const ruleMap = new Map(rules.map((r) => [r.id, r]));

  if (!data || data.items.length === 0) {
    return <p className="py-10 text-center text-sm text-muted-foreground">{t('history.empty')}</p>;
  }

  return (
    <div className="space-y-3">
      <div className="overflow-x-auto rounded-md border border-border">
        <table className="min-w-full">
          <thead>
            <tr className="border-b border-border bg-muted/40 text-xs text-muted-foreground">
              <th className="py-2.5 pl-3 pr-3 text-left font-medium">{t('history.time')}</th>
              <th className="py-2.5 pr-3 text-left font-medium">{t('history.rule')}</th>
              <th className="py-2.5 pr-3 text-left font-medium">{t('history.reason')}</th>
              <th className="py-2.5 pr-3 text-left font-medium">{t('history.depth')}</th>
              <th className="py-2.5 pr-3 text-left font-medium">{t('history.commit')}</th>
              <th className="py-2.5 pr-3 text-left font-medium">{t('history.expand')}</th>
            </tr>
          </thead>
          <tbody>
            {data.items.map((event) => (
              <EventRow key={event.id} event={event} rule={ruleMap.get(event.rule_id)} />
            ))}
          </tbody>
        </table>
      </div>

      <div className="flex items-center justify-between">
        <p className="text-xs text-muted-foreground">
          {t('history.pagination.info', { page, total: data.total })}
        </p>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" disabled={page <= 1} onClick={() => onPageChange(page - 1)}>
            {t('history.pagination.prev')}
          </Button>
          <Button
            variant="outline"
            size="sm"
            disabled={page * data.size >= data.total}
            onClick={() => onPageChange(page + 1)}
          >
            {t('history.pagination.next')}
          </Button>
        </div>
      </div>
    </div>
  );
};

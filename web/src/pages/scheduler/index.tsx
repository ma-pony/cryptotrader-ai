import { Suspense, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { ErrorBoundary } from '@/components/error-boundary';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Skeleton } from '@/components/ui/skeleton';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import type { ScheduleRule } from '@/types/api';

import { useRules } from './hooks/use-rules';
import { useTriggerHistory } from './hooks/use-trigger-history';
import { RuleFormDialog } from './components/rule-form-dialog';
import { RuleTable } from './components/rule-table';
import { TemplateSelector } from './components/template-selector';
import { TriggerHistoryTable } from './components/trigger-history-table';
import type { RuleFormValues } from './components/template-selector';

const SchedulerContent = () => {
  const { t } = useTranslation('scheduler');
  const { data: rules, isLoading, isError, refetch } = useRules();
  const [historyPage, setHistoryPage] = useState(1);
  const { data: historyData } = useTriggerHistory(undefined, historyPage);

  const [formOpen, setFormOpen] = useState(false);
  const [editingRule, setEditingRule] = useState<ScheduleRule | undefined>();
  const [prefill, setPrefill] = useState<Partial<RuleFormValues> | undefined>();
  const [templateOpen, setTemplateOpen] = useState(false);

  const handleEdit = (rule: ScheduleRule) => {
    setEditingRule(rule);
    setPrefill(undefined);
    setFormOpen(true);
  };

  const handleCreate = () => {
    setEditingRule(undefined);
    setPrefill(undefined);
    setFormOpen(true);
  };

  const handleSelectTemplate = (data: Partial<RuleFormValues>) => {
    setTemplateOpen(false);
    setPrefill(data);
    setEditingRule(undefined);
    setFormOpen(true);
  };

  const handleFormClose = (open: boolean) => {
    setFormOpen(open);
    if (!open) {
      setEditingRule(undefined);
      setPrefill(undefined);
    }
  };

  if (isLoading) return <Skeleton className="h-64 w-full" />;
  if (isError) {
    return (
      <div className="py-10 text-center">
        <p className="text-sm text-destructive">{t('table.empty')}</p>
        <Button variant="ghost" size="sm" className="mt-2" onClick={() => void refetch()}>
          {t('actions.cancel')}
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold text-foreground">{t('title')}</h1>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={() => setTemplateOpen(true)}>
            {t('actions.from_template')}
          </Button>
          <Button size="sm" onClick={handleCreate}>
            {t('actions.create')}
          </Button>
        </div>
      </div>

      <Tabs defaultValue="rules">
        <TabsList>
          <TabsTrigger value="rules">{t('tabs.rules')}</TabsTrigger>
          <TabsTrigger value="history">{t('tabs.history')}</TabsTrigger>
        </TabsList>

        <TabsContent value="rules">
          <RuleTable rules={rules ?? []} onEdit={handleEdit} />
        </TabsContent>

        <TabsContent value="history">
          <TriggerHistoryTable
            data={historyData}
            rules={rules ?? []}
            page={historyPage}
            onPageChange={setHistoryPage}
          />
        </TabsContent>
      </Tabs>

      <RuleFormDialog
        open={formOpen}
        onOpenChange={handleFormClose}
        rule={editingRule}
        prefill={prefill}
      />

      <Dialog open={templateOpen} onOpenChange={setTemplateOpen}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>{t('templates.title')}</DialogTitle>
          </DialogHeader>
          <TemplateSelector onSelect={handleSelectTemplate} />
        </DialogContent>
      </Dialog>
    </div>
  );
};

const SchedulerPage = () => (
  <ErrorBoundary>
    <Suspense fallback={<Skeleton className="h-96 w-full" />}>
      <SchedulerContent />
    </Suspense>
  </ErrorBoundary>
);

export default SchedulerPage;

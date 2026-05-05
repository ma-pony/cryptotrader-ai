import { useState } from 'react';
import { useTranslation } from 'react-i18next';

import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { PageBoundary } from '@/components/ui/page-boundary';
import { PageHeader } from '@/components/ui/page-header';
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
  // SchedulerPage's PageBoundary owns isLoading / isError / refetch — this
  // hook here only reads `data` from the shared React Query cache.
  const { data: rules } = useRules();
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

  return (
    <div className="space-y-6">
      <PageHeader
        title={t('title')}
        actions={
          <>
            <Button variant="outline" size="sm" onClick={() => setTemplateOpen(true)}>
              {t('actions.from_template')}
            </Button>
            <Button size="sm" onClick={handleCreate}>
              {t('actions.create')}
            </Button>
          </>
        }
      />

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

const SchedulerPage = () => {
  const { t } = useTranslation('scheduler');
  const { isLoading, isError, refetch } = useRules();
  return (
    <PageBoundary
      loading={isLoading}
      isError={isError}
      onRetry={() => void refetch()}
      errorTitle={t('table.empty')}
    >
      <SchedulerContent />
    </PageBoundary>
  );
};

export default SchedulerPage;

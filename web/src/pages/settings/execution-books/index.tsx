import { Plus, Save } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Button } from '@/components/ui/button';
import { PageBoundary } from '@/components/ui/page-boundary';
import { PageHeader } from '@/components/ui/page-header';
import { useRuntimeConfig } from '@/hooks/use-runtime-config';
import { toRuntimeDocument } from '@/hooks/use-runtime-config';
import { AllocationPreview } from './allocation-preview';
import { BookForm, newBook, validateBooks } from './book-form';

const ExecutionBooksPage = () => {
  const { t } = useTranslation('configuration');
  const runtime = useRuntimeConfig();
  const document = runtime.document;
  const [books, setBooks] = useState<NonNullable<typeof runtime.document>['execution']['books']>([]);
  const [equity, setEquity] = useState(100000);
  const [exposure, setExposure] = useState(0.5);
  const [saveError, setSaveError] = useState('');
  const hydrated = useRef(false);
  useEffect(() => {
    if (document && !hydrated.current) { setBooks(document.execution.books); hydrated.current = true; }
  }, [document]);
  const errors = document ? validateBooks(books, document.execution.connections) : [];
  const save = async () => {
    if (!document) return;
    try { setSaveError(''); await runtime.replace({ ...document, execution: { ...document.execution, books } }); }
    catch { setSaveError(t('saveFailed')); }
  };
  const reload = async () => {
    const result = await runtime.reload();
    if (result.isSuccess && !result.error && result.data) setBooks(toRuntimeDocument(result.data.document).execution.books);
  };
  return (
    <PageBoundary
      loading={runtime.isLoading}
      isError={runtime.isError}
      onRetry={() => void reload()}
      errorTitle={t('bookLoadError')}
      errorDescription={t('bookLoadDescription')}
    >
      {document ? (
        <div className="space-y-6">
          <PageHeader
            eyebrow="CAPITAL ROUTING"
            title={t('books')}
            subtitle={t('booksSubtitle')}
            actions={<span className="font-mono text-xs text-amber-500">{t('revisionValue', { revision: runtime.revision })}</span>}
          />
          <div className="grid gap-4">
            {(['simulated', 'real'] as const).map((scope) => (
              <section key={scope} className="rounded-2xl border border-border bg-card p-5">
                <h2 className="font-semibold">{scope === 'simulated' ? t('simulatedBooks') : t('realBooks')}</h2>
                <p className="mt-1 text-sm text-muted-foreground">
                  {scope === 'simulated' ? t('simulatedDescription') : t('realDescription')}
                </p>
                <div className="mt-4 space-y-3">
                  {books.map((book, index) =>
                    book.capital_scope === scope ? (
                      <div key={`${book.id}-${index}`}>
                        <BookForm
                          book={book}
                          connections={document.execution.connections}
                          onChange={(next) =>
                            setBooks((current) => current.map((item, itemIndex) => (itemIndex === index ? next : item)))
                          }
                          onRemove={() => setBooks((current) => current.filter((_, itemIndex) => itemIndex !== index))}
                        />
                        <AllocationPreview
                          equity={equity}
                          targetExposure={exposure}
                          allocations={book.allocations
                            .filter((item) => item.enabled)
                            .map((item) => ({
                              connectionId: item.connection_id,
                              label:
                                document.execution.connections.find(
                                  (connection) => connection.id === item.connection_id,
                                )?.label ?? item.connection_id,
                              weight: item.weight * 100,
                            }))}
                        />
                      </div>
                    ) : null,
                  )}
                </div>
              </section>
            ))}
          </div>
          <section className="rounded-2xl border border-border bg-card p-5">
            <h2 className="font-semibold">{t('allocationPreview')}</h2>
            <div className="mt-3 flex flex-wrap gap-3">
              <label>
                {t('sampleEquity')}
                <input
                  aria-label={t('sampleEquity')}
                  type="number"
                  value={equity}
                  onChange={(event) => setEquity(Number(event.target.value))}
                  className="ml-2 h-9 rounded border bg-background px-2"
                />
              </label>
              <label>
                {t('targetExposure')}
                <input
                  aria-label={t('targetExposure')}
                  type="number"
                  step="0.1"
                  value={exposure}
                  onChange={(event) => setExposure(Number(event.target.value))}
                  className="ml-2 h-9 rounded border bg-background px-2"
                />
              </label>
            </div>
          </section>
          <Button variant="outline" onClick={() => setBooks((current) => [...current, newBook()])}>
            <Plus className="h-4 w-4" />
            {t('addBook')}
          </Button>
          <Button variant="outline" onClick={() => void reload()}>
            {t('reload')}
          </Button>
          {errors.map((error, index) => (
            <p key={`${error.code}-${index}`} role="alert" className="text-sm text-trade-short">
              {t(`book.errors.${error.code}`, error.params ?? {})}
            </p>
          ))}
          {runtime.conflict ? <div className="flex gap-2"><p role="alert" className="text-sm text-trade-short">{t('conflict')}</p><Button variant="outline" onClick={() => void reload()}>{t('reload')}</Button></div> : null}
          {saveError ? <p role="alert" className="text-sm text-trade-short">{saveError}</p> : null}
          <Button disabled={runtime.isSaving || runtime.conflict || errors.length > 0} onClick={() => void save()}>
            <Save className="h-4 w-4" />
            {t('save')}
          </Button>
        </div>
      ) : null}
    </PageBoundary>
  );
};
export default ExecutionBooksPage;

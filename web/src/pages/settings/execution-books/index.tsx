import { Plus, Save } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import { PageBoundary } from '@/components/ui/page-boundary';
import { PageHeader } from '@/components/ui/page-header';
import { useRuntimeConfig } from '@/hooks/use-runtime-config';
import { toRuntimeDocument } from '@/hooks/use-runtime-config';
import { AllocationPreview } from './allocation-preview';
import { BookForm, newBook, validateBooks } from './book-form';

const ExecutionBooksPage = () => {
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
    catch { setSaveError('保存资金池失败，请重新加载后重试。'); }
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
      errorTitle="无法读取资金池配置"
      errorDescription="请检查配置服务后重试。"
    >
      {document ? (
        <div className="space-y-6">
          <PageHeader
            eyebrow="CAPITAL ROUTING"
            title="执行资金池"
            subtitle="固定权重将一个目标敞口按作用域分配到多个连接。"
            actions={<span className="font-mono text-xs text-amber-500">Revision {runtime.revision}</span>}
          />
          <div className="grid gap-4">
            {(['simulated', 'real'] as const).map((scope) => (
              <section key={scope} className="rounded-2xl border border-border bg-card p-5">
                <h2 className="font-semibold">{scope === 'simulated' ? '模拟资金池' : '实盘资金池'}</h2>
                <p className="mt-1 text-sm text-muted-foreground">
                  {scope === 'simulated' ? 'paper / demo / testnet' : '仅 live'}{' '}
                  连接；每个连接只能归属于一个启用资金池。
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
            <h2 className="font-semibold">分配预览</h2>
            <div className="mt-3 flex flex-wrap gap-3">
              <label>
                示例权益
                <input
                  aria-label="示例权益"
                  type="number"
                  value={equity}
                  onChange={(event) => setEquity(Number(event.target.value))}
                  className="ml-2 h-9 rounded border bg-background px-2"
                />
              </label>
              <label>
                目标敞口
                <input
                  aria-label="目标敞口"
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
            新增资金池
          </Button>
          <Button variant="outline" onClick={() => void reload()}>
            重新加载
          </Button>
          {errors.map((error) => (
            <p key={error} role="alert" className="text-sm text-trade-short">
              {error}
            </p>
          ))}
          {runtime.conflict ? <div className="flex gap-2"><p role="alert" className="text-sm text-trade-short">配置已被其他操作更新，请重新加载</p><Button variant="outline" onClick={() => void reload()}>重新加载</Button></div> : null}
          {saveError ? <p role="alert" className="text-sm text-trade-short">{saveError}</p> : null}
          <Button disabled={runtime.isSaving || errors.length > 0} onClick={() => void save()}>
            <Save className="h-4 w-4" />
            保存完整配置
          </Button>
        </div>
      ) : null}
    </PageBoundary>
  );
};
export default ExecutionBooksPage;

import { CheckCircle2, Plus } from 'lucide-react';
import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { PageBoundary } from '@/components/ui/page-boundary';
import { PageHeader } from '@/components/ui/page-header';
import { useRuntimeConfig } from '@/hooks/use-runtime-config';
import { VenueForm } from './venue-form';

const VenuesPage = ({ onTested }: { onTested?: (id: string) => void }) => {
  const runtime = useRuntimeConfig();
  const [adding, setAdding] = useState(false);
  const [editorGeneration, setEditorGeneration] = useState(0);
  const reload = async () => {
    const result = await runtime.reload();
    if (result.isSuccess && !result.error && result.data) setEditorGeneration((current) => current + 1);
  };
  return (
    <PageBoundary
      loading={runtime.isLoading}
      isError={runtime.isError}
      onRetry={() => void reload()}
      errorTitle="无法读取连接配置"
      errorDescription="请检查配置服务后重试。"
    >
      {runtime.document && runtime.revision !== undefined ? (
        <div className="space-y-6">
          <PageHeader
            eyebrow="VENUE CONTROL"
            title="平台连接"
            subtitle="连接环境创建后固定；凭据只保留在当前输入框，永不回显。"
            actions={<span className="font-mono text-xs text-amber-500">Revision {runtime.revision}</span>}
          />
          <div className="grid gap-4">
            {runtime.document.execution.connections.map((connection) => (
              <section key={connection.id} className="rounded-2xl border border-border bg-card p-5">
                <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <h2 className="font-semibold">{connection.label}</h2>
                    <p className="font-mono text-xs text-muted-foreground">
                      {connection.adapter_id} · {connection.id}
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="rounded-full border border-amber-500/30 px-2 py-1 text-xs text-amber-500">
                      {connection.environment}
                    </span>
                    {runtime.credentialStates[connection.id]?.configured ? (
                      <span className="flex items-center gap-1 text-xs text-trade-long">
                        <CheckCircle2 className="h-3 w-3" />
                        凭据已配置
                      </span>
                    ) : null}
                  </div>
                </div>
                <VenueForm
                  key={`${connection.id}-${editorGeneration}`}
                  revision={runtime.revision ?? 0}
                  connection={connection}
                  onSaved={() => void runtime.reload()}
                  {...(onTested ? { tested: onTested } : {})}
                />
              </section>
            ))}
          </div>
          {adding ? (
            <section className="rounded-2xl border border-dashed border-amber-500/40 bg-card p-5">
              <h2 className="mb-4 font-semibold">新增平台连接</h2>
              <VenueForm
                key={`new-${editorGeneration}`}
                revision={runtime.revision}
                onSaved={() => {
                  setAdding(false);
                  void runtime.reload();
                }}
                {...(onTested ? { tested: onTested } : {})}
              />
            </section>
          ) : (
            <Button variant="outline" onClick={() => setAdding(true)}>
              <Plus className="h-4 w-4" />
              新增连接
            </Button>
          )}
          <Button variant="outline" onClick={() => void reload()}>
            重新加载
          </Button>
          {runtime.conflict ? (
            <p role="alert" className="text-sm text-trade-short">
              配置已被其他操作更新，请重新加载
            </p>
          ) : null}
        </div>
      ) : null}
    </PageBoundary>
  );
};
export default VenuesPage;

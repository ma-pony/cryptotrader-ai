import { ArrowRight, CheckCircle2, Save } from 'lucide-react';
import { useEffect, useState } from 'react';
import { Button } from '@/components/ui/button';
import { useRuntimeConfig } from '@/hooks/use-runtime-config';
import { VenueForm } from '@/pages/settings/venues/venue-form';
import { BookForm, newBook, validateBooks } from '@/pages/settings/execution-books/book-form';

const STEPS = ['LLM', '信号组件', '行情来源', '平台连接', '执行资金池', '风控与审批', '调度器', '测试并激活'];
const SetupPage = () => {
  const runtime = useRuntimeConfig(); const [step, setStep] = useState(0); const [tested, setTested] = useState<Set<string>>(() => new Set()); const [books, setBooks] = useState<NonNullable<typeof runtime.document>['execution']['books']>([]);
  useEffect(() => { if (runtime.document) setBooks(runtime.document.execution.books); }, [runtime.document]);
  if (runtime.isLoading) return <div className="grid min-h-screen place-items-center font-mono text-amber-500">LOADING CONFIG…</div>;
  if (runtime.isError || !runtime.document || runtime.revision === undefined) return <main className="grid min-h-screen place-items-center"><div><h1>无法加载初始化配置</h1><Button onClick={() => void runtime.reload()}>重试</Button></div></main>;
  const document = runtime.document; const enabledComponents = document.signals.components.some((item) => item.enabled); const enabledBooks = books.filter((item) => item.enabled); const allocationErrors = validateBooks(books, document.execution.connections); const allEnabledConnectionsTested = enabledBooks.flatMap((book) => book.allocations.filter((allocation) => allocation.enabled).map((allocation) => allocation.connection_id)).every((id) => tested.has(id));
  const saveBooks = () => void runtime.replace({ ...document, execution: { ...document.execution, books } });
  const activate = () => void runtime.replace({ ...document, execution: { ...document.execution, books }, system: { ...document.system, active: true } });
  const stageContent = () => {
    if (step === 0) return <p className="text-sm text-muted-foreground">LLM 网关和模型配置由“信号组件”页的委员会模型字段写入同一 RuntimeConfig。</p>;
    if (step === 1) return <p className="text-sm text-muted-foreground">至少启用 Kronos、LLM 委员会或一个已安装的自定义组件，并在策略页完成 100% 信任权重。</p>;
    if (step === 2) return <p className="text-sm text-muted-foreground">行情来源是可插拔组件 ID；生产运行时将用数据库参数构造该来源。</p>;
    if (step === 3) return <div className="space-y-4"><p className="text-sm text-muted-foreground">创建或编辑连接；测试成功只记入本次初始化流程。</p>{document.execution.connections.map((connection) => <VenueForm key={connection.id} revision={runtime.revision ?? 0} connection={connection} onSaved={() => void runtime.reload()} tested={(id) => setTested((current) => new Set(current).add(id))}/>) }<VenueForm revision={runtime.revision ?? 0} onSaved={() => void runtime.reload()} /></div>;
    if (step === 4) return <div className="space-y-3">{books.map((book, index) => <BookForm key={`${book.id}-${index}`} book={book} connections={document.execution.connections} onChange={(next) => setBooks((current) => current.map((item, itemIndex) => itemIndex === index ? next : item))} onRemove={() => setBooks((current) => current.filter((_, itemIndex) => itemIndex !== index))}/>) }<Button variant="outline" onClick={() => setBooks((current) => [...current, newBook()])}>新增资金池</Button>{allocationErrors.map((error) => <p key={error} role="alert" className="text-sm text-trade-short">{error}</p>)}<Button disabled={allocationErrors.length > 0 || runtime.isSaving} onClick={saveBooks}><Save className="h-4 w-4"/>保存资金池</Button></div>;
    if (step === 5) return <p className="text-sm text-muted-foreground">每个资金池可单独开启 HITL；风险阈值和审批 TTL 都在完整 RuntimeConfig 内。</p>;
    if (step === 6) return <p className="text-sm text-muted-foreground">调度器默认保持暂停，激活后再按业务需要启用。</p>;
    return <div className="space-y-3"><p className={enabledComponents && enabledBooks.length > 0 && allocationErrors.length === 0 && allEnabledConnectionsTested ? 'text-trade-long' : 'text-trade-short'}>{enabledComponents && enabledBooks.length > 0 && allocationErrors.length === 0 && allEnabledConnectionsTested ? '所有激活条件已满足。' : '需有启用组件、合法 100% 资金池，并在本次向导测试所有启用连接。'}</p><Button disabled={!enabledComponents || enabledBooks.length === 0 || allocationErrors.length > 0 || !allEnabledConnectionsTested || runtime.isSaving} onClick={activate}><CheckCircle2 className="h-4 w-4"/>测试并激活</Button></div>;
  };
  return <main className="min-h-screen bg-background p-6 text-foreground"><div className="mx-auto max-w-5xl"><header className="border-b border-amber-500/30 pb-6"><p className="font-mono text-xs tracking-[.24em] text-amber-500">COMMISSIONING / REV {runtime.revision}</p><h1 className="mt-3 text-3xl font-semibold">初始化交易系统</h1><p className="mt-2 text-muted-foreground">依次完成八个 commissioning 阶段；激活后才进入操作台。</p></header><div className="mt-8 grid gap-6 lg:grid-cols-[230px_1fr]"><ol className="border-l border-amber-500/30">{STEPS.map((label, index) => <li key={label} className={`relative py-3 pl-5 text-sm ${index === step ? 'font-semibold text-amber-500' : index < step ? 'text-trade-long' : 'text-muted-foreground'}`}><span className="absolute -left-1.5 top-4 h-3 w-3 rounded-full bg-current"/>{index + 1}. {label}</li>)}</ol><section className="rounded-2xl border border-border bg-card p-6"><p className="font-mono text-xs text-amber-500">STAGE 0{step + 1}</p><h2 className="mt-2 text-xl font-semibold">{STEPS[step]}</h2><div className="mt-4">{stageContent()}</div>{step < 7 ? <Button className="mt-6" onClick={() => setStep((current) => Math.min(current + 1, 7))}>下一阶段 <ArrowRight className="h-4 w-4"/></Button> : null}{runtime.conflict ? <p role="alert" className="mt-4 text-sm text-trade-short">配置已被其他操作更新，请重新加载</p> : null}</section></div></div></main>;
};
export default SetupPage;

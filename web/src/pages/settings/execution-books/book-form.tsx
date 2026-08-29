import { Trash2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import type { RuntimeDocument } from '@/types/api';

type Book = RuntimeDocument['execution']['books'][number];
type Connection = RuntimeDocument['execution']['connections'][number];
const eligible = (scope: Book['capital_scope'], environment: Connection['environment']) => scope === 'simulated' ? environment !== 'live' : environment === 'live';

export const validateBooks = (books: Book[], connections: Connection[]) => {
  const used = new Set<string>(); const errors: string[] = [];
  for (const book of books.filter((item) => item.enabled)) {
    const enabled = book.allocations.filter((item) => item.enabled);
    if (enabled.length === 0 || Math.abs(enabled.reduce((sum, item) => sum + item.weight, 0) - 1) > 1e-9) errors.push(`${book.label} 的启用 allocation 权重必须精确合计 100%`);
    for (const allocation of enabled) { const connection = connections.find((item) => item.id === allocation.connection_id); if (!connection || !eligible(book.capital_scope, connection.environment)) errors.push(book.capital_scope === 'simulated' ? '模拟资金池不能包含实盘连接' : '实盘资金池只能包含 live 连接'); if (used.has(allocation.connection_id)) errors.push('同一启用连接不能进入多个启用资金池'); used.add(allocation.connection_id); }
  }
  return errors;
};

export const BookForm = ({ book, connections, onChange, onRemove }: { book: Book; connections: Connection[]; onChange: (book: Book) => void; onRemove: () => void }) => {
  const allowed = connections.filter((connection) => eligible(book.capital_scope, connection.environment));
  const allocationFor = (id: string) => book.allocations.find((allocation) => allocation.connection_id === id) ?? { connection_id: id, enabled: false, weight: 0 };
  const setAllocation = (id: string, update: Partial<Book['allocations'][number]>) => onChange({ ...book, allocations: book.allocations.some((item) => item.connection_id === id) ? book.allocations.map((item) => item.connection_id === id ? { ...item, ...update } : item) : [...book.allocations, { ...allocationFor(id), ...update }] });
  return <article className="rounded-xl border border-border bg-muted/10 p-4"><div className="grid gap-3 md:grid-cols-4"><label className="text-xs text-muted-foreground">资金池 ID<input aria-label="资金池 ID" value={book.id} onChange={(event) => onChange({ ...book, id: event.target.value })} className="mt-1 h-10 w-full rounded border bg-background px-3"/></label><label className="text-xs text-muted-foreground">名称<input aria-label="资金池名称" value={book.label} onChange={(event) => onChange({ ...book, label: event.target.value })} className="mt-1 h-10 w-full rounded border bg-background px-3"/></label><label className="text-xs text-muted-foreground">资金作用域<select aria-label="资金作用域" value={book.capital_scope} onChange={(event) => onChange({ ...book, capital_scope: event.target.value as Book['capital_scope'], allocations: [] })} className="mt-1 h-10 w-full rounded border bg-background px-3"><option value="simulated">simulated</option><option value="real">real</option></select></label><label className="flex items-center gap-2 pt-5 text-sm"><input type="checkbox" checked={book.hitl_required} onChange={(event) => onChange({ ...book, hitl_required: event.target.checked })}/>HITL 审批</label></div><div className="mt-4 space-y-2 border-t border-border pt-3">{allowed.map((connection) => { const allocation = allocationFor(connection.id); return <div key={connection.id} className="flex items-center gap-3 text-sm"><input aria-label={`${connection.label} 启用`} type="checkbox" checked={allocation.enabled} onChange={(event) => setAllocation(connection.id, { enabled: event.target.checked, weight: event.target.checked && allocation.weight === 0 ? 1 : allocation.weight })}/><span className="min-w-36">{connection.label} <small className="text-muted-foreground">{connection.environment}</small></span><input aria-label={`${connection.label} 权重`} disabled={!allocation.enabled} type="number" min="0" max="100" value={Math.round(allocation.weight * 100)} onChange={(event) => setAllocation(connection.id, { weight: Number(event.target.value) / 100 })} className="h-9 w-24 rounded border bg-background px-2 font-mono"/><span className="text-xs text-muted-foreground">%</span></div>; })}{allowed.length === 0 ? <p className="text-sm text-muted-foreground">当前作用域没有可用连接。</p> : null}</div><div className="mt-4 flex items-center justify-between"><label className="flex gap-2 text-sm"><input type="checkbox" checked={book.enabled} onChange={(event) => onChange({ ...book, enabled: event.target.checked })}/>启用资金池</label><Button type="button" variant="ghost" onClick={onRemove}><Trash2 className="h-4 w-4"/>删除</Button></div></article>;
};
export const newBook = (): Book => ({ id: '', label: '', capital_scope: 'simulated', enabled: true, hitl_required: true, allocations: [] });

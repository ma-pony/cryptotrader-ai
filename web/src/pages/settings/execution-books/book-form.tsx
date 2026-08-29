import { Trash2 } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { Button } from '@/components/ui/button';
import type { RuntimeDocument } from '@/types/api';

type Book = RuntimeDocument['execution']['books'][number];
type Connection = RuntimeDocument['execution']['connections'][number];
const eligible = (scope: Book['capital_scope'], environment: Connection['environment']) =>
  scope === 'simulated' ? environment !== 'live' : environment === 'live';

export type BookValidationError = { code: string; params?: Record<string, string> };

export const validateBooks = (books: Book[], connections: Connection[]): BookValidationError[] => {
  const used = new Set<string>();
  const errors: BookValidationError[] = [];
  const bookIds = new Set<string>();
  for (const book of books) {
    if (!book.id.trim() || !book.label.trim()) errors.push({ code: 'required' });
    if (bookIds.has(book.id)) errors.push({ code: 'duplicateBook' });
    bookIds.add(book.id);
    const allAllocations = new Set<string>();
    for (const allocation of book.allocations) {
      if (allAllocations.has(allocation.connection_id)) errors.push({ code: 'duplicateAllocation' });
      allAllocations.add(allocation.connection_id);
    }
    if (!book.enabled) continue;
    const enabled = book.allocations.filter((item) => item.enabled);
    if (enabled.length === 0 || Math.abs(enabled.reduce((sum, item) => sum + item.weight, 0) - 1) > 1e-9)
      errors.push({ code: 'weightTotal', params: { label: book.label } });
    const local = new Set<string>();
    for (const allocation of enabled) {
      const connection = connections.find((item) => item.id === allocation.connection_id);
      if (!connection || !connection.enabled || !eligible(book.capital_scope, connection.environment))
        errors.push({ code: book.capital_scope === 'simulated' ? 'invalidSimulated' : 'invalidReal' });
      if (local.has(allocation.connection_id)) errors.push({ code: 'duplicateAllocation' });
      local.add(allocation.connection_id);
      if (used.has(allocation.connection_id)) errors.push({ code: 'duplicateConnection' });
      used.add(allocation.connection_id);
    }
  }
  return errors;
};

export const BookForm = ({
  book,
  connections,
  onChange,
  onRemove,
}: {
  book: Book;
  connections: Connection[];
  onChange: (book: Book) => void;
  onRemove: () => void;
}) => {
  const { t } = useTranslation('configuration');
  const allowed = connections.filter((connection) => eligible(book.capital_scope, connection.environment));
  const allocationFor = (id: string) =>
    book.allocations.find((allocation) => allocation.connection_id === id) ?? {
      connection_id: id,
      enabled: false,
      weight: 0,
    };
  const setAllocation = (id: string, update: Partial<Book['allocations'][number]>) =>
    onChange({
      ...book,
      allocations: book.allocations.some((item) => item.connection_id === id)
        ? book.allocations.map((item) => (item.connection_id === id ? { ...item, ...update } : item))
        : [...book.allocations, { ...allocationFor(id), ...update }],
    });
  return (
    <article className="rounded-xl border border-border bg-muted/10 p-4">
      <div className="grid gap-3 md:grid-cols-4">
        <label className="text-xs text-muted-foreground">
          {t('book.id')}
          <input
            aria-label={t('book.id')}
            value={book.id}
            onChange={(event) => onChange({ ...book, id: event.target.value })}
            className="mt-1 h-10 w-full rounded border bg-background px-3"
          />
        </label>
        <label className="text-xs text-muted-foreground">
          {t('book.name')}
          <input
            aria-label={t('book.name')}
            value={book.label}
            onChange={(event) => onChange({ ...book, label: event.target.value })}
            className="mt-1 h-10 w-full rounded border bg-background px-3"
          />
        </label>
        <label className="text-xs text-muted-foreground">
          {t('book.scope')}
          <select
            aria-label={t('book.scope')}
            value={book.capital_scope}
            onChange={(event) =>
              onChange({ ...book, capital_scope: event.target.value as Book['capital_scope'], allocations: [] })
            }
            className="mt-1 h-10 w-full rounded border bg-background px-3"
          >
            <option value="simulated">simulated</option>
            <option value="real">real</option>
          </select>
        </label>
        <label className="flex items-center gap-2 pt-5 text-sm">
          <input
            type="checkbox"
            checked={book.hitl_required}
            onChange={(event) => onChange({ ...book, hitl_required: event.target.checked })}
          />
          {t('book.hitl')}
        </label>
      </div>
      <div className="mt-4 space-y-2 border-t border-border pt-3">
        {allowed.map((connection) => {
          const allocation = allocationFor(connection.id);
          return (
            <div key={connection.id} className="flex items-center gap-3 text-sm">
              <input
                aria-label={t('book.enabledConnection', { name: connection.label })}
                type="checkbox"
                checked={allocation.enabled}
                onChange={(event) =>
                  setAllocation(connection.id, {
                    enabled: event.target.checked,
                    weight: event.target.checked && allocation.weight === 0 ? 1 : allocation.weight,
                  })
                }
              />
              <span className="min-w-36">
                {connection.label} <small className="text-muted-foreground">{connection.environment}</small>
              </span>
              <input
                aria-label={t('book.weight', { name: connection.label })}
                disabled={!allocation.enabled}
                type="number"
                min="0"
                max="100"
                value={Math.round(allocation.weight * 100)}
                onChange={(event) => setAllocation(connection.id, { weight: Number(event.target.value) / 100 })}
                className="h-9 w-24 rounded border bg-background px-2 font-mono"
              />
              <span className="text-xs text-muted-foreground">%</span>
            </div>
          );
        })}
        {allowed.length === 0 ? <p className="text-sm text-muted-foreground">{t('book.noConnections')}</p> : null}
      </div>
      <div className="mt-4 flex items-center justify-between">
        <label className="flex gap-2 text-sm">
          <input
            type="checkbox"
            checked={book.enabled}
            onChange={(event) => onChange({ ...book, enabled: event.target.checked })}
          />
          {t('book.enabled')}
        </label>
        <Button type="button" variant="ghost" onClick={onRemove}>
          <Trash2 className="h-4 w-4" />
          {t('book.delete')}
        </Button>
      </div>
    </article>
  );
};
export const newBook = (): Book => ({
  id: '',
  label: '',
  capital_scope: 'simulated',
  enabled: true,
  hitl_required: true,
  allocations: [],
});

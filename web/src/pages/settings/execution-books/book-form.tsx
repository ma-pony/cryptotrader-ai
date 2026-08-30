import { useTranslation } from 'react-i18next';
import { BooleanField, ChoiceField, NumberField, TextField, type FieldErrors } from '@/components/configuration/field';
import { AdvancedSection } from '@/components/configuration/section';
import { eligibleConnection, type BookDraft, type Connection } from '@/lib/configuration-readiness';

export const newBook = (): BookDraft => ({
  id: 'book-' + crypto.randomUUID(),
  label: '',
  capital_scope: 'simulated',
  enabled: true,
  hitl_required: true,
  allocations: [],
});
export function BookForm({
  book,
  index,
  saved,
  connections,
  errors,
  onChange,
  onRemove,
}: {
  book: BookDraft;
  index: number;
  saved: boolean;
  connections: Connection[];
  errors: FieldErrors;
  onChange: (book: BookDraft) => void;
  onRemove: () => void;
}) {
  const { t } = useTranslation('configuration');
  const prefix = 'execution.books.' + index;
  const allowed = connections.filter((connection) => eligibleConnection(book.capital_scope, connection));
  const total = book.allocations
    .filter((item) => item.enabled)
    .reduce((sum, item) => sum + (typeof item.weight === 'number' ? item.weight * 100 : 0), 0);
  const setAllocation = (id: string, update: Partial<BookDraft['allocations'][number]>) =>
    onChange({
      ...book,
      allocations: book.allocations.some((item) => item.connection_id === id)
        ? book.allocations.map((item) => (item.connection_id === id ? { ...item, ...update } : item))
        : [...book.allocations, { connection_id: id, enabled: false, weight: 0, ...update }],
    });
  const unavailable = book.allocations.filter(
    (item) => !allowed.some((connection) => connection.id === item.connection_id),
  );
  return (
    <article className="configuration-section">
      <div className="configuration-grid">
        <TextField
          name={prefix + '.label'}
          label={t('book.name')}
          required
          value={book.label}
          error={errors[prefix + '.label']}
          onChange={(label) => onChange({ ...book, label })}
        />
        <ChoiceField
          name={prefix + '.capital_scope'}
          label={t('book.scope')}
          help={t('book.scopeHelp')}
          disabled={saved}
          value={book.capital_scope}
          options={[
            { value: 'simulated', label: t('simulatedBooks') },
            { value: 'real', label: t('realBooks') },
          ]}
          onChange={(scope) =>
            onChange({ ...book, capital_scope: scope as BookDraft['capital_scope'], allocations: [] })
          }
        />
        <BooleanField
          name={prefix + '.enabled'}
          label={t('book.enabled')}
          value={book.enabled}
          error={errors[prefix + '.enabled']}
          onChange={(enabled) => onChange({ ...book, enabled })}
        />
        <BooleanField
          name={prefix + '.hitl_required'}
          label={t('book.hitl')}
          value={book.hitl_required}
          help={t('forms.hitlHelp')}
          onChange={(hitl_required) => onChange({ ...book, hitl_required })}
        />
      </div>
      <AdvancedSection title={t('book.advanced')}>
        <TextField
          name={prefix + '.id'}
          label={t('book.id')}
          help={t('book.idHelp')}
          disabled={saved}
          required
          value={book.id}
          error={errors[prefix + '.id']}
          onChange={(id) => onChange({ ...book, id })}
        />
      </AdvancedSection>
      <div className="space-y-3">
        {allowed.map((connection) => {
          const row = book.allocations.findIndex((item) => item.connection_id === connection.id);
          const allocation = book.allocations[row];
          return (
            <div key={connection.id} className="configuration-grid">
              <BooleanField
                name={prefix + '.connection.' + connection.id}
                label={t('book.enabledConnection', { name: connection.label })}
                value={allocation?.enabled ?? false}
                onChange={(enabled) =>
                  setAllocation(connection.id, {
                    enabled,
                    weight: enabled && !allocation?.weight ? 1 : (allocation?.weight ?? 0),
                  })
                }
              />
              <NumberField
                id={prefix + '.connection.' + connection.id + '.weight'}
                name={
                  row < 0
                    ? prefix + '.connection.' + connection.id + '.weight'
                    : prefix + '.allocations.' + row + '.weight'
                }
                label={t('book.weight', { name: connection.label })}
                percent
                min={0}
                max={100}
                disabled={!allocation?.enabled}
                value={allocation?.weight ?? 0}
                error={errors[prefix + '.allocations.' + row + '.weight']}
                onChange={(weight) => setAllocation(connection.id, { weight })}
              />
            </div>
          );
        })}
        {unavailable.map((allocation) => (
          <div key={allocation.connection_id} className="configuration-inline">
            <p className="configuration-error">{t('book.unavailable', { name: allocation.connection_id })}</p>
            <button
              className="configuration-button"
              type="button"
              onClick={() => onChange({ ...book, allocations: book.allocations.filter((item) => item !== allocation) })}
            >
              {t('forms.remove')}
            </button>
          </div>
        ))}
        {!allowed.length ? <p className="configuration-help">{t('book.noConnections')}</p> : null}
        <p className="configuration-help" aria-live="polite">
          {t('forms.weightTotal', {
            total: Number(total.toPrecision(12)),
            remaining: Number((100 - total).toPrecision(12)),
          })}
        </p>
      </div>
      <button
        type="button"
        className="configuration-button"
        onClick={() => {
          if (!saved || window.confirm(t('book.removeConfirm'))) onRemove();
        }}
      >
        {t('forms.remove')}
      </button>
    </article>
  );
}

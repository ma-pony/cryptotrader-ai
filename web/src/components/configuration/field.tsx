import { useId, useState, type ReactNode } from 'react';
import { useTranslation } from 'react-i18next';
import type { ConfigurationDraft } from '@/types/api';
import { useRuntimeSecrets } from '@/hooks/use-runtime-secrets';
import { useRuntimeConfigConflict } from '@/hooks/runtime-config-conflict';
import { ApiError } from '@/lib/api-client';

export type FieldErrors = Record<string, string>;
export type DomainFormProps<T> = {
  value: ConfigurationDraft<T>;
  onChange: (value: ConfigurationDraft<T>) => void;
  errors?: FieldErrors;
};
export const percentToRatio = (percent: number) => percent / 100;

type FieldProps = {
  name: string;
  id?: string;
  label: string;
  help?: string | undefined;
  error?: string | undefined;
  children: ReactNode;
};
export function Field({ name, id = name, label, help, error, children }: FieldProps) {
  return (
    <div className="configuration-field">
      <label htmlFor={id}>{label}</label>
      {children}
      <p id={`${id}-help`} className={error ? 'configuration-error' : 'configuration-help'}>
        {error || help}
      </p>
    </div>
  );
}
type CommonFieldProps = {
  name: string;
  label: string;
  help?: string | undefined;
  error?: string | undefined;
  required?: boolean | undefined;
  disabled?: boolean | undefined;
};
function attributes({ name, error, required, disabled }: CommonFieldProps) {
  return {
    id: name,
    name,
    'aria-describedby': `${name}-help`,
    'aria-invalid': Boolean(error),
    'aria-required': required,
    required,
    disabled,
    className: 'configuration-control',
  };
}
export function TextField(
  props: CommonFieldProps & { value: string; onChange: (value: string) => void; type?: 'text' | 'url' | 'password' },
) {
  return (
    <Field {...props}>
      <input
        {...attributes(props)}
        type={props.type ?? 'text'}
        value={props.value}
        autoComplete={props.type === 'password' ? 'new-password' : 'off'}
        onChange={(event) => props.onChange(event.target.value)}
      />
    </Field>
  );
}
export function NumberField(
  props: CommonFieldProps & {
    value: number | '';
    id?: string;
    onChange: (value: number | '') => void;
    min?: number;
    max?: number;
    step?: number;
    percent?: boolean;
  },
) {
  const shown = props.value === '' ? '' : props.percent ? Number((props.value * 100).toPrecision(12)) : props.value;
  return (
    <Field {...props}>
      <input
        {...attributes({ ...props, required: props.required ?? true })}
        id={props.id ?? props.name}
        aria-describedby={`${props.id ?? props.name}-help`}
        type="number"
        min={props.min}
        max={props.max}
        step={props.step ?? 'any'}
        value={shown}
        onChange={(event) => {
          const raw = event.target.value;
          props.onChange(raw === '' ? '' : props.percent ? percentToRatio(Number(raw)) : Number(raw));
        }}
      />
    </Field>
  );
}
export function BooleanField(props: CommonFieldProps & { value: boolean; onChange: (value: boolean) => void }) {
  return (
    <div className="configuration-field configuration-checkbox">
      <label htmlFor={props.name}>
        <input
          id={props.name}
          name={props.name}
          type="checkbox"
          checked={props.value}
          disabled={props.disabled}
          aria-invalid={Boolean(props.error)}
          aria-describedby={`${props.name}-help`}
          onChange={(event) => props.onChange(event.target.checked)}
        />
        {props.label}
      </label>
      <p id={`${props.name}-help`} className={props.error ? 'configuration-error' : 'configuration-help'}>
        {props.error || props.help}
      </p>
    </div>
  );
}
export function ChoiceField(
  props: CommonFieldProps & {
    value: string;
    onChange: (value: string) => void;
    options: { value: string; label: string }[];
    placeholder?: string;
  },
) {
  return (
    <Field {...props}>
      <select {...attributes(props)} value={props.value} onChange={(event) => props.onChange(event.target.value)}>
        {props.placeholder !== undefined ? <option value="">{props.placeholder}</option> : null}
        {props.options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </Field>
  );
}
export function StringListField(
  props: CommonFieldProps & { value: string[]; onChange: (value: string[]) => void; addLabel?: string },
) {
  const { t } = useTranslation('configuration');
  const [entry, setEntry] = useState('');
  const localId = useId();
  const add = () => {
    const next = entry.trim();
    if (!next) return;
    if (!props.value.includes(next)) props.onChange([...props.value, next]);
    setEntry('');
  };
  return (
    <Field {...props}>
      <div className="configuration-inline">
        <input
          {...attributes(props)}
          required={false}
          value={entry}
          onChange={(event) => setEntry(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter') {
              event.preventDefault();
              add();
            }
          }}
        />
        <button type="button" className="configuration-button" disabled={props.disabled || !entry.trim()} onClick={add}>
          {props.addLabel ?? t('forms.add')}
        </button>
      </div>
      {props.value.length ? (
        <ul className="configuration-list">
          {props.value.map((item, index) => (
            <li key={`${localId}-${index}`}>
              <span>{item}</span>
              <button
                type="button"
                className="configuration-button"
                aria-label={t('forms.removeItem', { name: item })}
                disabled={props.disabled}
                onClick={() => props.onChange(props.value.filter((_, i) => i !== index))}
              >
                {t('forms.remove')}
              </button>
            </li>
          ))}
        </ul>
      ) : (
        <p className="configuration-help">{t('forms.emptyList')}</p>
      )}
    </Field>
  );
}

/** Open a closed advanced group before focusing its invalid field. */
export function focusFirstError(errors: FieldErrors, form?: HTMLFormElement) {
  for (const name of Object.keys(errors)) {
    const target = form?.elements.namedItem(name) ?? document.getElementsByName(name)[0];
    if (!(target instanceof HTMLElement)) continue;
    let parent = target.parentElement;
    while (parent) {
      if (parent instanceof HTMLDetailsElement) parent.open = true;
      parent = parent.parentElement;
    }
    target.focus();
    break;
  }
}

export type SecretFieldState = { revision: number; configured: boolean; updatedAt: string | null };
/** Write-only secret state never enters the shared document or mutation cache. */
export function RuntimeSecretField({ kind, state }: { kind: 'llm-gateway' | 'api-access' | 'news-provider'; state: SecretFieldState }) {
  const { t } = useTranslation('configuration');
  const writes = useRuntimeSecrets();
  const conflict = useRuntimeConfigConflict();
  const [token, setToken] = useState('');
  const [pending, setPending] = useState(false);
  const [saved, setSaved] = useState<{ revision: number; updated_at: string; savedNeedsReload: boolean }>();
  const [failed, setFailed] = useState(false);
  const gateway = kind === 'llm-gateway';
  const news = kind === 'news-provider';
  return (
    <div className="configuration-secret">
      <TextField
        name={`credential.${kind}`}
        label={t(news ? 'runtimeSecrets.news' : gateway ? 'runtimeSecrets.llm' : 'runtimeSecrets.api')}
        help={t(news ? 'runtimeSecrets.newsHint' : 'runtimeSecrets.hint')}
        type="password"
        value={token}
        onChange={(next) => {
          setToken(next);
          setFailed(false);
        }}
        error={failed ? t(conflict ? 'forms.accessWriteUnconfirmed' : 'forms.accessWriteFailed') : undefined}
      />
      <button
        type="button"
        className="configuration-button"
        disabled={pending || conflict || !token.trim()}
        onClick={() => {
          setPending(true);
          setFailed(false);
          const write = news ? writes.writeNewsProvider : gateway ? writes.writeLlmGateway : writes.writeApiAccess;
          void write(Math.max(state.revision, saved?.revision ?? 0), token)
            .then((result) => {
              setToken('');
              setSaved(result);
            })
            .catch((error: unknown) => {
              setFailed(true);
              if (error instanceof ApiError) setToken('');
            })
            .finally(() => setPending(false));
        }}
      >
        {pending
          ? t('forms.saving')
          : t(
              news ? (state.configured || saved ? 'runtimeSecrets.rotateNews' : 'runtimeSecrets.saveNews') : gateway
                ? state.configured || saved
                  ? 'runtimeSecrets.rotateGateway'
                  : 'runtimeSecrets.saveGateway'
                : state.configured || saved
                  ? 'runtimeSecrets.rotateApi'
                  : 'runtimeSecrets.saveApi',
            )}
      </button>
      <p role="status" className="configuration-help">
        {conflict ? t('conflict') : saved || state.configured ? t('forms.accessConfigured') : t('forms.accessMissing')}
        {saved?.updated_at || state.updatedAt ? ` · ${saved?.updated_at ?? state.updatedAt}` : ''}
      </p>
      {saved?.savedNeedsReload && conflict ? (
        <p role="alert" className="configuration-error">{t('forms.accessSavedNeedsReload')}</p>
      ) : null}
    </div>
  );
}

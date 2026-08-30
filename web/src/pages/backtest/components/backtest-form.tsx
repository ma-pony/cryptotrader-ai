import { useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Button } from '@/components/ui/button';
import { Field, NumberField, ChoiceField, focusFirstError, type FieldErrors } from '@/components/configuration/field';
import { useBacktestSessions, useLoadBacktestSession, useStartBacktest } from '@/hooks/use-backtest';

interface Props {
  onRunStarted: (runId: string) => void;
}
const PAIRS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT'];

export const BacktestForm = ({ onRunStarted }: Props) => {
  const { t } = useTranslation('backtest');
  const sessions = useBacktestSessions();
  const loadSession = useLoadBacktestSession();
  const startMutation = useStartBacktest();
  const form = useRef<HTMLFormElement>(null);
  const [pair, setPair] = useState('BTC/USDT');
  const [start, setStart] = useState('');
  const [end, setEnd] = useState('');
  const [capital, setCapital] = useState<number | ''>(10000);
  const [session, setSession] = useState('');
  const [errors, setErrors] = useState<FieldErrors>({});
  const today = new Date().toISOString().slice(0, 10);
  const busy = startMutation.isPending || loadSession.isPending;
  const pairOptions = PAIRS.includes(pair) ? PAIRS : [...PAIRS, pair];
  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    if (busy) return;
    const next: FieldErrors = {};
    if (!start) next.start = t('errors.start_required');
    if (!end) next.end = t('errors.end_required');
    else if (start && start >= end) next.end = t('errors.range');
    else if (end > today) next.end = t('errors.future');
    if (capital === '' || !Number.isFinite(capital) || capital < 100) next.initial_capital = t('errors.capital');
    setErrors(next);
    if (Object.keys(next).length || capital === '') {
      focusFirstError(next, form.current ?? undefined);
      return;
    }
    startMutation.mutate(
      { pair, start, end, initial_capital: capital },
      { onSuccess: (data) => onRunStarted(data.run_id) },
    );
  };

  return (
    <form ref={form} noValidate onSubmit={handleSubmit} className="space-y-4">
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <ChoiceField
          name="pair"
          label={t('form.pair')}
          value={pair}
          onChange={setPair}
          options={pairOptions.map((value) => ({ value, label: value }))}
          disabled={busy}
        />
        {(['start', 'end'] as const).map((name) => (
          <Field
            key={name}
            name={name}
            label={t(name === 'start' ? 'form.start_date' : 'form.end_date')}
            error={errors[name]}
          >
            <input
              id={name}
              name={name}
              type="date"
              required
              aria-required
              className="configuration-control"
              value={name === 'start' ? start : end}
              max={today}
              aria-invalid={Boolean(errors[name])}
              aria-describedby={`${name}-help`}
              disabled={busy}
              onChange={(event) => {
                (name === 'start' ? setStart : setEnd)(event.target.value);
                if (startMutation.isError) startMutation.reset();
              }}
            />
          </Field>
        ))}
        <NumberField
          name="initial_capital"
          label={t('form.initial_capital')}
          min={100}
          value={capital}
          onChange={(value) => {
            setCapital(value);
            if (startMutation.isError) startMutation.reset();
          }}
          error={errors.initial_capital}
          disabled={busy}
        />
      </div>
      {sessions.data?.sessions.length ? (
        <ChoiceField
          name="reuse-session"
          label={t('sessions.title')}
          help={t('sessions.help')}
          value={session}
          placeholder={t('sessions.select')}
          disabled={busy}
          options={sessions.data.sessions.map((value) => ({ value, label: value }))}
          onChange={(name) => {
            setSession(name);
            if (!name) {
              loadSession.reset();
              return;
            }
            loadSession.mutate(name, {
              onSuccess: ({ params }) => {
                setPair(params.pair);
                setStart(params.start);
                setEnd(params.end);
                setCapital(params.initial_capital);
                setErrors({});
                startMutation.reset();
              },
            });
          }}
        />
      ) : null}
      {Object.keys(errors).length ? (
        <p role="alert" className="configuration-error">
          {Object.values(errors)[0]}
        </p>
      ) : null}
      {startMutation.isError ? (
        <p role="alert" className="configuration-error">
          {t('errors.start_failed')}
        </p>
      ) : null}
      {loadSession.isError || sessions.isError ? (
        <p role="alert" className="configuration-error">
          {t('errors.load_failed')}
        </p>
      ) : null}
      {loadSession.isPending ? <p role="status">{t('sessions.loading')}</p> : null}
      <Button type="submit" disabled={busy}>
        {startMutation.isPending ? t('form.starting') : t('form.submit')}
      </Button>
    </form>
  );
};

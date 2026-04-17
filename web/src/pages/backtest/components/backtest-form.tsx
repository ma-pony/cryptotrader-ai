import { useState } from 'react';
import { useTranslation } from 'react-i18next';

import { Button } from '@/components/ui/button';
import { useBacktestSessions, useStartBacktest } from '@/hooks/use-backtest';

interface Props {
  onRunStarted: (runId: string) => void;
}

const PAIRS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT'];

export const BacktestForm = ({ onRunStarted }: Props) => {
  const { t } = useTranslation('backtest');
  const sessions = useBacktestSessions();
  const startMutation = useStartBacktest();

  const [pair, setPair] = useState('BTC/USDT');
  const [start, setStart] = useState('');
  const [end, setEnd] = useState('');
  const [capital, setCapital] = useState(10000);
  const [mode, setMode] = useState<'rules' | 'llm'>('rules');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!start || !end) return;
    startMutation.mutate(
      { pair, start, end, initial_capital: capital, mode },
      { onSuccess: (data) => onRunStarted(data.run_id) },
    );
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <label className="space-y-1 text-xs">
          <span className="text-muted-foreground">{t('form.pair')}</span>
          <select className="block h-8 w-full rounded-md border border-input bg-background px-2 text-sm" value={pair} onChange={(e) => setPair(e.target.value)}>
            {PAIRS.map((p) => <option key={p} value={p}>{p}</option>)}
          </select>
        </label>
        <label className="space-y-1 text-xs">
          <span className="text-muted-foreground">{t('form.start_date')}</span>
          <input type="date" required className="block h-8 w-full rounded-md border border-input bg-background px-2 text-sm" value={start} onChange={(e) => setStart(e.target.value)} />
        </label>
        <label className="space-y-1 text-xs">
          <span className="text-muted-foreground">{t('form.end_date')}</span>
          <input type="date" required className="block h-8 w-full rounded-md border border-input bg-background px-2 text-sm" value={end} onChange={(e) => setEnd(e.target.value)} />
        </label>
        <label className="space-y-1 text-xs">
          <span className="text-muted-foreground">{t('form.initial_capital')}</span>
          <input type="number" min={100} step={100} className="block h-8 w-full rounded-md border border-input bg-background px-2 text-sm tabular-nums" value={capital} onChange={(e) => setCapital(Number(e.target.value))} />
        </label>
      </div>

      <div className="flex items-center gap-4">
        <label className="flex items-center gap-2 text-xs">
          <input type="checkbox" checked={mode === 'llm'} onChange={(e) => setMode(e.target.checked ? 'llm' : 'rules')} className="rounded" />
          {t('form.use_llm')}
        </label>

        {sessions.data && sessions.data.sessions.length > 0 && (
          <label className="space-y-1 text-xs">
            <span className="text-muted-foreground">{t('sessions.title')}</span>
            <select className="ml-2 h-8 rounded-md border border-input bg-background px-2 text-sm">
              <option value="">{t('sessions.select')}</option>
              {sessions.data.sessions.map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
          </label>
        )}

        <Button type="submit" size="sm" disabled={startMutation.isPending || !start || !end}>
          {startMutation.isPending ? '...' : t('form.submit')}
        </Button>
      </div>
    </form>
  );
};

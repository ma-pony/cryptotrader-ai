import { type FC } from 'react';
import { useTranslation } from 'react-i18next';

import { cn } from '@/lib/cn';

interface ExchangeSelectorProps {
  value: 'binance' | 'okx';
  onChange: (exchange: 'binance' | 'okx') => void;
}

const exchanges = ['binance', 'okx'] as const;

export const ExchangeSelector: FC<ExchangeSelectorProps> = ({ value, onChange }) => {
  const { t } = useTranslation('market');

  return (
    <div className="inline-flex rounded-md border border-border">
      {exchanges.map((ex) => (
        <button
          key={ex}
          type="button"
          className={cn(
            'px-4 py-1.5 text-sm font-medium transition-colors first:rounded-l-md last:rounded-r-md',
            value === ex ? 'bg-primary text-primary-foreground' : 'bg-background text-muted-foreground hover:bg-accent',
          )}
          onClick={() => onChange(ex)}
        >
          {t(`exchange.${ex}`)}
        </button>
      ))}
    </div>
  );
};

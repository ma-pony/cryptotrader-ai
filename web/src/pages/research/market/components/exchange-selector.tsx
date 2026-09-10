import { type FC } from 'react';
import { useTranslation } from 'react-i18next';

import { Button } from '@/components/ui/button';

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
        <Button
          key={ex}
          type="button"
          variant={value === ex ? 'secondary' : 'ghost'}
          className="rounded-none first:rounded-l-md last:rounded-r-md"
          aria-pressed={value === ex}
          onClick={() => onChange(ex)}
        >
          {t(`exchange.${ex}`)}
        </Button>
      ))}
    </div>
  );
};

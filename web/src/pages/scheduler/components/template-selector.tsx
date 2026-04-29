import { useTranslation } from 'react-i18next';

import type { TriggerType } from '@/types/api';

export interface RuleFormValues {
  name: string;
  trigger_type: TriggerType;
  pair: string;
  parameters: Record<string, unknown>;
  cooldown_minutes: number;
}

interface Template {
  nameKey: string;
  descKey: string;
  values: RuleFormValues;
}

const TEMPLATES: Template[] = [
  {
    nameKey: 'templates.btc_price_drop.name',
    descKey: 'templates.btc_price_drop.description',
    values: {
      name: 'BTC 价格跌破阈值',
      trigger_type: 'price_threshold',
      pair: 'BTC/USDT',
      parameters: { direction: 'below', price: 60000 },
      cooldown_minutes: 60,
    },
  },
  {
    nameKey: 'templates.eth_volatility.name',
    descKey: 'templates.eth_volatility.description',
    values: {
      name: 'ETH 短期异常波动',
      trigger_type: 'pct_change',
      pair: 'ETH/USDT',
      parameters: { window_minutes: 15, threshold_pct: 3 },
      cooldown_minutes: 30,
    },
  },
  {
    nameKey: 'templates.funding_rate_alert.name',
    descKey: 'templates.funding_rate_alert.description',
    values: {
      name: '资金费率异常',
      trigger_type: 'funding_rate',
      pair: 'BTC/USDT',
      parameters: { threshold_pct: 0.1 },
      cooldown_minutes: 480,
    },
  },
  {
    nameKey: 'templates.btc_bearish_candles.name',
    descKey: 'templates.btc_bearish_candles.description',
    values: {
      name: 'BTC 连续阴线',
      trigger_type: 'candle_pattern',
      pair: 'BTC/USDT',
      parameters: { interval: '1h', consecutive_count: 3, direction: 'bearish' },
      cooldown_minutes: 120,
    },
  },
];

interface Props {
  onSelect: (data: Partial<RuleFormValues>) => void;
}

export const TemplateSelector = ({ onSelect }: Props) => {
  const { t } = useTranslation('scheduler');

  return (
    <div className="space-y-3">
      <p className="text-sm font-medium text-foreground">{t('templates.title')}</p>
      <div className="grid grid-cols-2 gap-3">
        {TEMPLATES.map((tpl) => (
          <button
            key={tpl.nameKey}
            type="button"
            onClick={() => onSelect(tpl.values)}
            className="flex flex-col gap-1 rounded-md border border-border bg-muted/50 p-3 text-left transition-colors hover:border-primary/50 hover:bg-primary/5"
          >
            <span className="text-sm font-medium text-foreground">{t(tpl.nameKey)}</span>
            <span className="text-xs text-muted-foreground">{t(tpl.descKey)}</span>
          </button>
        ))}
      </div>
    </div>
  );
};

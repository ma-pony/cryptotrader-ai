/**
 * PairBadge — render a trading pair with a market-type-aware badge.
 *
 * Spec 013 (`specs/013-pair-value-object/contracts/api_response_schema.md`):
 * - `pair`: ccxt canonical str (e.g. "BTC/USDT" spot, "BTC/USDT:USDT" perp)
 * - `pairDisplay` (optional): human form (e.g. "BTC/USDT (perp)"); falls
 *   back to deriving from `pair` + `marketType`.
 * - `marketType`: drives badge color + suffix label.
 *
 * Used in PortfolioPositions and DecisionDetail per US4.
 */
import { useTranslation } from 'react-i18next';

import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/cn';
import type { MarketType } from '@/types/api.schema';

export interface PairBadgeProps {
  pair: string;
  pairDisplay?: string | undefined;
  marketType?: MarketType;
  className?: string;
}

const MARKET_TYPE_VARIANT: Record<MarketType, 'secondary' | 'warning' | 'default' | 'outline'> = {
  spot: 'secondary',
  swap: 'warning',
  future: 'default',
  option: 'outline',
};

function basePairOf(pair: string): string {
  // Strip ccxt derivative suffix for compact display: "BTC/USDT:USDT" → "BTC/USDT"
  const colon = pair.indexOf(':');
  return colon === -1 ? pair : pair.slice(0, colon);
}

export function PairBadge({ pair, pairDisplay, marketType = 'spot', className }: PairBadgeProps) {
  const { t } = useTranslation('common');
  const symbol = basePairOf(pair);
  const variant = MARKET_TYPE_VARIANT[marketType];
  const label = t(`pair.market_type.${marketType}`);
  const tooltip = pairDisplay ?? pair;
  return (
    <span className={cn('inline-flex items-center gap-2', className)} title={tooltip}>
      <span className="font-mono text-sm tabular-nums">{symbol}</span>
      <Badge variant={variant} className="px-1.5 py-0 text-[10px] uppercase tracking-wide">
        {label}
      </Badge>
    </span>
  );
}

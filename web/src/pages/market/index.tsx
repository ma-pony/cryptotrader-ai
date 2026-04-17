import { useState } from 'react';
import { useTranslation } from 'react-i18next';

import { Card } from '@/components/ui/card';

import { ExchangeSelector } from './components/exchange-selector';
import { MarketSidebar } from './components/market-sidebar';
import { TradingViewChart } from './components/tradingview-chart';

const DEFAULT_PAIR = 'BTC/USDT';

const MarketPage = () => {
  const { t } = useTranslation('market');
  const [pair] = useState(DEFAULT_PAIR);
  const [exchange, setExchange] = useState<'binance' | 'okx'>('binance');

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold text-foreground">{t('title')}</h1>
        <ExchangeSelector value={exchange} onChange={setExchange} />
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[1fr_280px]">
        {/* TradingView chart */}
        <Card className="h-[600px] overflow-hidden">
          <TradingViewChart symbol={pair} exchange={exchange} />
        </Card>

        {/* Sidebar: funding rate / OI / liquidations */}
        <MarketSidebar pair={pair} exchange={exchange} />
      </div>
    </div>
  );
};

export default MarketPage;

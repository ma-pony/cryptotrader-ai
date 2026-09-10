import { useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Link } from 'react-router';

import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { PageHeader } from '@/components/ui/page-header';
import { ResearchNav } from '../presentation';
import type { CandlestickChartHandle } from '@/types/market-chart';

import { ChartTabPanel } from './components/chart-tab-panel';
import { ExchangeSelector } from './components/exchange-selector';
import { MarketSidebar } from './components/market-sidebar';

const DEFAULT_PAIR = 'BTC/USDT';

const TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h', '1d'] as const;

const MarketPage = () => {
  const { t } = useTranslation('market');
  const [pair] = useState(DEFAULT_PAIR);
  const [exchange, setExchange] = useState<'binance' | 'okx'>('binance');
  const [timeframe, setTimeframe] = useState('1h');
  const [activeTab, setActiveTab] = useState('tradingview');
  const chartRef = useRef<CandlestickChartHandle | null>(null);

  return (
    // Mobile: natural document flow with min-height on chart card.
    // lg+ : flex column locked to <main>'s available height (viewport - topbar
    // - main padding) so the chart card can flex-1 and fill the screen.
    <div className="flex flex-col gap-4 lg:h-full lg:min-h-[640px]">
      <ResearchNav />
      <PageHeader
        title={t('title')}
        actions={
          <>
            <select
              value={timeframe}
              onChange={(e) => {
                setTimeframe(e.target.value);
              }}
              className="configuration-control w-auto"
              aria-label={t('timeframe')}
            >
              {TIMEFRAMES.map((tf) => (
                <option key={tf} value={tf}>
                  {tf}
                </option>
              ))}
            </select>
            <ExchangeSelector value={exchange} onChange={setExchange} />
          </>
        }
      />

      {/* grid-rows-[minmax(0,1fr)] makes the single row fill the parent's
          flex-1 height while still allowing inner overflow-auto to work
          (default 'auto' rows would size to content, breaking flex-1). */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[1fr_280px] lg:grid-rows-[minmax(0,1fr)] lg:min-h-0 lg:flex-1">
        <div className="flex min-w-0 flex-col gap-2 lg:min-h-0">
          <Card className="h-[480px] overflow-hidden lg:h-auto lg:min-h-[420px] lg:flex-1">
            <ChartTabPanel
              symbol={pair}
              exchange={exchange}
              timeframe={timeframe}
              chartRef={chartRef}
              activeTab={activeTab}
              onTabChange={setActiveTab}
            />
          </Card>

          <div className="flex flex-wrap items-center gap-3">
            <Button asChild variant="outline">
              <Link to="/research/analysis">前往仅分析</Link>
            </Button>
            <p className="text-sm text-muted-foreground">行情浏览不会触发模型或交易。</p>
          </div>
        </div>

        <MarketSidebar pair={pair} exchange={exchange} />
      </div>
    </div>
  );
};

export default MarketPage;

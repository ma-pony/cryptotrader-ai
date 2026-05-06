import * as Tabs from '@radix-ui/react-tabs';
import { type RefObject, Suspense, lazy } from 'react';
import { useTranslation } from 'react-i18next';

import type { CandlestickChartHandle } from '@/types/chart-analysis';
import { TradingViewChart } from './tradingview-chart';

const CandlestickChart = lazy(() =>
  import('@/components/charts/candlestick-chart').then((m) => ({
    default: m.CandlestickChart,
  })),
);

interface ChartTabPanelProps {
  symbol: string;
  exchange: 'binance' | 'okx';
  timeframe: string;
  chartRef: RefObject<CandlestickChartHandle | null>;
  activeTab: string;
  onTabChange: (tab: string) => void;
}

export function ChartTabPanel({
  symbol,
  exchange,
  timeframe,
  chartRef,
  activeTab,
  onTabChange,
}: ChartTabPanelProps) {
  const { t } = useTranslation('market');

  return (
    // flex-col + h-full so that Tabs.Content can use flex-1 to fill the
    // Card height. Without this, Tabs.Content has no intrinsic size and
    // TradingView's autosize:true sees a 0-height container.
    <Tabs.Root value={activeTab} onValueChange={onTabChange} className="flex h-full flex-col">
      <Tabs.List className="flex shrink-0 border-b border-border">
        <Tabs.Trigger
          value="tradingview"
          className="px-4 py-2 text-sm data-[state=active]:border-b-2 data-[state=active]:border-primary"
        >
          {t('ai_analysis.tab_tradingview')}
        </Tabs.Trigger>
        <Tabs.Trigger
          value="candlestick"
          className="px-4 py-2 text-sm data-[state=active]:border-b-2 data-[state=active]:border-primary"
        >
          {t('ai_analysis.tab_candlestick')}
        </Tabs.Trigger>
      </Tabs.List>

      <Tabs.Content value="tradingview" className="min-h-0 flex-1 outline-none">
        <TradingViewChart symbol={symbol} exchange={exchange} />
      </Tabs.Content>

      <Tabs.Content value="candlestick" className="min-h-0 flex-1 outline-none">
        <Suspense fallback={<div className="flex h-full items-center justify-center text-muted-foreground">Loading...</div>}>
          <CandlestickChart
            ref={chartRef}
            symbol={symbol}
            timeframe={timeframe}
            exchange={exchange}
            fillContainer
          />
        </Suspense>
      </Tabs.Content>
    </Tabs.Root>
  );
}

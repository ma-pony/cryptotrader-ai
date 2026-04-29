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
    <Tabs.Root value={activeTab} onValueChange={onTabChange}>
      <Tabs.List className="flex border-b border-border">
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

      <Tabs.Content value="tradingview">
        <TradingViewChart symbol={symbol} exchange={exchange} />
      </Tabs.Content>

      <Tabs.Content value="candlestick">
        <Suspense fallback={<div className="flex h-[400px] items-center justify-center text-muted-foreground">Loading...</div>}>
          <CandlestickChart
            ref={chartRef}
            symbol={symbol}
            timeframe={timeframe}
            exchange={exchange}
          />
        </Suspense>
      </Tabs.Content>
    </Tabs.Root>
  );
}

import { useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useQuery } from '@tanstack/react-query';

import { Card } from '@/components/ui/card';
import { PageHeader } from '@/components/ui/page-header';
import type { CandlestickChartHandle, ChartCapturePayload, OHLCVBar } from '@/types/chart-analysis';
import { useChartAnalysis } from '@/hooks/use-chart-analysis';
import { generateDescription } from '@/lib/indicators';

import { AiAnalysisPanel } from './components/ai-analysis-panel';
import { ChartTabPanel } from './components/chart-tab-panel';
import { ExchangeSelector } from './components/exchange-selector';
import { MarketSidebar } from './components/market-sidebar';

const DEFAULT_PAIR = 'BTC/USDT';

async function fetchOHLCVBars(symbol: string, timeframe: string, exchange: string): Promise<OHLCVBar[]> {
  const pair = symbol.replace('/', '-');
  const res = await fetch(`/api/market/${pair}/ohlcv?timeframe=${timeframe}&limit=100&exchange=${exchange}`);
  if (!res.ok) return [];
  const json = (await res.json()) as { bars: OHLCVBar[] };
  return json.bars;
}

const TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h', '1d'] as const;

const MarketPage = () => {
  const { t } = useTranslation('market');
  const [pair] = useState(DEFAULT_PAIR);
  const [exchange, setExchange] = useState<'binance' | 'okx'>('binance');
  const [timeframe, setTimeframe] = useState('1h');
  const [secondaryTimeframe, setSecondaryTimeframe] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState('tradingview');
  const chartRef = useRef<CandlestickChartHandle | null>(null);

  const { result, triggerFast, triggerDeep, stop, resetResult } = useChartAnalysis();

  const { data: bars = [] } = useQuery({
    queryKey: ['ohlcv-analysis', pair, timeframe, exchange],
    queryFn: () => fetchOHLCVBars(pair, timeframe, exchange),
    staleTime: 30_000,
    enabled: activeTab === 'candlestick',
  });

  const { data: secondaryBars = [] } = useQuery({
    queryKey: ['ohlcv-analysis', pair, secondaryTimeframe, exchange],
    queryFn: () => fetchOHLCVBars(pair, secondaryTimeframe!, exchange),
    staleTime: 30_000,
    enabled: activeTab === 'candlestick' && secondaryTimeframe !== null,
  });

  useEffect(() => {
    resetResult();
  }, [pair, timeframe, resetResult]);

  const buildExtraPayloads = (): ChartCapturePayload[] => {
    if (!secondaryTimeframe || secondaryBars.length === 0) return [];
    return [{
      symbol: pair,
      timeframe: secondaryTimeframe,
      exchange,
      dataUrl: null,
      description: generateDescription(secondaryBars, pair, secondaryTimeframe),
      capturedAt: new Date().toISOString(),
    }];
  };

  const isCandlestick = activeTab === 'candlestick';
  const isAnalyzing = result.status === 'loading' || result.status === 'streaming';

  return (
    // Mobile: natural document flow with min-height on chart card.
    // lg+ : flex column locked to <main>'s available height (viewport - topbar
    // - main padding) so the chart card can flex-1 and fill the screen.
    <div className="flex flex-col gap-4 lg:h-full lg:min-h-[640px]">
      <PageHeader
        title={t('title')}
        actions={
          <>
            <select
              value={timeframe}
              onChange={(e) => {
                setTimeframe(e.target.value);
                if (e.target.value === secondaryTimeframe) setSecondaryTimeframe(null);
              }}
              className="rounded border border-border bg-card px-2 py-1 text-sm"
              aria-label={t('ai_analysis.tab_candlestick')}
            >
              {TIMEFRAMES.map((tf) => <option key={tf} value={tf}>{tf}</option>)}
            </select>
            <select
              value={secondaryTimeframe ?? ''}
              onChange={(e) => setSecondaryTimeframe(e.target.value || null)}
              className="rounded border border-border bg-card px-2 py-1 text-sm"
              aria-label={t('ai_analysis.secondary_timeframe', '对比时间周期')}
            >
              <option value="">{t('ai_analysis.no_secondary', '—')}</option>
              {TIMEFRAMES.filter((tf) => tf !== timeframe).map((tf) => (
                <option key={tf} value={tf}>{tf}</option>
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

          <div className="flex gap-2">
            <button
              type="button"
              disabled={!isCandlestick || isAnalyzing}
              aria-busy={isAnalyzing}
              aria-label={t('ai_analysis.fast_btn')}
              title={!isCandlestick ? t('ai_analysis.tab_tradingview_disabled_tip') : undefined}
              onClick={() => triggerFast(chartRef, bars, pair, timeframe, undefined, buildExtraPayloads())}
              className="rounded bg-primary px-3 py-1.5 text-sm text-primary-foreground disabled:opacity-50"
            >
              {isAnalyzing ? t('ai_analysis.status_loading') : t('ai_analysis.fast_btn')}
            </button>

            <button
              type="button"
              disabled={!isCandlestick || isAnalyzing}
              aria-label={t('ai_analysis.deep_btn')}
              title={!isCandlestick ? t('ai_analysis.tab_tradingview_disabled_tip') : undefined}
              onClick={() => triggerDeep(chartRef, bars, pair, timeframe, undefined, buildExtraPayloads())}
              className="rounded border border-border px-3 py-1.5 text-sm disabled:opacity-50"
            >
              {t('ai_analysis.deep_btn')}
            </button>
          </div>

          {/* Cap analysis panel height on lg+ so a long markdown response
              cannot push the chart card off-screen. Mobile: full content. */}
          <div className="lg:max-h-[28vh] lg:overflow-auto">
            <AiAnalysisPanel
              result={result}
              onStop={stop}
              onRetry={() => triggerFast(chartRef, bars, pair, timeframe, undefined, buildExtraPayloads())}
            />
          </div>
        </div>

        <MarketSidebar pair={pair} exchange={exchange} />
      </div>
    </div>
  );
};

export default MarketPage;

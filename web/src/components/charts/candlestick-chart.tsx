import {
  ColorType,
  createChart,
  type IChartApi,
  type ISeriesApi,
  type Time,
} from 'lightweight-charts';
import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';

import type { CandlestickChartHandle, OHLCVBar } from '@/types/chart-analysis';
import { calcSMA } from '@/lib/indicators';

interface CandlestickChartProps {
  symbol: string;
  timeframe?: string;
  exchange?: 'binance' | 'okx';
  height?: number;
}

const EMPTY_BARS: OHLCVBar[] = [];

async function fetchOHLCV(
  symbol: string,
  timeframe: string,
  exchange: string,
): Promise<OHLCVBar[]> {
  const pair = symbol.replace('/', '-');
  const res = await fetch(
    `/api/market/${pair}/ohlcv?timeframe=${timeframe}&limit=100&exchange=${exchange}`,
  );
  if (!res.ok) return [];
  const json = (await res.json()) as { bars: OHLCVBar[] };
  return json.bars;
}

export const CandlestickChart = forwardRef<CandlestickChartHandle, CandlestickChartProps>(
  ({ symbol, timeframe = '1h', exchange = 'binance', height = 400 }, ref) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);
    const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
    const [isReady, setIsReady] = useState(false);

    const { data: bars = EMPTY_BARS } = useQuery({
      queryKey: ['ohlcv', symbol, timeframe, exchange],
      queryFn: () => fetchOHLCV(symbol, timeframe, exchange),
      staleTime: 30_000,
    });

    useImperativeHandle(ref, () => ({
      captureScreenshot(): string | null {
        if (!isReady || !chartRef.current) return null;
        try {
          const canvas = chartRef.current.takeScreenshot();
          return canvas.toDataURL('image/png');
        } catch {
          return null;
        }
      },
    }));

    useEffect(() => {
      const el = containerRef.current;
      if (!el) return;

      const chart = createChart(el, {
        height,
        layout: {
          background: { type: ColorType.Solid, color: 'transparent' },
          textColor: '#9ca3b6',
        },
        grid: {
          vertLines: { color: '#1f2737' },
          horzLines: { color: '#1f2737' },
        },
        timeScale: { timeVisible: true, secondsVisible: false },
        rightPriceScale: { borderVisible: false },
      });

      const candleSeries = chart.addCandlestickSeries({
        upColor: '#22c55e',
        downColor: '#ef4444',
        borderUpColor: '#22c55e',
        borderDownColor: '#ef4444',
        wickUpColor: '#22c55e',
        wickDownColor: '#ef4444',
      });

      const sma20Series = chart.addLineSeries({
        color: '#3b82f6',
        lineWidth: 1,
        priceLineVisible: false,
      });

      const sma50Series = chart.addLineSeries({
        color: '#f59e0b',
        lineWidth: 1,
        priceLineVisible: false,
      });

      chartRef.current = chart;
      candleSeriesRef.current = candleSeries;

      if (bars.length > 0) {
        const candleData = bars.map((b) => ({
          time: Math.floor(b.time / 1000) as Time,
          open: b.open,
          high: b.high,
          low: b.low,
          close: b.close,
        }));
        candleSeries.setData(candleData);

        const sma20Data = bars
          .map((_, i) => {
            if (i < 19) return null;
            const val = calcSMA(bars.slice(0, i + 1), 20);
            return { time: Math.floor(bars[i]!.time / 1000) as Time, value: val };
          })
          .filter(Boolean) as { time: Time; value: number }[];
        sma20Series.setData(sma20Data);

        const sma50Data = bars
          .map((_, i) => {
            if (i < 49) return null;
            const val = calcSMA(bars.slice(0, i + 1), 50);
            return { time: Math.floor(bars[i]!.time / 1000) as Time, value: val };
          })
          .filter(Boolean) as { time: Time; value: number }[];
        sma50Series.setData(sma50Data);

        chart.timeScale().fitContent();
      }

      chart.timeScale().subscribeVisibleTimeRangeChange(() => {
        setIsReady(true);
      });

      const ro = new ResizeObserver((entries) => {
        for (const entry of entries) {
          chart.applyOptions({ width: entry.contentRect.width });
        }
      });
      ro.observe(el);

      return () => {
        ro.disconnect();
        chart.remove();
        chartRef.current = null;
        candleSeriesRef.current = null;
        setIsReady(false);
      };
    }, [bars, height]);

    return <div ref={containerRef} style={{ width: '100%', height }} />;
  },
);

CandlestickChart.displayName = 'CandlestickChart';

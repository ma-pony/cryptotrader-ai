import { ColorType, createChart, type IChartApi, type ISeriesApi, type Time } from 'lightweight-charts';
import { useEffect, useRef } from 'react';

export interface TrendSeries {
  id: string;
  name: string;
  color: string;
  points: { ts: string; value: number }[];
}

interface TrendChartProps {
  series: TrendSeries[];
  height?: number;
  className?: string;
  theme?: 'light' | 'dark';
}

export const TrendChart = ({ series, height = 240, className, theme = 'dark' }: TrendChartProps) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<Map<string, ISeriesApi<'Line'>>>(new Map());

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const isDark = theme === 'dark';
    const localSeries = seriesRef.current;
    const chart = createChart(el, {
      height,
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: isDark ? '#9ca3b6' : '#475569',
      },
      grid: {
        vertLines: { color: isDark ? '#1f2737' : '#e2e8f0' },
        horzLines: { color: isDark ? '#1f2737' : '#e2e8f0' },
      },
      timeScale: { timeVisible: true, secondsVisible: false },
      rightPriceScale: { borderVisible: false },
    });
    chartRef.current = chart;

    const onResize = () => chart.applyOptions({ width: el.clientWidth });
    onResize();
    const ro = new ResizeObserver(onResize);
    ro.observe(el);

    return () => {
      ro.disconnect();
      chart.remove();
      chartRef.current = null;
      localSeries.clear();
    };
  }, [height, theme]);

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    seriesRef.current.forEach((s) => chart.removeSeries(s));
    seriesRef.current.clear();
    for (const s of series) {
      const line = chart.addLineSeries({ color: s.color, lineWidth: 2, title: s.name });
      line.setData(
        s.points.map((p) => ({
          time: Math.floor(new Date(p.ts).getTime() / 1000) as Time,
          value: p.value,
        })),
      );
      seriesRef.current.set(s.id, line);
    }
    chart.timeScale().fitContent();
  }, [series]);

  return <div ref={containerRef} className={className} style={{ width: '100%', height }} />;
};

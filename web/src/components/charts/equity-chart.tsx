import { ColorType, createChart, type IChartApi, type ISeriesApi, type Time } from 'lightweight-charts';
import { useEffect, useMemo, useRef } from 'react';

import type { EquityPoint } from '@/types/api';

interface EquityChartProps {
  data: EquityPoint[];
  height?: number;
  className?: string;
  mode?: 'line' | 'area';
  theme?: 'light' | 'dark';
}

const AGGREGATION_THRESHOLD = 5000;

const aggregate = (points: EquityPoint[], buckets = 1000): EquityPoint[] => {
  if (points.length <= buckets) return points;
  const step = Math.ceil(points.length / buckets);
  const out: EquityPoint[] = [];
  for (let i = 0; i < points.length; i += step) {
    const slice = points.slice(i, Math.min(i + step, points.length));
    const sum = slice.reduce((s, p) => s + p.equity, 0);
    const last = slice[slice.length - 1];
    if (!last) continue;
    out.push({ ts: last.ts, equity: sum / slice.length });
  }
  return out;
};

const toSeriesData = (points: EquityPoint[]): { time: Time; value: number }[] =>
  points.map((p) => ({
    time: Math.floor(new Date(p.ts).getTime() / 1000) as Time,
    value: p.equity,
  }));

export const EquityChart = ({ data, height = 320, className, mode = 'area', theme = 'dark' }: EquityChartProps) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<'Line'> | ISeriesApi<'Area'> | null>(null);

  const seriesData = useMemo(() => toSeriesData(aggregate(data, AGGREGATION_THRESHOLD)), [data]);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const isDark = theme === 'dark';
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
    // Amber accent — AI decisions heritage. Colors align with --amber-500/600 in OKLCH,
    // approximated in sRGB hex so lightweight-charts' canvas renderer can consume them.
    const AMBER_LINE = '#d9a74f';
    const AMBER_TOP = 'rgba(217,167,79,0.42)';
    const AMBER_BOTTOM = 'rgba(217,167,79,0)';
    const series =
      mode === 'area'
        ? chart.addAreaSeries({
            lineColor: AMBER_LINE,
            topColor: AMBER_TOP,
            bottomColor: AMBER_BOTTOM,
            lineWidth: 2,
          })
        : chart.addLineSeries({ color: AMBER_LINE, lineWidth: 2 });
    chartRef.current = chart;
    seriesRef.current = series;

    const onResize = () => chart.applyOptions({ width: el.clientWidth });
    onResize();
    const ro = new ResizeObserver(onResize);
    ro.observe(el);

    return () => {
      ro.disconnect();
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
    };
  }, [height, mode, theme]);

  useEffect(() => {
    seriesRef.current?.setData(seriesData);
    chartRef.current?.timeScale().fitContent();
  }, [seriesData]);

  return <div ref={containerRef} className={className} style={{ width: '100%', height }} />;
};

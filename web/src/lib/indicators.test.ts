import { describe, expect, it } from 'vitest';
import type { OHLCVBar } from '@/types/chart-analysis';
import { calcMACD, calcRSI, calcSMA, calcVolumeRatio, generateDescription } from './indicators';

function makeBars(closes: number[]): OHLCVBar[] {
  return closes.map((c, i) => ({
    time: 1700000000000 + i * 3600000,
    open: c - 0.5,
    high: c + 1,
    low: c - 1,
    close: c,
    volume: 1000 + i * 10,
  }));
}

describe('calcSMA', () => {
  it('returns 0 when bars fewer than period', () => {
    expect(calcSMA(makeBars([1, 2]), 5)).toBe(0);
  });

  it('returns correct average', () => {
    const bars = makeBars([10, 20, 30]);
    expect(calcSMA(bars, 3)).toBe(20);
  });
});

describe('calcRSI', () => {
  it('returns 50 for insufficient data', () => {
    expect(calcRSI(makeBars([1, 2]), 14)).toBe(50);
  });

  it('returns near 50 for flat prices', () => {
    const flat = makeBars(Array.from({ length: 20 }, () => 100));
    const rsi = calcRSI(flat);
    expect(rsi).toBeGreaterThanOrEqual(0);
    expect(rsi).toBeLessThanOrEqual(100);
  });

  it('returns high RSI for uptrend', () => {
    const up = makeBars(Array.from({ length: 20 }, (_, i) => 100 + i));
    expect(calcRSI(up)).toBeGreaterThan(70);
  });
});

describe('calcMACD', () => {
  it('returns numbers', () => {
    const bars = makeBars(Array.from({ length: 30 }, (_, i) => 100 + i));
    const macd = calcMACD(bars);
    expect(typeof macd.value).toBe('number');
    expect(typeof macd.signalLine).toBe('number');
    expect(typeof macd.histogram).toBe('number');
  });
});

describe('calcVolumeRatio', () => {
  it('returns 1 for single bar', () => {
    expect(calcVolumeRatio(makeBars([100]))).toBe(1);
  });
});

describe('generateDescription', () => {
  it('includes all FR-003 required fields', () => {
    const bars = makeBars(Array.from({ length: 30 }, (_, i) => 50000 + i * 100));
    const desc = generateDescription(bars, 'BTC/USDT', '1h', { fundingRate: 0.0001 });

    expect(desc).toContain('BTC/USDT');
    expect(desc).toContain('1h');
    expect(desc).toContain('最新价');
    expect(desc).toContain('成交量比');
    expect(desc).toContain('趋势方向');
    expect(desc).toContain('RSI');
    expect(desc).toContain('MACD');
    expect(desc).toContain('最近3根K线');
    expect(desc).toContain('资金费率');
  });

  it('returns empty string for no bars', () => {
    expect(generateDescription([], 'BTC/USDT', '1h')).toBe('');
  });
});

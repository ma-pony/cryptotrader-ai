import type { OHLCVBar } from '@/types/chart-analysis';

export function calcSMA(bars: OHLCVBar[], period: number): number {
  if (bars.length < period) return 0;
  const slice = bars.slice(-period);
  return slice.reduce((sum, b) => sum + b.close, 0) / period;
}

export function calcRSI(bars: OHLCVBar[], period = 14): number {
  if (bars.length < period + 1) return 50;
  const closes = bars.slice(-(period + 1)).map((b) => b.close);
  let avgGain = 0;
  let avgLoss = 0;
  for (let i = 1; i < closes.length; i++) {
    const delta = closes[i]! - closes[i - 1]!;
    if (delta > 0) avgGain += delta;
    else avgLoss -= delta;
  }
  avgGain /= period;
  avgLoss /= period;
  if (avgLoss === 0) return 100;
  const rs = avgGain / avgLoss;
  return 100 - 100 / (1 + rs);
}

export function calcMACD(
  bars: OHLCVBar[],
  fast = 12,
  slow = 26,
  signal = 9,
): { value: number; signalLine: number; histogram: number } {
  const closes = bars.map((b) => b.close);
  const emaFast = ema(closes, fast);
  const emaSlow = ema(closes, slow);
  const macdLine = emaFast - emaSlow;
  const macdValues: number[] = [];
  for (let i = 0; i < closes.length; i++) {
    const ef = ema(closes.slice(0, i + 1), fast);
    const es = ema(closes.slice(0, i + 1), slow);
    macdValues.push(ef - es);
  }
  const signalLine = ema(macdValues, signal);
  return { value: macdLine, signalLine, histogram: macdLine - signalLine };
}

function ema(values: number[], period: number): number {
  if (values.length === 0) return 0;
  if (values.length < period) {
    return values.reduce((a, b) => a + b, 0) / values.length;
  }
  const k = 2 / (period + 1);
  let result = values.slice(0, period).reduce((a, b) => a + b, 0) / period;
  for (let i = period; i < values.length; i++) {
    result = values[i]! * k + result * (1 - k);
  }
  return result;
}

export function calcVolumeRatio(bars: OHLCVBar[], period = 20): number {
  if (bars.length < 2) return 1;
  const lookback = bars.slice(-period);
  const avg = lookback.reduce((sum, b) => sum + b.volume, 0) / lookback.length;
  if (avg === 0) return 1;
  return bars[bars.length - 1]!.volume / avg;
}

interface MarketDataContext {
  fundingRate?: number;
}

export function generateDescription(
  bars: OHLCVBar[],
  symbol: string,
  timeframe: string,
  marketData?: MarketDataContext,
): string {
  if (bars.length === 0) return '';

  const last = bars[bars.length - 1]!;
  const rsi = calcRSI(bars);
  const macd = calcMACD(bars);
  const sma20 = calcSMA(bars, 20);
  const sma50 = calcSMA(bars, 50);
  const volRatio = calcVolumeRatio(bars);

  const trend = sma20 > sma50 ? '上升趋势' : sma20 < sma50 ? '下降趋势' : '横盘';
  const rsiState = rsi > 70 ? '超买' : rsi < 30 ? '超卖' : '中性';
  const macdState = macd.histogram > 0 ? '多头动能' : '空头动能';

  const recent = bars.slice(-3);
  const recentDesc = recent
    .map(
      (b) =>
        `O:${b.open.toFixed(2)} H:${b.high.toFixed(2)} L:${b.low.toFixed(2)} C:${b.close.toFixed(2)}`,
    )
    .join(' | ');

  const lines = [
    `交易对: ${symbol}`,
    `时间周期: ${timeframe}`,
    `最新价: ${last.close.toFixed(2)}`,
    `成交量比: ${volRatio.toFixed(2)}x`,
    `趋势方向: ${trend} (SMA20=${sma20.toFixed(2)}, SMA50=${sma50.toFixed(2)})`,
    `RSI(14): ${rsi.toFixed(1)} — ${rsiState}`,
    `MACD: ${macd.value.toFixed(4)} / Signal: ${macd.signalLine.toFixed(4)} — ${macdState}`,
    `最近3根K线: ${recentDesc}`,
  ];

  if (marketData?.fundingRate != null) {
    lines.push(`资金费率: ${(marketData.fundingRate * 100).toFixed(4)}%`);
  }

  return lines.join('\n');
}

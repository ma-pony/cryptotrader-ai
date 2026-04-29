import type { ConnectionStatus } from '@/contexts/market-data';

interface FundingWindow {
  startHour: number;
  startMinute: number;
  endHour: number;
  endMinute: number;
}

const FUNDING_WINDOWS: FundingWindow[] = [
  { startHour: 7, startMinute: 30, endHour: 8, endMinute: 30 },
  { startHour: 15, startMinute: 30, endHour: 16, endMinute: 30 },
  { startHour: 23, startMinute: 30, endHour: 24, endMinute: 30 },
];

function isInFundingWindow(now: Date): boolean {
  const h = now.getUTCHours();
  const m = now.getUTCMinutes();
  const minutesOfDay = h * 60 + m;

  for (const w of FUNDING_WINDOWS) {
    const start = w.startHour * 60 + w.startMinute;
    const end = (w.endHour % 24) * 60 + w.endMinute;

    if (w.endHour >= 24) {
      if (minutesOfDay >= start || minutesOfDay < end) return true;
    } else {
      if (minutesOfDay >= start && minutesOfDay < end) return true;
    }
  }
  return false;
}

function isActiveMarket(priceChangePercent: number | undefined, threshold: number): boolean {
  if (priceChangePercent === undefined) return false;
  return Math.abs(priceChangePercent) > threshold;
}

interface UseAdaptivePollingOptions {
  wsStatus: ConnectionStatus;
  priceChangePercent?: number | undefined;
  volatilityThreshold?: number | undefined;
}

export function useAdaptivePolling({
  wsStatus,
  priceChangePercent,
  volatilityThreshold = 1.0,
}: UseAdaptivePollingOptions): { refetchInterval: number | false } {
  if (wsStatus === 'connected' || wsStatus === 'connecting') {
    return { refetchInterval: false };
  }

  if (wsStatus === 'degraded') {
    const active = isActiveMarket(priceChangePercent, volatilityThreshold) || isInFundingWindow(new Date());
    return { refetchInterval: active ? 10_000 : 60_000 };
  }

  // disconnected | reconnecting → conservative
  return { refetchInterval: 10_000 };
}

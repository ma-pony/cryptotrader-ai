import { createContext } from 'react';

export interface TickerData {
  pair: string;
  price: number;
  priceChangePercent: number;
  volume24h: number;
  ts: number;
}

export type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'degraded' | 'reconnecting';

export interface MarketDataContextValue {
  connectionStatus: ConnectionStatus;
  subscribe: (pair: string) => void;
  unsubscribe: (pair: string) => void;
  getPrice: (pair: string) => TickerData | undefined;
  subscribeToPrice: (pair: string, callback: () => void) => () => void;
}

export const MarketDataContext = createContext<MarketDataContextValue | null>(null);

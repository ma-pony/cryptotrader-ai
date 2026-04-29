export interface OHLCVBar {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface ChartCapturePayload {
  symbol: string;
  timeframe: string;
  exchange: 'binance' | 'okx';
  dataUrl: string | null;
  description: string;
  capturedAt: string;
}

export interface AdditionalContext {
  payloads: ChartCapturePayload[];
  model: string;
}

export type VisualAnalysisStatus = 'idle' | 'loading' | 'streaming' | 'done' | 'error';

export interface VisualAnalysisResult {
  status: VisualAnalysisStatus;
  contentMd: string;
  screenshotFailed: boolean;
  contextNotice: string;
  error: string;
}

export interface CandlestickChartHandle {
  captureScreenshot(): string | null;
}

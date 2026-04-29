import { useCallback, useRef, useState } from 'react';
import { useNavigate } from 'react-router';

import type {
  AdditionalContext,
  CandlestickChartHandle,
  ChartCapturePayload,
  OHLCVBar,
  VisualAnalysisResult,
  VisualAnalysisStatus,
} from '@/types/chart-analysis';
import { generateDescription } from '@/lib/indicators';
import { streamFetch, type SSEEvent } from '@/lib/stream-fetch';

const MAX_IMAGE_BYTES = 4_718_592;

const INITIAL_RESULT: VisualAnalysisResult = {
  status: 'idle',
  contentMd: '',
  screenshotFailed: false,
  contextNotice: '',
  error: '',
};

interface MarketDataContext {
  fundingRate?: number;
}

export interface UseChartAnalysisReturn {
  result: VisualAnalysisResult;
  triggerFast: (
    chartRef: React.RefObject<CandlestickChartHandle | null>,
    bars: OHLCVBar[],
    symbol: string,
    timeframe: string,
    marketData?: MarketDataContext,
    extraPayloads?: ChartCapturePayload[],
  ) => void;
  triggerDeep: (
    chartRef: React.RefObject<CandlestickChartHandle | null>,
    bars: OHLCVBar[],
    symbol: string,
    timeframe: string,
    marketData?: MarketDataContext,
    extraPayloads?: ChartCapturePayload[],
  ) => void;
  stop: () => void;
  resetResult: () => void;
}

function buildContext(
  chartRef: React.RefObject<CandlestickChartHandle | null>,
  bars: OHLCVBar[],
  symbol: string,
  timeframe: string,
  marketData?: MarketDataContext,
  extraPayloads?: ChartCapturePayload[],
): { context: AdditionalContext; screenshotFailed: boolean } {
  let screenshot: string | null;
  try {
    screenshot = chartRef.current?.captureScreenshot() ?? null;
  } catch {
    screenshot = null;
  }

  const screenshotFailed = screenshot === null;

  if (screenshot && new Blob([screenshot]).size > MAX_IMAGE_BYTES) {
    screenshot = null;
  }

  const description = generateDescription(bars, symbol, timeframe, marketData);

  const payloads: ChartCapturePayload[] = [
    {
      symbol,
      timeframe,
      exchange: 'binance',
      dataUrl: screenshot,
      description,
      capturedAt: new Date().toISOString(),
    },
    ...(extraPayloads ?? []),
  ];

  return {
    context: { payloads, model: '' },
    screenshotFailed,
  };
}

export function useChartAnalysis(): UseChartAnalysisReturn {
  const [result, setResult] = useState<VisualAnalysisResult>(INITIAL_RESULT);
  const abortRef = useRef<AbortController>(new AbortController());
  const navigate = useNavigate();

  const update = useCallback((patch: Partial<VisualAnalysisResult>) => {
    setResult((prev) => ({ ...prev, ...patch }));
  }, []);

  const triggerFast = useCallback(
    (
      chartRef: React.RefObject<CandlestickChartHandle | null>,
      bars: OHLCVBar[],
      symbol: string,
      timeframe: string,
      marketData?: MarketDataContext,
      extraPayloads?: ChartCapturePayload[],
    ) => {
      abortRef.current.abort();
      abortRef.current = new AbortController();

      const { context, screenshotFailed } = buildContext(
        chartRef,
        bars,
        symbol,
        timeframe,
        marketData,
        extraPayloads,
      );

      update({ status: 'loading', contentMd: '', screenshotFailed, contextNotice: '', error: '' });

      void streamFetch('/api/chat/stream', {
        body: { message: '请分析当前图表形态', additional_context: context },
        signal: abortRef.current.signal,
        onEvent: (ev: SSEEvent) => {
          if (ev.event === 'content_delta') {
            const delta = (ev.data as { delta?: string }).delta ?? '';
            setResult((prev) => ({
              ...prev,
              status: 'streaming' as VisualAnalysisStatus,
              contentMd: prev.contentMd + delta,
            }));
          } else if (ev.event === 'context_notice') {
            const type = (ev.data as { type?: string }).type ?? '';
            update({ contextNotice: type });
          } else if (ev.event === 'done') {
            update({ status: 'done' });
          }
        },
        onError: (err: Error) => {
          update({ status: 'error', error: err.message });
        },
      });
    },
    [update],
  );

  const triggerDeep = useCallback(
    (
      chartRef: React.RefObject<CandlestickChartHandle | null>,
      bars: OHLCVBar[],
      symbol: string,
      timeframe: string,
      marketData?: MarketDataContext,
      extraPayloads?: ChartCapturePayload[],
    ) => {
      const { context } = buildContext(chartRef, bars, symbol, timeframe, marketData, extraPayloads);
      void navigate('/chat', { state: { additionalContext: context } });
    },
    [navigate],
  );

  const stop = useCallback(() => {
    abortRef.current.abort();
    update({ status: 'done' });
  }, [update]);

  const resetResult = useCallback(() => {
    abortRef.current.abort();
    setResult(INITIAL_RESULT);
  }, []);

  return { result, triggerFast, triggerDeep, stop, resetResult };
}

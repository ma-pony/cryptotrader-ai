import { createRef } from 'react';
import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import '@/lib/i18n';
import type { CandlestickChartHandle } from '@/types/market-chart';
import { ChartTabPanel } from './chart-tab-panel';

vi.mock('./tradingview-chart', () => ({ TradingViewChart: () => <div /> }));
vi.mock('@/components/charts/candlestick-chart', () => new Promise(() => undefined));

describe('ChartTabPanel', () => {
  it('shows the localized loading state while the candlestick chart loads', () => {
    render(
      <ChartTabPanel
        symbol="BTC/USDT"
        exchange="binance"
        timeframe="1h"
        chartRef={createRef<CandlestickChartHandle>()}
        activeTab="candlestick"
        onTabChange={vi.fn()}
      />,
    );

    expect(screen.getByText('加载中…')).toBeVisible();
    expect(screen.queryByText('Loading...')).not.toBeInTheDocument();
  });
});

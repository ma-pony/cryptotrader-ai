import { useEffect, useRef, type FC } from 'react';
import { useTranslation } from 'react-i18next';

interface TradingViewChartProps {
  symbol: string;
  exchange: 'binance' | 'okx';
}

export const TradingViewChart: FC<TradingViewChartProps> = ({ symbol, exchange }) => {
  const { t } = useTranslation('market');
  const containerRef = useRef<HTMLDivElement>(null);
  const widgetRef = useRef<unknown>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    container.innerHTML = '';

    const tvSymbol = exchange === 'binance' ? `BINANCE:${symbol.replace('/', '')}` : `OKX:${symbol.replace('/', '-')}-SWAP`;

    try {
      // @ts-expect-error TradingView is loaded via CDN script
      // eslint-disable-next-line @typescript-eslint/no-unsafe-call, @typescript-eslint/no-unsafe-member-access
      widgetRef.current = new window.TradingView.widget({
        container_id: container.id,
        autosize: true,
        symbol: tvSymbol,
        interval: '60',
        timezone: 'Asia/Shanghai',
        theme: 'dark',
        style: '1',
        locale: 'zh_CN',
        toolbar_bg: '#0a0a0a',
        enable_publishing: false,
        hide_top_toolbar: false,
        hide_legend: false,
        save_image: false,
        allow_symbol_change: true,
        withdateranges: true,
        details: true,
      });
    } catch {
      container.innerHTML = `<div class="flex h-full items-center justify-center text-muted-foreground text-sm">${t('tradingview.fallback')}</div>`;
    }
  }, [symbol, exchange, t]);

  return (
    <div
      ref={containerRef}
      id="tradingview-chart-container"
      className="h-full w-full"
    />
  );
};

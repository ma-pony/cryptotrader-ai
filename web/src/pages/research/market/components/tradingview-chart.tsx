import { useEffect, useRef, useState, type FC } from 'react';
import { useTranslation } from 'react-i18next';

type TradingViewApi = {
  widget: new (options: Record<string, unknown>) => unknown;
};

declare global {
  interface Window {
    TradingView?: TradingViewApi;
  }
}

let tradingViewLoad: Promise<TradingViewApi> | undefined;

function loadTradingView(): Promise<TradingViewApi> {
  if (window.TradingView) return Promise.resolve(window.TradingView);
  tradingViewLoad ??= new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = 'https://s3.tradingview.com/tv.js';
    script.async = true;
    script.onload = () => (window.TradingView ? resolve(window.TradingView) : reject(new Error('TradingView missing')));
    script.onerror = () => reject(new Error('TradingView unavailable'));
    document.head.appendChild(script);
  });
  return tradingViewLoad;
}

interface TradingViewChartProps {
  symbol: string;
  exchange: 'binance' | 'okx';
}

export const TradingViewChart: FC<TradingViewChartProps> = ({ symbol, exchange }) => {
  const { t } = useTranslation('market');
  const containerRef = useRef<HTMLDivElement>(null);
  const widgetRef = useRef<unknown>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    container.innerHTML = '';
    setFailed(false);

    const tvSymbol = exchange === 'binance' ? `BINANCE:${symbol.replace('/', '')}` : `OKX:${symbol.replace('/', '-')}-SWAP`;

    let active = true;
    void loadTradingView()
      .then((tradingView) => {
        if (!active) return;
        widgetRef.current = new tradingView.widget({
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
      })
      .catch(() => {
        if (active) setFailed(true);
      });
    return () => {
      active = false;
      widgetRef.current = null;
      container.innerHTML = '';
    };
  }, [symbol, exchange, t]);

  return (
    <div ref={containerRef} id="tradingview-chart-container" className="h-full w-full">
      {failed ? (
        <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
          {t('tradingview.fallback')}
        </div>
      ) : null}
    </div>
  );
};

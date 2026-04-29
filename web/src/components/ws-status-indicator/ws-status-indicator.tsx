import type { FC } from 'react';
import { useTranslation } from 'react-i18next';

import type { ConnectionStatus } from '@/contexts/market-data';

interface WSStatusIndicatorProps {
  status: ConnectionStatus;
  refetchInterval?: number | undefined;
}

export const WSStatusIndicator: FC<WSStatusIndicatorProps> = ({ status, refetchInterval }) => {
  const { t } = useTranslation();

  if (status === 'connected') return null;

  if (status === 'connecting') {
    return (
      <span className="inline-flex items-center gap-1" role="status" aria-live="polite" aria-label={t('ws.connecting', '连接中')}>
        <span className="h-2 w-2 animate-pulse rounded-full bg-muted-foreground" />
      </span>
    );
  }

  if (status === 'reconnecting') {
    return (
      <span
        className="inline-flex items-center gap-1 text-xs text-warning"
        role="status"
        aria-live="polite"
        aria-label={t('ws.reconnecting', '重连中')}
      >
        <span className="h-2 w-2 animate-spin rounded-full border border-warning border-t-transparent" />
        <span>{t('ws.reconnecting', '重连中…')}</span>
      </span>
    );
  }

  // degraded | disconnected
  const intervalText = refetchInterval ? `${String(refetchInterval / 1000)}s` : undefined;
  const ariaLabel = intervalText
    ? t('ws.degraded_interval', { interval: intervalText, defaultValue: `数据延迟中，当前轮询间隔 ${intervalText}` })
    : t('ws.degraded', '数据延迟中，轮询模式');

  return (
    <span className="inline-flex items-center gap-1 text-xs text-warning" role="status" aria-live="polite" aria-label={ariaLabel}>
      <span className="text-sm">⚠</span>
      <span>
        {t('ws.polling_mode', '轮询模式')}
        {intervalText ? ` (${intervalText})` : ''}
      </span>
    </span>
  );
};

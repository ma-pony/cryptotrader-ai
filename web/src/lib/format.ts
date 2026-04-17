import i18n from './i18n';

const localeOf = (): string => (i18n.language || 'zh-CN').replace('_', '-');

const dateFmtCache = new Map<string, Intl.DateTimeFormat>();
const numberFmtCache = new Map<string, Intl.NumberFormat>();

const dateFormatter = (options: Intl.DateTimeFormatOptions): Intl.DateTimeFormat => {
  const key = `${localeOf()}|${JSON.stringify(options)}`;
  let f = dateFmtCache.get(key);
  if (!f) {
    f = new Intl.DateTimeFormat(localeOf(), options);
    dateFmtCache.set(key, f);
  }
  return f;
};

const numberFormatter = (options: Intl.NumberFormatOptions): Intl.NumberFormat => {
  const key = `${localeOf()}|${JSON.stringify(options)}`;
  let f = numberFmtCache.get(key);
  if (!f) {
    f = new Intl.NumberFormat(localeOf(), options);
    numberFmtCache.set(key, f);
  }
  return f;
};

export const formatDateTime = (input: string | Date): string => {
  const d = typeof input === 'string' ? new Date(input) : input;
  if (Number.isNaN(d.getTime())) return '—';
  return dateFormatter({
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  }).format(d);
};

export const formatDate = (input: string | Date): string => {
  const d = typeof input === 'string' ? new Date(input) : input;
  if (Number.isNaN(d.getTime())) return '—';
  return dateFormatter({ year: 'numeric', month: '2-digit', day: '2-digit' }).format(d);
};

export const formatTime = (input: string | Date): string => {
  const d = typeof input === 'string' ? new Date(input) : input;
  if (Number.isNaN(d.getTime())) return '—';
  return dateFormatter({ hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false }).format(d);
};

export const formatCurrency = (value: number, currency = 'USDT', fractionDigits = 2): string => {
  if (!Number.isFinite(value)) return '—';
  const symbol = currency === 'USDT' || currency === 'USD' ? '$' : '';
  return `${symbol}${numberFormatter({
    minimumFractionDigits: fractionDigits,
    maximumFractionDigits: fractionDigits,
  }).format(value)} ${currency === 'USDT' ? 'USDT' : ''}`.trim();
};

export const formatNumber = (value: number, fractionDigits = 2): string => {
  if (!Number.isFinite(value)) return '—';
  return numberFormatter({
    minimumFractionDigits: fractionDigits,
    maximumFractionDigits: fractionDigits,
  }).format(value);
};

export const formatPercent = (value: number, fractionDigits = 2): string => {
  if (!Number.isFinite(value)) return '—';
  const sign = value > 0 ? '+' : '';
  return `${sign}${numberFormatter({
    minimumFractionDigits: fractionDigits,
    maximumFractionDigits: fractionDigits,
  }).format(value)}%`;
};

export const pnlClass = (value: number): string => {
  if (!Number.isFinite(value) || value === 0) return 'text-muted-foreground';
  return value > 0 ? 'text-success' : 'text-destructive';
};

export const formatRelative = (input: string | Date): string => {
  const d = typeof input === 'string' ? new Date(input) : input;
  if (Number.isNaN(d.getTime())) return '—';
  const diffMs = d.getTime() - Date.now();
  const absSec = Math.abs(diffMs) / 1000;
  const rtf = new Intl.RelativeTimeFormat(localeOf(), { numeric: 'auto' });
  if (absSec < 60) return rtf.format(Math.round(diffMs / 1000), 'second');
  if (absSec < 3600) return rtf.format(Math.round(diffMs / 60000), 'minute');
  if (absSec < 86400) return rtf.format(Math.round(diffMs / 3600000), 'hour');
  return rtf.format(Math.round(diffMs / 86400000), 'day');
};

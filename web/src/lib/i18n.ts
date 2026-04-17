import i18next from 'i18next';
import { initReactI18next } from 'react-i18next';

import commonZh from '@/locales/zh-CN/common.json';
import dashboardZh from '@/locales/zh-CN/dashboard.json';
import decisionsZh from '@/locales/zh-CN/decisions.json';
import backtestZh from '@/locales/zh-CN/backtest.json';
import riskZh from '@/locales/zh-CN/risk.json';
import metricsZh from '@/locales/zh-CN/metrics.json';
import chatZh from '@/locales/zh-CN/chat.json';
import marketZh from '@/locales/zh-CN/market.json';

import commonEn from '@/locales/en-US/common.json';
import dashboardEn from '@/locales/en-US/dashboard.json';
import decisionsEn from '@/locales/en-US/decisions.json';
import backtestEn from '@/locales/en-US/backtest.json';
import riskEn from '@/locales/en-US/risk.json';
import metricsEn from '@/locales/en-US/metrics.json';
import chatEn from '@/locales/en-US/chat.json';
import marketEn from '@/locales/en-US/market.json';

const STORAGE_KEY = 'cryptotrader-locale';

const detectLocale = (): string => {
  if (typeof window === 'undefined') return 'zh-CN';
  try {
    const saved = window.localStorage.getItem(STORAGE_KEY);
    if (saved) return saved;
  } catch {
    // localStorage unavailable; ignore.
  }
  // Read persisted UI store (if present) before i18next bootstrap to avoid flash.
  try {
    const uiRaw = window.localStorage.getItem('cryptotrader-ui');
    if (uiRaw) {
      const parsed = JSON.parse(uiRaw) as { state?: { locale?: string } };
      const fromStore = parsed.state?.locale;
      if (fromStore) return fromStore;
    }
  } catch {
    // ignore
  }
  return 'zh-CN';
};

await i18next.use(initReactI18next).init({
  resources: {
    'zh-CN': {
      common: commonZh,
      dashboard: dashboardZh,
      decisions: decisionsZh,
      backtest: backtestZh,
      risk: riskZh,
      metrics: metricsZh,
      chat: chatZh,
      market: marketZh,
    },
    'en-US': {
      common: commonEn,
      dashboard: dashboardEn,
      decisions: decisionsEn,
      backtest: backtestEn,
      risk: riskEn,
      metrics: metricsEn,
      chat: chatEn,
      market: marketEn,
    },
  },
  lng: detectLocale(),
  fallbackLng: 'zh-CN',
  defaultNS: 'common',
  ns: ['common', 'dashboard', 'decisions', 'backtest', 'risk', 'metrics', 'chat', 'market'],
  interpolation: { escapeValue: false },
  returnNull: false,
});

export const setLocaleAndPersist = (locale: 'zh-CN' | 'en-US') => {
  void i18next.changeLanguage(locale);
  try {
    window.localStorage.setItem(STORAGE_KEY, locale);
  } catch {
    // ignore
  }
};

export default i18next;

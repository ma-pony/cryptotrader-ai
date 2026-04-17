import { onCLS, onINP, onLCP, onTTFB, type Metric } from 'web-vitals';

const log = (metric: Metric) => {
  if (!import.meta.env.DEV) return;

  console.info('[web-vitals]', metric.name, Math.round(metric.value), metric.rating);
};

export const initWebVitals = () => {
  try {
    onCLS(log);
    onINP(log);
    onLCP(log);
    onTTFB(log);
  } catch (err) {
    console.warn('[web-vitals] init failed', err);
  }
};

import type { MetricsSummary } from '@/types/api';

const DB_NAME = 'ct_metrics';
const STORE_NAME = 'samples';
const MAX_SAMPLES = 60;
const DB_VERSION = 1;

interface Sample {
  ts: string;
  pipeline_p50_ms: number;
  pipeline_p95_ms: number;
  execution_p50_ms: number;
  execution_p95_ms: number;
}

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: 'ts' });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(new Error(String(req.error)));
  });
}

export async function appendSample(summary: MetricsSummary): Promise<void> {
  const db = await openDb();
  const tx = db.transaction(STORE_NAME, 'readwrite');
  const store = tx.objectStore(STORE_NAME);
  const sample: Sample = {
    ts: summary.collected_at,
    pipeline_p50_ms: summary.percentiles.pipeline_p50_ms,
    pipeline_p95_ms: summary.percentiles.pipeline_p95_ms,
    execution_p50_ms: summary.percentiles.execution_p50_ms,
    execution_p95_ms: summary.percentiles.execution_p95_ms,
  };
  store.put(sample);

  const countReq = store.count();
  countReq.onsuccess = () => {
    if (countReq.result > MAX_SAMPLES) {
      const cursorReq = store.openCursor();
      let deleted = 0;
      const toDelete = countReq.result - MAX_SAMPLES;
      cursorReq.onsuccess = () => {
        const cursor = cursorReq.result;
        if (cursor && deleted < toDelete) {
          cursor.delete();
          deleted++;
          cursor.continue();
        }
      };
    }
  };
  await new Promise<void>((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(new Error(String(tx.error)));
  });
  db.close();
}

export async function loadHistory(): Promise<Sample[]> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readonly');
    const store = tx.objectStore(STORE_NAME);
    const req = store.getAll();
    req.onsuccess = () => {
      db.close();
      resolve(req.result as Sample[]);
    };
    req.onerror = () => {
      db.close();
      reject(new Error(String(req.error)));
    };
  });
}

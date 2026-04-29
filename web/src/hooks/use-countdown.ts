import { useEffect, useState } from 'react';

/**
 * Shared countdown hook — avoids per-component 1-second setInterval storms.
 *
 * Previously ``SidebarFooter`` and ``SchedulerCard`` each ran their own
 * ``setInterval(1000)`` / ``useState`` / re-render cascade, so Dashboard-visible
 * sessions burned 2 full component tree re-renders per second. This hook
 * centralises the tick and lets consumers opt into coarser precision.
 *
 * Arguments:
 *   - ``targetIso``: ISO 8601 target timestamp (or null — returns ``null``)
 *   - ``intervalMs``: tick resolution. Default ``1000`` for sub-minute precision;
 *     pass ``60_000`` for minute-level consumers (sidebar footer) to cut the
 *     re-render cost to 1/minute.
 *
 * Returns: ``{ ms, formatted }`` where ``formatted`` is ``mm:ss`` or ``hh:mm:ss``.
 * ``ms`` is ``null`` when ``targetIso`` is null/invalid; negative values clamp to 0.
 */
export interface CountdownResult {
  ms: number | null;
  formatted: string;
}

const formatMs = (ms: number): string => {
  if (!Number.isFinite(ms) || ms < 0) return '即将';
  const totalSec = Math.floor(ms / 1000);
  const h = Math.floor(totalSec / 3600);
  const m = Math.floor((totalSec % 3600) / 60);
  const s = totalSec % 60;
  return h > 0
    ? `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`
    : `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
};

export const useCountdown = (
  targetIso: string | null | undefined,
  intervalMs: number = 1_000,
): CountdownResult => {
  const [now, setNow] = useState<number>(() => Date.now());

  useEffect(() => {
    if (!targetIso) return;
    const id = window.setInterval(() => setNow(Date.now()), intervalMs);
    return () => window.clearInterval(id);
  }, [targetIso, intervalMs]);

  if (!targetIso) return { ms: null, formatted: '—' };
  const targetMs = new Date(targetIso).getTime();
  if (!Number.isFinite(targetMs)) return { ms: null, formatted: '—' };
  const remaining = targetMs - now;
  return { ms: remaining, formatted: formatMs(remaining) };
};

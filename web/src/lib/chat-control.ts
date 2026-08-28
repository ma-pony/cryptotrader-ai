import { buildApiUrl } from './api-url';
import { env } from './env';
import { useSettingsStore } from '@/stores/use-settings-store';

const cancellationBarriers = new Map<string, Promise<void>>();

export function interruptChatSession(sessionId: string): Promise<Response> {
  const apiKey = useSettingsStore.getState().apiKey;
  const request = fetch(
    buildApiUrl(`/api/chat/interrupt/${encodeURIComponent(sessionId)}`, env.VITE_API_BASE_URL),
    {
      method: 'POST',
      headers: { 'X-API-Key': apiKey },
    },
  );
  const settledRequest = request.then(
    () => undefined,
    () => undefined,
  );
  const previousBarrier = cancellationBarriers.get(sessionId);
  const barrier = previousBarrier
    ? Promise.all([previousBarrier, settledRequest]).then(() => undefined)
    : settledRequest;
  cancellationBarriers.set(sessionId, barrier);
  void barrier.finally(() => {
    if (cancellationBarriers.get(sessionId) === barrier) {
      cancellationBarriers.delete(sessionId);
    }
  });
  return request;
}

export async function waitForChatCancellation(sessionId: string): Promise<void> {
  while (true) {
    const barrier = cancellationBarriers.get(sessionId);
    if (!barrier) return;
    await barrier;
  }
}

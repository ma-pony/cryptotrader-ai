import { buildApiUrl } from './api-url';
import { env } from './env';
import { useSettingsStore } from '@/stores/use-settings-store';

export function interruptChatSession(sessionId: string): Promise<Response> {
  const apiKey = useSettingsStore.getState().apiKey;
  return fetch(
    buildApiUrl(`/api/chat/interrupt/${encodeURIComponent(sessionId)}`, env.VITE_API_BASE_URL),
    {
      method: 'POST',
      headers: { 'X-API-Key': apiKey },
    },
  );
}

import { type Page } from '@playwright/test';

const ALLOWED_PORTS = new Set(['4174', '8011']);

function isAllowed(url: URL) {
  return url.hostname === '127.0.0.1' && ALLOWED_PORTS.has(url.port);
}

export interface NetworkAttempts {
  externalHttp: string[];
  externalWebSockets: string[];
}

export async function installNetworkGuard(page: Page): Promise<NetworkAttempts> {
  const attempts: NetworkAttempts = { externalHttp: [], externalWebSockets: [] };
  await page.route('**/*', async (route) => {
    const url = new URL(route.request().url());
    if (isAllowed(url) || ['data:', 'blob:'].includes(url.protocol)) await route.continue();
    else {
      attempts.externalHttp.push(url.href);
      await route.abort('blockedbyclient');
    }
  });
  await page.routeWebSocket(/.*/, async (socket) => {
    const url = new URL(socket.url());
    if (isAllowed(url)) {
      socket.connectToServer();
      return;
    }
    attempts.externalWebSockets.push(url.href);
    await socket.close({ code: 1008, reason: 'External WebSocket blocked by workbench test' });
  });
  return attempts;
}

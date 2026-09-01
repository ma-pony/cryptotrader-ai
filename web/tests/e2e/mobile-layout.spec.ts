import path from 'node:path';

import { devices, expect, test } from '@playwright/test';

import { installNetworkGuard } from './network-guard';

const TEST_KEY = 'workbench-test-access';
const EVIDENCE_DIR = path.resolve(
  import.meta.dirname,
  '../../../.superpowers/sdd/2026-08-31-trading-workbench/U3-fix1-images',
);

test('真实移动设备上下文使用390布局并呈现抽屉与保存栏', async ({ browser }) => {
  const context = await browser.newContext({ ...devices['iPhone 13'] });
  const page = await context.newPage();
  const attempts = await installNetworkGuard(page);
  try {
    await page.goto('/engine');
    expect(await page.evaluate(() => window.innerWidth)).toBe(390);
    const access = page.getByLabel('已有接口访问密钥');
    await access.fill(TEST_KEY);
    await page.getByRole('button', { name: '解锁并重新加载' }).click();
    await expect(access).toBeHidden();

    await page.getByRole('button', { name: '切换侧边栏' }).click();
    const drawer = page.getByRole('dialog');
    await expect(drawer).toBeVisible();
    await drawer.getByRole('link', { name: '引擎' }).click();
    await expect(drawer).toBeHidden();
    await expect(page.getByRole('button', { name: '保存配置' }).first()).toBeVisible();
    await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
    await page.screenshot({ path: path.join(EVIDENCE_DIR, '05-mobile-engine-390-light-fix1.png'), fullPage: true });

    expect(attempts.externalHttp).toEqual([]);
    expect(attempts.externalWebSockets).toEqual([]);
  } finally {
    await context.close();
  }
});

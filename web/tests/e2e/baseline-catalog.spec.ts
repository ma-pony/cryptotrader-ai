import { expect, test, type Page } from '@playwright/test';

import { installNetworkGuard } from './network-guard';

const TEST_KEY = 'workbench-test-access';

async function unlock(page: Page) {
  await page.goto('/engine');
  const access = page.getByLabel('已有接口访问密钥');
  await expect(access).toBeVisible();
  await access.fill(TEST_KEY);
  await page.getByRole('button', { name: '解锁并重新加载' }).click();
  await expect(access).toBeHidden();
}

test('同一前端构建在基础注册表中不显示代码扩展示例', async ({ page }) => {
  const attempts = await installNetworkGuard(page);
  await unlock(page);

  await expect(page.getByRole('heading', { name: '测试趋势信号' })).toHaveCount(0);
  await page.getByRole('link', { name: '账户' }).click();
  await page.getByRole('link', { name: '管理连接' }).click();
  await page.getByRole('button', { name: '新增连接' }).click();
  await expect(page.getByLabel('交易平台').locator('option[value="sample_venue"]')).toHaveCount(0);

  expect(attempts.externalHttp).toEqual([]);
  expect(attempts.externalWebSockets).toEqual([]);
});

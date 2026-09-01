import { expect, test, type Page } from '@playwright/test';

import { installNetworkGuard } from './network-guard';

const TEST_KEY = 'workbench-test-access';

async function navigate(page: Page, path: string) {
  await page.evaluate((next) => {
    window.history.pushState({}, '', next);
    window.dispatchEvent(new PopStateEvent('popstate'));
  }, path);
}

async function unlock(page: Page) {
  await page.goto('/engine');
  const access = page.getByLabel('已有接口访问密钥');
  await expect(access).toBeVisible();
  await access.fill('错误密钥');
  await page.getByRole('button', { name: '解锁并重新加载' }).click();
  await expect(page.getByText('验证未通过，请检查访问密钥后重试。')).toBeVisible();
  await access.fill(TEST_KEY);
  await page.getByRole('button', { name: '解锁并重新加载' }).click();
  await expect(access).toBeHidden();
  await expect(page.getByRole('heading', { name: '引擎', exact: true })).toBeVisible();
}

test('同一通用前端读取后端代码注册的平台与信号声明', async ({ page }) => {
  const attempts = await installNetworkGuard(page);
  await unlock(page);

  await expect(page.getByRole('heading', { name: '测试趋势信号' })).toBeVisible();
  await expect(page.getByRole('link', { name: '查看 测试趋势信号 结果' })).toBeVisible();

  await navigate(page, '/accounts/connections');
  await page.getByRole('button', { name: '新增连接' }).click();
  await page.getByLabel('交易平台').selectOption('sample_venue');
  await expect(page.getByLabel('环境')).toHaveValue('sandbox');
  await expect(page.getByLabel('账户编码')).toBeVisible();
  await expect(page.getByLabel('访问令牌')).toBeVisible();
  await expect(page.getByLabel('租户 PIN')).toBeVisible();

  expect(attempts.externalHttp).toEqual([]);
  expect(attempts.externalWebSockets).toEqual([]);
});

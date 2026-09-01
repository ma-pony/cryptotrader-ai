import path from 'node:path';

import { expect, test, type APIRequestContext, type APIResponse, type Locator, type Page } from '@playwright/test';

import { installNetworkGuard } from './network-guard';

const API = 'http://127.0.0.1:8011';
const TEST_KEY = 'workbench-test-access';
const headers = { 'X-API-Key': TEST_KEY };
const EVIDENCE_DIR = path.resolve(
  import.meta.dirname,
  '../../../.superpowers/sdd/2026-08-31-trading-workbench/U3-fix1-images',
);

interface Connection {
  id: string;
  label: string;
  credential_configured: boolean;
}

interface Book {
  id: string;
  label: string;
}

interface ConfigurationPayload {
  revision: number;
  document: {
    market_data: { timeframe: string };
    signals: { components: Array<{ component_id: string; enabled: boolean }> };
    execution: {
      connections: Connection[];
      books: Book[];
    };
  };
}

interface WorkbenchState {
  venue: { connect_calls: number; account_reads: number; order_writes: number };
}

interface PaperLedger {
  fills: Array<{
    side: 'buy' | 'sell';
    amount: string;
    occurred_at: string;
    fee: MoneyFact;
    realized_pnl: MoneyFact;
  }>;
  positions: Record<string, { signed_amount: string }>;
}

interface MoneyFact {
  amount: string | null;
  currency: string;
  unavailable_reason: string | null;
}

interface AccountIncome {
  realized_gross: MoneyFact[];
  fees: MoneyFact[];
  funding: MoneyFact[];
  net_trading: MoneyFact[];
  methodology: string;
}

interface TradingScope {
  books: Array<{ book_id: string; eligible: boolean }>;
}

async function json<T>(response: APIResponse): Promise<T> {
  return (await response.json()) as T;
}

function displayedAmount(value: string) {
  return /^[+-]?0+(?:\.0+)?(?:e[+-]?\d+)?$/i.test(value) ? '0' : value;
}

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
  await access.fill(TEST_KEY);
  await page.getByRole('button', { name: '解锁并重新加载' }).click();
  await expect(access).toBeHidden();
  await expect(page.getByRole('heading', { name: '引擎', exact: true })).toBeVisible();
}

async function getConfig(request: APIRequestContext): Promise<ConfigurationPayload> {
  const response = await request.get(`${API}/api/config`, { headers });
  if (!response.ok()) {
    throw new Error(`GET /api/config failed (${response.status()}): ${await response.text()}`);
  }
  return json<ConfigurationPayload>(response);
}

function waitForConfigurationSave(page: Page) {
  return page.waitForResponse(
    (response) => response.url() === `${API}/api/config` && response.request().method() === 'PUT',
  );
}

async function state(request: APIRequestContext): Promise<WorkbenchState> {
  const response = await request.get(`${API}/__workbench__/state`);
  expect(response.ok()).toBeTruthy();
  return json<WorkbenchState>(response);
}

async function createConnection(page: Page, label: string, code: string) {
  await navigate(page, '/accounts/connections');
  await page.getByRole('button', { name: '新增连接' }).click();
  const form = page.getByRole('form', { name: '新增平台连接' });
  await form.getByLabel('名称').fill(label);
  await form.getByLabel('交易平台').selectOption('sample_venue');
  await expect(form.getByLabel('环境')).toHaveValue('sandbox');
  const accountCode = form.getByLabel('账户编码');
  await accountCode.pressSequentially(code);
  await accountCode.press('Tab');
  await form.getByLabel('访问令牌').fill(`${code}-token`);
  await form.getByLabel('租户 PIN').fill('2468');
  await expect(accountCode).toHaveValue(code);
  const connectionSaved = page.waitForResponse(
    (response) => response.url() === `${API}/api/venue-connections` && response.request().method() === 'POST',
  );
  const credentialsSaved = page.waitForResponse(
    (response) => response.url().endsWith('/credentials') && response.request().method() === 'PUT',
  );
  const connectionChecked = page.waitForResponse(
    (response) => response.url().endsWith('/test') && response.request().method() === 'POST',
  );
  await form.getByRole('button', { name: '保存并检查' }).click();
  const responses = await Promise.all([connectionSaved, credentialsSaved, connectionChecked]);
  expect(responses.every((response) => response.ok())).toBe(true);
  await expect(page.getByRole('form', { name: label }).getByText('账户读取已验证')).toBeVisible();
}

async function approveBook(page: Page, bookId: string) {
  await navigate(page, `/accounts/books/${encodeURIComponent(bookId)}`);
  const approve = page.getByRole('button', { name: '批准' });
  await expect(approve).toBeVisible();
  await approve.click();
  const confirm = page.getByRole('button', { name: '确认批准' });
  await expect(confirm).toBeFocused();
  await confirm.press('Enter');
  await expect(page.getByText(/^已执行 ·/)).toBeVisible();
}

async function setCheckbox(locator: Locator, checked: boolean) {
  if ((await locator.isChecked()) !== checked) await locator.setChecked(checked);
}

async function selectTheme(page: Page, name: '浅色' | '深色') {
  await page.getByRole('button', { name: '跟随系统' }).click();
  await page.getByRole('menuitem', { name }).click();
  await expect(page.locator('html')).toHaveAttribute('data-theme', name === '浅色' ? 'light' : 'dark');
}

test('首次配置到次日复盘、逐池审批与单池人工退出', async ({ page, request }) => {
  const attempts = await installNetworkGuard(page);
  await unlock(page);

  // 首次无交易配置仍可浏览，并可通过键盘保存后端注册的固定信号。
  await navigate(page, '/');
  await expect(page.getByRole('heading', { name: '工作台' })).toBeVisible();
  await expect(page.getByText('交易未就绪')).toBeVisible();
  await navigate(page, '/engine');
  await page.getByLabel('添加信号组件').selectOption('sample_signal');
  await setCheckbox(page.getByLabel('启用 Kronos 时序模型'), false);
  await setCheckbox(page.getByLabel('启用 大模型四智能体委员会'), false);
  await page.getByLabel('测试趋势信号权重（%）').fill('100');
  await page.getByLabel('观察窗口').fill('24');
  await page.getByLabel('最大目标敞口（%）').fill('40');
  const evaluationInterval = page.getByLabel('信号评估周期');
  await evaluationInterval.fill('2h');
  const signalSave = waitForConfigurationSave(page);
  await evaluationInterval.press('Enter');
  expect((await signalSave).ok()).toBe(true);
  await expect
    .poll(async () => {
      const saved = await getConfig(request);
      return saved.document.signals.components
        .filter((item) => ['kronos', 'llm_committee', 'sample_signal'].includes(item.component_id))
        .map((item) => [item.component_id, item.enabled]);
    })
    .toEqual([
      ['kronos', false],
      ['llm_committee', false],
      ['sample_signal', true],
    ]);
  const timeframe = page.getByLabel('全局参考周期');
  await timeframe.fill('4h');
  const marketSave = waitForConfigurationSave(page);
  await timeframe.press('Enter');
  expect((await marketSave).ok()).toBe(true);
  await expect.poll(async () => (await getConfig(request)).document.market_data.timeframe).toBe('4h');
  await expect
    .poll(async () => {
      const response = await request.get(`${API}/api/runtime/status`, { headers });
      return (await json<{ analysis: { ready: boolean } }>(response)).analysis.ready;
    })
    .toBe(true);

  const beforeAnalysis = await state(request);
  expect(beforeAnalysis.venue).toEqual({ connect_calls: 0, account_reads: 0, order_writes: 0 });
  await navigate(page, '/');
  await page.getByLabel('交易对').fill('BTC/USDT');
  await page.getByRole('button', { name: '仅分析一次' }).click();
  await expect(page).toHaveURL(/\/decisions\//);
  const analysisPath = new URL(page.url()).pathname;
  await request.post(`${API}/__workbench__/tasks/drain`);
  await page.reload();
  await expect(page.getByText('密钥未提供')).toBeVisible();
  await unlock(page);
  await navigate(page, analysisPath);
  await expect(page.getByText('测试指标')).toBeVisible();
  await expect(page.getByText('测试结果')).toBeVisible();
  await page.setViewportSize({ width: 1440, height: 900 });
  await selectTheme(page, '浅色');
  await expect(page.getByText('行情流：未连接')).toBeVisible();
  await page.screenshot({ path: path.join(EVIDENCE_DIR, '01-analysis-result-1440-light-fix1.png'), fullPage: true });
  const afterAnalysis = await state(request);
  expect(afterAnalysis.venue.account_reads).toBe(0);
  expect(afterAnalysis.venue.order_writes).toBe(0);

  // 两个动态凭据连接均自动只读检查。
  await createConnection(page, 'Alpha 模拟账户', 'alpha');
  await createConnection(page, 'Beta 模拟账户', 'beta');
  let configuration = await getConfig(request);
  const alpha = configuration.document.execution.connections.find((item) => item.label.startsWith('Alpha'));
  const beta = configuration.document.execution.connections.find((item) => item.label.startsWith('Beta'));
  if (!alpha || !beta) throw new Error('expected both workbench connections');
  expect(alpha.credential_configured).toBe(true);
  expect(beta.credential_configured).toBe(true);

  // 配置交易范围和两个互不混资、逐池审批的模拟资金池。
  await navigate(page, '/accounts/books');
  const pairInput = page.getByLabel('交易品种范围');
  await pairInput.fill('BTC/USDT');
  await pairInput.press('Enter');
  await page.getByRole('button', { name: '新增资金池' }).click();
  const alphaEditor = page.locator('article.configuration-section').nth(0);
  await alphaEditor.getByLabel('名称').fill('Alpha 池');
  await setCheckbox(alphaEditor.getByLabel('启用 Alpha 模拟账户'), true);
  await page.getByRole('button', { name: '新增资金池' }).click();
  const betaEditor = page.locator('article.configuration-section').nth(1);
  await betaEditor.getByLabel('名称').fill('Beta 池');
  await setCheckbox(betaEditor.getByLabel('启用 Beta 模拟账户'), true);
  await page.getByRole('button', { name: '保存配置' }).click();
  await expect(page.getByText('本节已保存')).toBeVisible();
  configuration = await getConfig(request);
  const alphaBook = configuration.document.execution.books.find((item) => item.label === 'Alpha 池');
  const betaBook = configuration.document.execution.books.find((item) => item.label === 'Beta 池');
  if (!alphaBook || !betaBook) throw new Error('expected both workbench books');

  // 一次全局模拟交易必须逐池确认，再分别进行 HITL 批准。
  await navigate(page, '/');
  await expect(page.getByText('交易已就绪')).toBeVisible();
  await page.getByRole('button', { name: '运行一次交易' }).click();
  await setCheckbox(page.getByLabel(/Alpha 池 · 模拟资金/), true);
  await setCheckbox(page.getByLabel(/Beta 池 · 模拟资金/), true);
  await page.getByRole('button', { name: '确认全部范围并发起交易' }).click();
  await expect(page).toHaveURL(/\/decisions\//);
  const tradingDecisionId = page.url().split('/').pop()!;
  await expect
    .poll(async () => {
      const response = await request.get(`${API}/api/decisions/${tradingDecisionId}`, { headers });
      return (await json<{ status: string }>(response)).status;
    })
    .toBe('awaiting_approval');
  await expect
    .poll(async () => {
      const response = await request.get(`${API}/api/hitl/pending`, { headers });
      return (await json<Array<{ book_id: string }>>(response)).map((approval) => approval.book_id).sort();
    })
    .toEqual([alphaBook.id, betaBook.id].sort());
  await approveBook(page, alphaBook.id);
  await approveBook(page, betaBook.id);
  await expect
    .poll(async () => {
      const response = await request.get(`${API}/api/decisions/${tradingDecisionId}`, { headers });
      return (await json<{ status: string }>(response)).status;
    })
    .toBe('completed');

  // 页面读取真实 fake/Paper 账本成交与收益，不使用前端造数。
  for (const connection of [alpha, beta]) {
    await navigate(page, `/accounts/connections/${encodeURIComponent(connection.id)}`);
    await page.getByRole('button', { name: '刷新账户' }).click();
    await expect(page.getByText('最近成功同步：')).not.toContainText('尚未同步');
    await page.getByRole('tab', { name: '成交与收益' }).click();
    await expect(page.getByText(/BTC\/USDT · 买入/)).toBeVisible();
    await expect(page.getByRole('heading', { name: '净交易收益', exact: true })).toBeVisible();
    const ledger = await json<PaperLedger>(await request.get(`${API}/__workbench__/paper/${connection.id}`));
    expect(ledger.fills.length).toBeGreaterThan(0);
    expect(ledger.fills[0]?.side).toBe('buy');
    expect(Number(ledger.fills[0]?.amount)).toBeCloseTo(0.3, 10);
    const firstFill = ledger.fills[0]!;
    const localFillTime = await page.evaluate((timestamp) => {
      const value = new Date(timestamp);
      const part = (number: number) => String(number).padStart(2, '0');
      return `${String(value.getFullYear())}-${part(value.getMonth() + 1)}-${part(value.getDate())}T${part(value.getHours())}:${part(value.getMinutes())}`;
    }, firstFill.occurred_at);
    await page.getByLabel('开始时间').fill(localFillTime);
    await page.getByRole('button', { name: '查询' }).click();
    const incomeResponse = await request.get(
      `${API}/api/accounts/${connection.id}/income?start=${encodeURIComponent(firstFill.occurred_at)}`,
      { headers },
    );
    expect(incomeResponse.ok()).toBe(true);
    const income = await json<AccountIncome>(incomeResponse);
    const fees = income.fees.find((item) => item.currency === firstFill.fee.currency);
    const realized = income.realized_gross.find((item) => item.currency === firstFill.realized_pnl.currency);
    const funding = income.funding.find((item) => item.currency === firstFill.realized_pnl.currency);
    const net = income.net_trading.find((item) => item.currency === firstFill.realized_pnl.currency);
    expect(fees?.amount).not.toBeNull();
    expect(realized?.amount).not.toBeNull();
    expect(funding?.amount).not.toBeNull();
    expect(net?.amount).not.toBeNull();
    expect(Number(fees!.amount)).toBeCloseTo(Number(firstFill.fee.amount), 10);
    expect(Number(realized!.amount)).toBeCloseTo(Number(firstFill.realized_pnl.amount), 10);
    expect(Number(net!.amount)).toBeCloseTo(
      Number(realized!.amount) - Number(fees!.amount) + Number(funding!.amount),
      10,
    );
    const netCard = page.getByRole('heading', { name: '净交易收益', exact: true }).locator('..');
    await expect(netCard.getByText(`${displayedAmount(net!.amount!)} ${net!.currency}`, { exact: true })).toBeVisible();
    await expect(netCard.getByText(/E[+-]?\d+/i)).toHaveCount(0);
    await expect(page.getByText(income.methodology, { exact: true })).toBeVisible();
  }

  // 推进隔离测试时钟后，页面显示冻结预测的真实评估结果。
  const advanced = await request.post(`${API}/__workbench__/clock/advance`, { data: { hours: 2 } });
  expect((await json<{ completed: number }>(advanced)).completed).toBeGreaterThan(0);
  await navigate(page, '/engine/components/sample_signal');
  await page.getByRole('button', { name: '效果' }).click();
  await expect(page.getByText('方向命中')).toBeVisible();
  const tradingEvaluation = page.getByRole('heading', { name: /实时交易决策/ }).locator('..');
  await expect(tradingEvaluation.getByText('100.0%', { exact: true })).toBeVisible();
  await selectTheme(page, '深色');
  await page.screenshot({ path: path.join(EVIDENCE_DIR, '02-evaluation-1440-dark-fix1.png'), fullPage: true });

  // 停用 Alpha 池并独立退出；Beta 池保持可执行且仓位不被误平。
  await navigate(page, '/accounts/books');
  const alphaArticle = page.locator('article.configuration-section').nth(0);
  await expect(alphaArticle.getByLabel('名称')).toHaveValue('Alpha 池');
  await expect(alphaArticle.getByLabel('启用资金池')).toBeVisible();
  await setCheckbox(alphaArticle.getByLabel('启用资金池'), false);
  const disabledSaved = waitForConfigurationSave(page);
  await page.getByRole('button', { name: '保存配置' }).click();
  expect((await disabledSaved).ok()).toBe(true);
  await expect(page.getByText('本节已保存')).toBeVisible();
  await navigate(page, `/accounts/connections/${encodeURIComponent(alpha.id)}`);
  await page.getByRole('button', { name: '人工退出' }).click();
  const prepared = page.waitForResponse(
    (response) => response.url().endsWith(`/api/accounts/${alpha.id}/operations/prepare`),
    { timeout: 10_000 },
  );
  await page.getByRole('button', { name: '确认停用并读取计划' }).click();
  expect((await prepared).ok()).toBe(true);
  await expect(page.getByText(/退出计划 v/)).toBeVisible();
  const executed = page.waitForResponse(
    (response) => response.url().includes('/api/account-operations/') && response.url().endsWith('/execute'),
    { timeout: 10_000 },
  );
  await page.getByRole('button', { name: '确认执行此计划' }).click();
  expect((await executed).ok()).toBe(true);
  await expect(page.getByText('退出已完成，账户保持停用')).toBeVisible();
  await page.getByRole('button', { name: '关闭' }).click();
  await expect(page.getByRole('button', { name: '人工退出' })).toBeFocused();

  const alphaLedger = await json<PaperLedger>(await request.get(`${API}/__workbench__/paper/${alpha.id}`));
  const betaLedger = await json<PaperLedger>(await request.get(`${API}/__workbench__/paper/${beta.id}`));
  expect(alphaLedger.fills.map((fill) => fill.side)).toEqual(['buy', 'sell']);
  expect(betaLedger.fills.map((fill) => fill.side)).toEqual(['buy']);
  expect(Number(alphaLedger.fills[1]?.amount)).toBeCloseTo(Number(alphaLedger.fills[0]?.amount), 10);
  expect(Object.values(alphaLedger.positions).every((item) => item.signed_amount === '0')).toBe(true);
  expect(Object.values(betaLedger.positions).some((item) => item.signed_amount !== '0')).toBe(true);
  const scope = await json<TradingScope>(
    await request.get(`${API}/api/trading-runs/scope?pair=BTC%2FUSDT`, { headers }),
  );
  expect(scope.books.find((item) => item.book_id === alphaBook.id)?.eligible).toBe(false);
  expect(scope.books.find((item) => item.book_id === betaBook.id)?.eligible).toBe(true);

  // 再次进入后仍从后端读取已保存事实，不依赖当前页面草稿。
  await page.reload();
  await expect(page.getByText('密钥未提供')).toBeVisible();
  await unlock(page);
  await navigate(page, '/accounts/books');
  await expect(page.locator('article.configuration-section').nth(0).getByLabel('启用资金池')).not.toBeChecked();
  await expect(page.locator('article.configuration-section').nth(1).getByLabel('启用资金池')).toBeChecked();
  await page.setViewportSize({ width: 390, height: 844 });
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await page.screenshot({ path: path.join(EVIDENCE_DIR, '03-books-reentry-390-dark-fix1.png'), fullPage: true });
  await selectTheme(page, '浅色');
  await page.screenshot({ path: path.join(EVIDENCE_DIR, '04-books-reentry-390-light-fix1.png'), fullPage: true });

  expect(attempts.externalHttp).toEqual([]);
  expect(attempts.externalWebSockets).toEqual([]);
});

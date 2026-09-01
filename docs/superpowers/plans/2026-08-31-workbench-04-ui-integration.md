# 第四批：页面整合与验收 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用户用六个入口完成首次配置、次日复盘和人工退出，新后端类型使用同一份前端构建。

**Architecture:** 前三批页面按领域接到导航，工作台聚合已有状态和事项。配置草稿只包裹编辑领域，历史／账户／研究页面使用独立查询；删除旧路由和未接线控制。

**Tech Stack:** React Router、TanStack Query、既有 UI／图表／主题，Vitest 与独立 Playwright 测试环境。

**Spec:** [设计规格](/Users/rccpony/Projects/cryptotrader-ai/docs/superpowers/specs/2026-08-31-trading-workbench-design.md)，依赖前三批完整通过；任务／验收映射见[总计划](/Users/rccpony/Projects/cryptotrader-ai/docs/superpowers/plans/2026-08-31-trading-workbench.md)。

## Global Constraints

- 主导航收敛为六项。资金状态、决策过程和配置入口各有一个主位置，通过链接关联，不复制业务实现。
- 旧 `/setup` 的强制跳转取消。未就绪、暂停自动运行或尚未启用交易时，也能看首页、历史、账户配置和缺项说明。
- 本期不保留独立聊天入口。
- 统一使用中文业务名称、百分比、币种与时间格式。平台品牌和模型名称保留原名；API Key 等字段附中文含义。
- 整个日常流程无需操作参数 JSON。
- 不用真实资金验证这次界面重构；外部模型和官方模拟交易另行授权。

---

## 文件责任图

| 文件 | 操作及责任 |
| --- | --- |
| `web/src/App.tsx`、`components/layout/app-shell.tsx`、`sidebar.tsx`、`top-bar.tsx` | Modify；六入口、详情路由、全局状态 |
| `web/src/pages/workbench/index.tsx`、`first-run.tsx` | Create；事项、最近决策、分资金性质概况与渐进配置 |
| `web/src/pages/engine/index.tsx`、`accounts/index.tsx`、`research/index.tsx`、`settings/index.tsx` | Modify；接入前三批子页面和保留功能 |
| `web/src/pages/settings/configuration-context.tsx`、`navigation.ts`、`configuration-access.tsx` | Modify；领域草稿、保存／应用和访问错误 |
| `web/src/pages/settings/forms/market-settings.tsx`、`signal-settings.tsx`、`risk-settings.tsx`、`scheduler-settings.tsx` | Modify；作为引擎子表单消费后端声明与真实执行参数 |
| `web/src/pages/settings/forms/model-settings.tsx`、`system-settings.tsx` | Modify；模型／通知／安全，仅显示当前有效配置 |
| `web/src/lib/configuration-readiness.ts`、`format.ts`、`styles/globals.css` | Modify；删除前端业务就绪推断、统一中文与布局 |
| `design.md` | Modify；规范与本次规格对齐 |
| `tests/workbench_app.py`、`tests/test_workbench_route_contract.py` | Create；隔离后端夹具应用与旧接口删除证明 |
| `web/playwright.workbench.config.ts`、`web/tests/e2e/workbench.spec.ts`、`extension-catalog.spec.ts` | Create；独立端口、真实前端／假后端 E2E |
| `web/src/App.named-routes.test.tsx`、`web/src/pages/configuration-i18n.test.tsx`、`tests/conftest.py` | Modify；新路由、中文状态与过时 fixture 清理 |

## U1：六入口及完整日常页面

**Files:** 上表 App／layout／workbench／四领域页面、配置上下文、forms、readiness／format／styles、design.md；Modify `web/src/locales/zh-CN/common.json`、`configuration.json`、`web/src/test/configuration-workflow.tsx`。Test：Create `web/src/pages/workbench/workbench.test.tsx`；Modify named-routes／configuration-i18n 及既有草稿生命周期测试。

**Interfaces:** Consumes F5 ReadinessOut、F4 DecisionListOut、B2 AccountSnapshot／IncomeSummary、R4 AlertOut。Produces六主路由及通用详情；页面不生成新的风控结论。

- [ ] 先写路由／首次使用测试：inactive 时根页仍显示工作台及具体缺项；仅分析就绪、交易未就绪时只允许相应操作；无资金数据不显示0收益；再次进入显示原连接状态和已保存凭据标记。

```tsx
expect(screen.getByRole('heading', { name: '工作台' })).toBeVisible();
expect(screen.getByRole('button', { name: '仅分析一次' })).toBeEnabled();
expect(screen.getByRole('button', { name: '运行一次交易' })).toBeDisabled();
expect(screen.getByText('尚未分配到资金池')).toBeVisible();
expect(window.location.pathname).not.toBe('/setup');
```

- [ ] 在 web 目录执行总计划的 Vitest 入口，目标 `src/App.named-routes.test.tsx src/pages/workbench/workbench.test.tsx`；确认失败指向旧跳转和缺失页面。
- [ ] 配路由并移除全站 ConfigurationProvider；只有引擎／账户配置／系统设置的编辑子树使用各自领域 draft。保存时仍基于完整服务端文档 CAS 合并相应领域，冲突保留输入并提示重新载入，不让两个草稿互相覆盖。

```tsx
<Route index element={<WorkbenchPage />} />
<Route path="decisions" element={<DecisionsPage />} />
<Route path="decisions/:decisionId" element={<DecisionDetailPage />} />
<Route path="engine/*" element={<EnginePage />} />
<Route path="accounts/*" element={<AccountsPage />} />
<Route path="research/*" element={<ResearchPage />} />
<Route path="settings/*" element={<SettingsPage />} />
```

EnginePage 内注册 components/:componentId；AccountsPage 内注册 connections/:connectionId、books/:bookId；ResearchPage 内注册 backtests/:runId 和 compare。无需使用品牌名做路径。

- [ ] 工作台顺序为运行状态／操作、待处理事项、最近决策、模拟／真实账户概况；点事项到同一审批／决策对象。新手提示按后端 reasons.path 跳到具体字段，完成一项后提示下一项，不用八项全完成作为所有页面门禁。
- [ ] 引擎包含行情／上下文、信号、融合／风险、自动运行；账户使用 B2–B4 页；研究保留市场观察、仅分析和回测；系统保留模型服务、通知、安全、真实运行指标和智能体资料。复用 `pages/memory` 的资料视图及 metrics 有效内容，不声称模型已学习或有未接线的风险保护。
- [ ] 按 frontend-design／React 最佳实践技能检查具体实现：沿用字体和双主题，正文≥14px、桌面控件≥40px／触屏≥44px、焦点可见、字段错误关联。重复大卡片改紧凑行，表格窄屏明确滚动，预测区域与历史区域用文字和样式区分。
- [ ] 跑本任务测试、所有配置草稿／凭据回归及 typecheck。检查模型失败、账户同步失败、无记录、保存失败和第二次进入的界面；读取只显示更新时间，不沿用过期数据的绿色状态。

## U2：删除旧入口与失效调用方式

**Files:** `src/api/main.py`、`dependencies.py`、`routes/decisions.py`、`routes/backtest.py`、前端 App／layout／schemas；以下清单逐文件执行。Test：Create `tests/test_workbench_route_contract.py`，更新现有受影响测试而非保留旧路由。

**Interfaces:** Consumes F4–R4 新接口；Produces 无兼容别名的路由集合。有效业务能力先迁入接收位置并通过测试，再删除旧入口。

| 删除／替换目标 | 先满足的接收位置 |
| --- | --- |
| `web/src/pages/setup/index.tsx` | workbench/first-run；原 setup-page 用例改为新手流程测试 |
| `web/src/pages/dashboard/index.tsx` | workbench；旧 pnl-attribution-card 的伪收益不迁入 |
| `web/src/pages/cycles/index.tsx`、`cycle-detail.tsx` | decisions 列表／detail |
| `web/src/pages/debate/index.tsx` | F3 timeline renderer；保留可复用展示部件，删除独立路由 |
| `web/src/pages/chat/index.tsx`、`web/src/lib/chat-control.ts` | F4/F5 明确运行 API；取消聊天触发交易 |
| `web/src/pages/risk/index.tsx` | book-detail 风险／审批；有效审批部件迁入 `components/trading/` |
| `web/src/pages/strategy/index.tsx`、`scheduler/index.tsx` | engine 各子页和统一自动开关 |
| `web/src/pages/settings/venues/index.tsx`、`venue-form.tsx` | F2 accounts 连接列表／表单 |
| `web/src/pages/settings/execution-books/index.tsx`、`book-form.tsx` | B2/B4 accounts 资金池页面；有效 allocation-preview 迁入 accounts |
| `web/src/pages/backtest/index.tsx`、`market/index.tsx` | research；图表／表单有效部件迁入相邻领域目录 |
| `web/src/pages/metrics/index.tsx` | settings 的运行指标页签；保留真实指标读取 |
| `src/api/routes/cycles.py` | decisions API 及严格响应 DTO；编解码辅助函数迁入 `src/api/routes/response_dto.py` 后删除 |
| `src/api/routes/portfolio_v2.py` | accounts／portfolio_books；不迁入返回空 cycles 的收益算法 |
| `src/api/routes/risk.py` | B3 风险状态；删除旧 circuit-breaker/reset 控制 |
| `src/api/routes/chat.py`、`chat_control.py`、`src/cryptotrader/chat/analysis_runner.py` | F4/F5 RunService；task_manager 迁入 tasks.py，移除旧模块导入 |
| `src/cryptotrader/chat/task_manager.py` | F4 tasks.py；仍用于事件流的 event_bus/event_buffer 按真实引用保留，无旧入口 |
| `src/cryptotrader/configuration/catalog.py` 中包发现函数，三个 registry 中 entry points/load_factory 分支 | F1 单一代码注册 |
| `examples/configuration_plugin/pyproject.toml` | 删除安装式示例；`configuration_example.py` 改为代码注册示例，不能要求安装插件 |
| `src/api/routes/backtest.py` 的 `/run`、`/sessions` 及文件结果读取 | R3 `/runs`；DELETE `/runs/{id}` 取消动作保留并同步写 DB 状态 |
| `src/cryptotrader/runtime_config/models.py` 的 `signals.hitl_required`，旧前端全局 HITL 字段 | 各资金池 hitl_required |
| `src/cryptotrader/runtime_config/models.py` 的 system.active、setup_required；旧配置激活按钮与 scheduler.pairs | F5 能力就绪、execution.pairs 和自动总开关；迁移保留停用效果 |

- [ ] 写 canonical route 测试：新路由存在、旧 OpenAPI 路由不存在；只读接口未启用交易也可读，但未授权请求仍401。旧链接直接404，不重定向到交易操作。

```python
paths = app.openapi()["paths"]
assert "/api/analyses" in paths
assert "/api/trading-runs" in paths
assert "/api/chat/stream" not in paths
assert "/api/risk/circuit-breaker/reset" not in paths
assert "/api/portfolio/snapshot" not in paths
```

- [ ] 红测 `rtk proxy env -u DATABASE_URL -u CONFIG_MASTER_KEY .venv/bin/python -m pytest --no-cov tests/test_workbench_route_contract.py -q`。
- [ ] 按表迁移有效部件后使用 apply_patch 删除目标文件／旧分支，同时更新 import、API Schema、页面链接和测试。删除旧页面的测试前，把有效凭据、保存和审批断言移到对应新测试，不能靠删测试消除回归。
- [ ] 从 `tests/conftest.py` 移除旧 portfolio_v2 缓存、backtest _RUNS 清理，改用显式临时 store fixture。检索所有源码和脚本的旧路径／旧类型引用，修 `scripts/signal_canary.py`、`venue_canary.py` 实际命中的失效调用；本轮不运行这些外部 canary。

```sh
rtk proxy rg -n 'metadata\.entry_points|installed_plugin_factories|/api/chat|/api/cycles|/api/portfolio|circuit-breaker/reset|signals\.hitl_required' src web/src scripts
```

预期没有生产路径命中；测试可以出现验证删除的字符串。历史文档不改写成新事实，新增 README 指向本次代码注册方式。

- [ ] 运行目标路由测试、全部前端测试和类型检查；确认六导航可达、详情链接有效。用 `rtk git diff --check` 检查补丁，列明删了哪些代码文件。删除仅限 Git 可恢复源码，运行数据、旧回测备份、密钥及用户 local.toml 不在清单中。

## U3：隔离网页验收、后端扩展证明与交付记录

**Files:** tests/workbench_app、Playwright独立配置、两个 E2E、`web/package.json`（仅测试命令）、`docs/superpowers/plans/2026-08-31-trading-workbench.md`（记录执行结果）。Test：全量后端、前端、E2E。

**Interfaces:** Consumes 所有新 API。Produces 可复跑隔离测试入口和 A1–A22 的证据索引；不生成生产收益样本。

- [ ] 先建专用 test app：显式临时 SQLite／测试密钥，注入 F1 sample registry、固定行情／模型和 fake Webhook。不得读取用户 Bootstrap 环境或 local.toml。测试启动命令在 `tests/workbench_app.py` 内构造测试 Runtime，原 API 路由不因 fixture 改业务逻辑。
- [ ] Playwright配置使用 API8011、前端4173，两个服务器均 `reuseExistingServer:false`。child process 使用已验证的绝对 Python／Node 路径；浏览器请求只能到两个测试端口。创建目录使用 mktemp/测试 temp fixture，禁止对运行地址5173/8000执行写请求。

```ts
use: { baseURL: 'http://127.0.0.1:4173' },
webServer: [
  { command: backendTestCommand, url: 'http://127.0.0.1:8011/health', reuseExistingServer: false },
  { command: frontendPreviewCommand, url: 'http://127.0.0.1:4173', reuseExistingServer: false },
],
```

`backendTestCommand` 在该配置中定义为仓库 `.venv/bin/python -m uvicorn tests.workbench_app:app --host 127.0.0.1 --port 8011`；`frontendPreviewCommand` 使用 `process.execPath` 调用 Vite preview，host127.0.0.1/port4173/strictPort。构建时 `VITE_API_BASE_URL` 明确指向测试API，不继承运行地址。

- [ ] 写完整场景：无配置打开首页→配置 sample signal→仅分析→保存模拟连接和自定义凭据自动检查→分配资金池→确认全局模拟交易→按池审批→核对成交收益→推进测试时钟做评估→停用指定池并人工退出。另一池继续可用。所有断言通过网页可见结果和测试账本，不用伪 UI 数字通过。
- [ ] 做 A1/A2 扩展证明：先构建前端并记录 dist 文件 hash；启动第二个隔离后端配置，只在其代码注册表增加 sample_venue/sample_signal，再复用相同 dist。网页出现新字段、新凭据、新环境和结果块，完成账户持仓／成交及退出。测试代码可声明类型，生产前端源码与 dist hash 均不变。
- [ ] 覆盖首次／失败／再次进入，以及1440px和390px、深浅主题、键盘保存／确认／返回焦点。手动浏览核验用 browser 技能打开隔离测试地址，检查真实可见界面和截图，不能把 DOM 元素存在当作可读性通过。更新 design.md 的实际布局标准。
- [ ] 在仓库根目录运行全量后端，在 web 工作目录运行前端、类型、构建和 E2E；首次记录已有基线失败，不降低覆盖门槛或把失败隐藏为通过。

```sh
rtk proxy env -u DATABASE_URL -u CONFIG_MASTER_KEY .venv/bin/python -m pytest tests/ -q
```

```sh
rtk proxy /Users/rccpony/.nvm/versions/node/v24.19.0/bin/node node_modules/vitest/vitest.mjs run
rtk proxy /Users/rccpony/.nvm/versions/node/v24.19.0/bin/node node_modules/typescript/bin/tsc -b
rtk proxy /Users/rccpony/.nvm/versions/node/v24.19.0/bin/node node_modules/eslint/bin/eslint.js .
rtk proxy env VITE_API_BASE_URL=http://127.0.0.1:8011 /Users/rccpony/.nvm/versions/node/v24.19.0/bin/node node_modules/vite/bin/vite.js build
rtk proxy /Users/rccpony/.nvm/versions/node/v24.19.0/bin/node node_modules/@playwright/test/cli.js test --config playwright.workbench.config.ts
```

- [ ] 使用 verification-before-completion 核对实际结果，再按 requesting-code-review 技能做独立审查；该技能如要求审查子代理，只派发审查范围，不据此并行实施其他任务。发现未覆盖的资金／审批／历史问题先修正重测，不自行创建新的用户任务或推送分支。
- [ ] 交付 A1–A22 结果表、关键截图、代码删除清单、未验证外部能力及运行库迁移步骤。此时可说明“隔离测试通过”；只有用户另行授权后，才备份／迁移运行库、重启、调用真实模型和官方模拟盘验证。提交／合并／推送也须对应授权。

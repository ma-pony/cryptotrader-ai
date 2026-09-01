# 配置中心验收记录

日期：2026-08-30。记录 Task 6 集成验收、最终评审修复和交付前真实页面回验。独立复审已确认最初六项发现全部修复；真实回验发现的连接更新请求契约问题也已补修、独立复审并通过实际页面验证。未合并、部署或启用交易。

## 最终检查

| 检查 | 实际结果 |
| --- | --- |
| `.venv/bin/pytest --no-cov -q` | 控制任务独立复跑：2119 passed，1 skipped，8 warnings，48.30 秒，exit 0 |
| `node node_modules/vitest/vitest.mjs run` | 连接更新补修后：45 files，252 passed，exit 0 |
| `node node_modules/typescript/bin/tsc --noEmit` | exit 0 |
| `node node_modules/eslint/bin/eslint.js .` | exit 0 |
| `node node_modules/vite/bin/vite.js build` | 连接更新补修后：2226 modules，exit 0；同源隔离预览构建 2.02 秒，exit 0 |
| 本轮 7 个 Python 文件 Ruff check / format --check | exit 0 |
| `git diff --check` | exit 0 |

Node 使用 `/Users/rccpony/.nvm/versions/node/v24.19.0/bin/node`。最终六项修复提交为 `94bb61427f47803950cc4aaec1d45766d11d7e57`。实现任务串行检查为后端 2119 passed / 1 skipped / 8 warnings（50.99 秒）、前端 251 passed（14.64 秒）；控制任务另行独立复跑得到相同计数（后端 48.30 秒、前端 18.88 秒），TypeScript、ESLint 和差异检查均 exit 0。隔离预览构建 2226 modules / 1.67 秒 / exit 0。所有长检查等待到实际进程退出。此前 Task 6 的完整结果为后端 2113 passed / 1 skipped / 8 warnings（76.85 秒），前端 237 passed（10.78 秒）。

后端八条警告与此前相同：一个未注册的 benchmark mark，一个 LangGraph pending deprecation，以及两个既有 Redis mock 测试中的六条 AsyncMock 未 await 警告。未隐藏警告。

首次完整前端检查与后端、ESLint、TypeScript 和浏览器同时运行，出现七个失败：forms 全字段测试、books 跨写入测试超时；strategy 保存、两个 setup 测试、venue draft reload 和 books identity 测试在等待懒加载页面/字段时失败。共 230 passed、7 failed，28.90 秒。停止并发重负载后同一完整命令通过，期间没有修改代码、断言或超时。结果符合负载相关时序问题，但单次复跑不证明每项失败的唯一原因；保留该限制供 CI 复核。

## 最终评审修复

基线：`bbf62059a303f6fd23f638c08f9cbb3d3a158449`。六项发现同一轮处理；限定范围独立复审确认全部 ADDRESSED，无新增 Critical/Important 或范围外发现。以下为自动化证据，页面补充回验见后文。

| 发现 | 修复及验证 |
| --- | --- |
| 成功轮换凭据后应用状态陈旧 | 写入确认后刷新公共配置。活跃系统轮换后仍显示正确激活状态，其他分区草稿保留；API 密钥在确认后仅内存交接并用于刷新认证。刷新失败明确显示已保存但需重新加载，缓存不含凭据。 |
| 首次启用遗漏 Redis | 检查清单校验待启用状态，系统行说明 Redis 前提并链接配置表单。空 Redis 禁用激活；仍允许未启用时保存其他系统字段。 |
| Paper 无实际逐仓行为 | 现有目录新增代码定义的 `margin_modes`，表单据此展示。Paper 固定全仓、共享账户权益，保留杠杆；OKX/Bybit 的 Cross/Isolated 可选项保留。 |
| gt/lt 丢失独占语义 | 目录独立传输独占上下界，ge/le 仍为包含边界；参数表单共用数值校验。零 Paper 资金显示字段错误并聚焦，0.00001 可保存，服务端零资金落库前拒绝不变。 |
| 风控监控术语错误 | 中英文改为连接名义敞口上限；风险计算没有变化。 |
| 测试名称夸大键盘覆盖 | 改名为带标签选择器路由、焦点和草稿保留。下文原生键盘工具限制保留。 |

凭据状态自查还覆盖发布失败：后端可能保存新版本后返回 503，此时旧缓存不能证明系统当前激活。页面要求显式重新加载，再显示服务端失败状态；没有推断回滚，也没有使用未经确认的新 API 密钥。

TDD 实际结果：后端目录 RED 6 failed / 10 passed（缺少能力和独占边界），hooks RED 5 failed / 14 passed（旧应用版本和独占边界误放行），页面 RED 5 failed / 14 passed。初始 GREEN 为前端 7 files / 48 passed、后端契约 104 passed。发布失败补充 RED 1 failed（503 后仍声称已激活），对应 GREEN 4 files / 24 passed。最终完整结果见上表；未修改超时或隐藏警告。

- [x] 六项修复及失败发布状态回归完成，最终串行检查通过。
- [x] 控制任务关闭六项发现的限定范围独立复审，并刷新隔离预览复核。
- [x] 交付前连接更新契约补修、定点复审及真实页面回验完成。
- [ ] 另行授权的部署与旧数据清理。

## 交付前真实回验

最终构建以独立临时目录提供。旧验收标签页因未保存草稿仍停留在旧文档；新建自有验收标签页后，实际 DOM 的脚本地址与服务端最终构建一致，再继续检查。没有改动产品来绕过浏览器行为，也未操作用户的 5173 页面。

- 初始化清单实际显示 Redis 必需说明与系统配置入口，激活按钮不可用。
- Paper 实际显示固定全仓和共享账户权益说明，保留杠杆；资金 0 显示关联错误并聚焦输入，`aria-invalid=true`；0.00001 创建成功，配置版本从 1 到 2。
- OKX 的未保存表单仍提供 Demo/Live、全仓/逐仓、Key/Secret/Passphrase 及对应说明。没有保存外部连接或凭据。
- 随后把已创建 Paper 改为 10000 保存时，真实 API 拒绝了请求中的只读 `id`；原值与版本保持不变，页面保留草稿并禁用检查。后端模型独立复现 `id: extra_forbidden`，移除该字段的同一请求合法。未以此前全绿的宽松模拟测试代替真实验收。
- 补修提交 `3314425351a67e15573b9e742ae7017f45ea14ab` 明确区分创建和更新请求：创建携带 ID，更新只在路径携带 ID，请求体仅含可写字段；后端严格校验不变。严格模拟 API 的测试先 RED 2 failed / 5 passed（版本未递增），再 GREEN 7 passed。最终相关前端 12 passed、真实 venue API 17 passed（1 条既有第三方警告）、完整前端 252 passed；TypeScript、ESLint、构建和差异检查通过。后端生产代码未改，完整后端结果沿用上方控制任务的独立检查。
- 独立限定范围复审确认该项 ADDRESSED、spec/quality PASS，没有新增缺陷。控制任务重新构建并核对浏览器实际脚本版本后，重走真实路径：0.00001 改为 10000 保存成功，配置版本 2 → 3；整页重载后金额与版本保留，连接 ID 不变且不可编辑，保存按钮在无修改时禁用，只读检查成功（18:53:44）。所有写入仅影响临时 fixture。

执行裁定：在原六项最终修复已复审通过后，追加一次限定范围的连接更新修复与独立差异复审。原因是实测证明它违反已批准的第二次访问编辑要求；误判代价是额外一轮有限验证和请求契约返工。没有重开全项目审查、放宽后端校验或扩大外部操作权限。

## 真实路径证据

浏览器使用 in-app Browser 的已文档化 Node 客户端，没有独立 Playwright、CDP 或隐藏 React 状态注入。`browser.capabilities.get('viewport').set({width,height})` 调整真实渲染 viewport；未使用 iframe。八个配置分区在 320、375、414、768 和 1280 CSS px 均已访问，`innerWidth` 与根元素 `scrollWidth` 相同，可见表单控件越界数量为零。320/375/414/768 高度为 900，最后桌面测量为 1280×1000。

| 路径 | 浏览器 / API 实际观察 | 补充自动化覆盖 |
| --- | --- | --- |
| 初始化/日常配置 | 未启用可访问八节；分区保存并回访 | setup、configuration workflow |
| 类型化代码注册组件 | 后端注册 `configuration_example` 后出现窗口、周期和嵌套高级开关；窗口 24、15m 和 diagnostics 保存后仍在 | `test_configuration_acceptance.py`、catalog/field suites |
| 跨节编辑 | model draft 保留，保存 risk 7.5% 后 GET 得到 0.075，再保存 model；未启用状态不变 | draft lifecycle、workflow |
| Paper | 默认 10000，改为 12500，创建/回访/实际本地只读检查成功；无凭据字段 | venue/model/API suites |
| 外部平台 | OKX 仅 Demo/Live；缺凭据检查失败，缺 Passphrase 不能保存凭据；公共测试字符串写 vault 后，假外部适配器返回 authentication_failed | real adapter read-only/close、credential redaction suites |
| Books | 保存 simulated book，100% 总计/0% 剩余；2000×25%×100%=500 USDT 示例，ID/scope 保存后固定，live 开关关闭 | disabled/canary/wrong-scope 拒绝与稳定输入测试 |
| 冲突 | 第二 HTTP writer 生成 revision 10，浏览器 revision 9 保存返回 409；9% draft 保留，Reload 后仍保留，可再保存 | CAS/API、workflow |
| 保存/应用分离 | 故意发布失败后 DB revision 11、applied 10、drawdown 0.09；Reload 显示“已保存但应用失败”，随后新保存恢复 applied | application barrier、failed publication、HITL revision tests |
| 错误与焦点 | 空数字保持空，错误关联到字段且聚焦；320px dirty savebar 三个按钮均可见 | numeric field、validation、discard lifecycle |
| 回测 | saved session 加载 ETH/USDT、2025-01-01 至 02-01、2500；被 fixture 拒绝的启动显示错误并保留参数；空资金聚焦错误 | reversed/future dates、prior-output preservation |
| 规则 | 320px 新建规则缺名称/目标价显示错误并聚焦名称，dialog 操作可见 | operational required values、request failure、percentage-point units |
| 安全/消费者 | 未激活、未执行订单或模型；真实 API-key 认证保留 | actual-auth fixture、HITL TTL memory/SQL、四项风险、news vault、webhook/OTel consumer tests |

认证 browser walk 使用受保护 fixture：错误的公共测试口令显示验证失败并清空输入；正确口令解锁同一 revision 3 的清单，未轮换密钥或保存配置。规则请求失败由自动化测试覆盖；开启 fixture 认证后，旧 operational 页收到 401 并退出 dialog，因此没有把这一步算作浏览器规则保存失败验收。

## 视觉证据与限制

以下是未修改的实际截图。除 scrolled dirty-state 外，代表截图包含黄色隔离验收标识。页面中的值均为临时测试数据。

- [320px 分区选择器与可见焦点](mobile-320-selector.png)
- [320px 未保存修改与完整保存操作](mobile-320-dirty.png)：滚动后标识在视口外，仍为同一隔离 fixture。
- [375px 浅色风控字段](mobile-375-light.png)
- [414px English 深色风控字段](mobile-414-dark-en.png)
- [桌面回测字段关联错误](desktop-backtest-error.png)
- [桌面未启用配置预览](desktop-inactive-dark.png)，默认 viewport 重置后为 1280×720。
- [最终中文初始化清单](desktop-final-setup-zh.png)：最终补修构建，1280×720，临时配置版本 3；黄色隔离标识和 Redis 前提可见，底部内容可正常滚动查看。

移动导航原先在 414px 换成 4/3/1 行；本次改为带标签的原生 selector，保留同样八个路由，桌面 N3 导航不变。连接检查成功时间改用既有 locale-aware formatter，API 原始时间保留在 `<time dateTime>`。

从真实 DOM 读取颜色并按 WCAG 2.1 sRGB 相对亮度计算：浅色正文 17.87:1、帮助 4.70:1、主按钮 4.96:1；深色正文 18.68:1、帮助 9.40:1、主按钮 5.37:1；浅色错误 5.41:1、深色错误 5.69:1。浅色 focus ring 5.17:1，2px solid、offset 2px，显示无过渡。错误 Lab 颜色先按 D50→D65 转为 sRGB。这里是实际取样的表单颜色，不声称已遍历全站每个颜色状态。

移动表单与保存按钮测得 44px 高；桌面保持已有紧凑 40px 规格。Native disabled 属性、not-allowed cursor、透明度和错误帮助槽保留。Hallmark 复查沿用已批准的 Workbench、系统字体、语义 token、N3 rail/C4 sticky bar；字体更换、跨页主题轮换、营销 hero/装饰/影片门槛不适用。未声称“58项全部通过”，也未测量任意 320–1920 连续宽度或所有 hover 组合。

浏览器工具限制：原生 discard confirm 导致自有 tab 1 操作阻塞；保留该 tab，另建自有 tab 2 继续，未关闭或修改用户标签页。discard 由自动化测试覆盖，未宣称浏览器原生确认完成。原生 select 可聚焦、显示焦点并通过 selectOption 正常路由，但工具 ArrowDown/Enter 未改变选择；原生日期 fill 未提交新值，逆序日期只以自动化测试证明。没有用脚本覆盖 confirm、注入状态或改产品绕过工具。

## 复现隔离预览

在项目 worktree 根目录执行，要求已经安装项目开发依赖。使用独立代码注册示例目标目录和临时 SQLite，以下变量不涉及用户运行配置：

```bash
fixture_dir=$(mktemp -d /private/tmp/cryptotrader-configuration.XXXXXX)
uv pip install --python .venv/bin/python --target "$fixture_dir/components" \
  --no-deps --no-build-isolation examples/configuration_plugin
cd web
VITE_API_BASE_URL= VITE_OTLP_UI_ENDPOINT= node node_modules/vite/bin/vite.js build \
  --outDir "$fixture_dir/spa"
cd ..
PYTHONPATH="$fixture_dir/components" .venv/bin/python -m tests.manual.configuration_preview \
  --spa-dir "$fixture_dir/spa" --port 8765
```

打开 `http://127.0.0.1:8765/setup`，确认黄色“隔离验收数据”标识后再操作。构建显式设置同源 API 和空 UI telemetry endpoint；不要省略后复用指向用户 API 的旧构建。外部 outDir 的 Vite 警告仅表示不会自动清空该目录。

预览使用真实 config/catalog/venue routers、repository、`verify_api_key` 和本地 Paper adapter。`PreviewRuntime` 不构造生产 Runtime、scheduler、Redis client、模型、市场源或执行周期；外部 adapters 一律拒绝认证。所有 activation/live-enable 和订单/模型/运行请求被阻止。保存和凭据写入仅影响临时 fixture DB，固定测试 master key 绝不能用于真实环境。

`--active-ui` 仅预置 `system.active=true` 的测试文档以显示 operational forms，没有运行时激活行为。公开测试控制端点 `/__fixture__/fail-next-publication` 和 `/__fixture__/protect` 分别制造发布失败和配置访问认证；后者的临时测试口令固定为 `fixture-access-only`。这些端点只存在于测试 harness，不进入生产 API。

CSP 的 `connect-src 'self'` 和 `script-src 'self'` 阻止旧 TopBar Binance WSS、TradingView CDN 及内联启动脚本。离线/连接中状态是 fixture 的已知限制；没有连接外部行情，未为验收改写生产 feed。早期 fixture 未提供 scheduler status，曾出现 fixture 404；最终 harness 已提供明确的暂停状态。API 服务只绑定 `127.0.0.1`。

停止自己的前台预览进程即可清理它拥有的临时 DB。保留截图直到评审完成；安装目标和 SPA 构建目录是可丢弃产物。不要停止用户 Compose 容器，不要操作其 5173/8003 应用或数据库。

部署前旧字段清理的精确路径、备份和权限边界见 [配置指南](../../CONFIGURATION.md#已移除字段与部署前清理)。本次没有执行清理、启动交易、部署、合并或推送。

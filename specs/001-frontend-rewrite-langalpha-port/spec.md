# Feature Specification: 前端重写 — LangAlpha 移植 + Crypto 化

**Feature Branch**: `001-frontend-rewrite-langalpha-port`
**Created**: 2026-04-16
**Status**: Draft
**Input**: User description: "完全弃用 Streamlit dashboard，照搬 LangAlpha React+Vite+TS 前端，crypto 化所有股票语义"

## User Scenarios & Testing *(mandatory)*

### User Story 1 — 自动交易监控（总览页，Priority: P1）

作为运维者，我打开前端首页（总览/Dashboard），无需登录即可看到当前自动交易系统的关键状态：投资组合总权益、可用现金、24 小时盈亏、当前回撤；下方是权益曲线（可切换 24h/7d/30d/全部时间）；持仓表显示每个币对的方向（做多/做空）、数量、均价、未实现盈亏；调度器卡片显示下一次自动交易触发的币对与剩余倒计时。页面每 10 秒自动刷新数据。

**Why this priority**：自动交易系统的运维者每天最高频的操作就是确认"系统活着、钱没出问题"。总览页是 5 大业务页面中**信息密度最高、被访问最频繁**的入口，必须 P1 完成。

**Independent Test**：在已运行自动交易系统的环境下打开 `/`，看到 4 个 metric 卡 + 权益曲线 + 持仓表 + 调度器卡，且 10 秒后数字自动刷新（无需手动 reload）。

**Acceptance Scenarios**:

1. **Given** 后端 API 正常、Redis 可用、有持仓数据，**When** 用户访问 `/`，**Then** 页面在 2 秒内显示 4 个 metric 卡、权益曲线、持仓表、调度器倒计时
2. **Given** Redis 不可用，**When** 用户访问 `/`，**Then** 调度器卡片显示警告信息"调度器状态不可用 — Redis 未连接"，其它卡片正常显示
3. **Given** 用户停留在页面 10 秒，**When** 计时器触发，**Then** 数据自动重新拉取并刷新视图（不重载整页）
4. **Given** 用户切换权益曲线时间范围从 24h 到 7d，**When** 点击切换器，**Then** 曲线重新拉取 7 天数据并平滑过渡

---

### User Story 2 — 决策复盘（实时决策页，Priority: P1）

作为策略研究员，我进入"实时决策"页面，按币对（如 BTC/USDT）和日期范围筛选历史决策记录。决策列表分页显示（默认 20/页），每行展示时间戳、币对、价格、裁决动作（做多/做空/持有）、信心度、是否成交。点击任一行展开右侧详情面板，看到该决策的完整管线：节点执行时间线、4 个 Agent 分析（NewsAgent/MacroAgent/SentimentAgent/TechnicalAgent，含评分/信心度/Markdown 推理）、经验记忆（成功模式/禁区/战略洞察）、Bull/Bear 双轮辩论、最终裁决（含是否走加权降级）、风控门检查序列（✓/✗ 含拒绝原因）、订单执行结果。若环境配置了 OTel 端点，trace_id 渲染为可点击链接跳转 Jaeger。

**Why this priority**：策略迭代的核心反馈循环依赖决策可追溯性。研究员每次调整 prompt/参数后都需要回看决策轨迹，确认 AI 推理符合预期。这是**策略研发的必备工具**，P1。

**Independent Test**：在有历史决策数据的环境下访问 `/decisions`，按 BTC/USDT 筛选，点击第一行，详情面板显示 8 个 section 全部有内容（除非该决策本身缺失某节）。

**Acceptance Scenarios**:

1. **Given** 数据库有 100+ 决策记录，**When** 用户访问 `/decisions` 选 BTC/USDT 过滤，**Then** 列表显示该币对的决策，分页器显示总数
2. **Given** 用户点击列表中第 5 行，**When** 行被选中，**Then** 详情面板显示节点时间线、Agent 分析、辩论、裁决等完整内容；URL 同步追加 `?id=<commit_hash>`
3. **Given** 用户复制带 `?id=xxx` 的 URL 在新标签页打开，**When** 页面加载，**Then** 直接定位到该决策详情
4. **Given** 决策的 trace_id 存在且环境变量 `OTLP_ENDPOINT` 配置，**When** 详情头部渲染，**Then** trace_id 显示为可点击链接

---

### User Story 3 — 回测新会话（回测页，Priority: P1）

作为研究员，我进入"回测"页面，在"新建回测"标签填写表单：开始日期、结束日期、币对、初始资金、策略模式（rules / llm）、可选会话名。提交后页面显示进度提示，每 5 秒轮询任务状态；完成后自动切换到结果视图，显示 5 项核心 metric（总收益率/夏普比率/最大回撤/胜率/交易次数）、权益曲线（lightweight-charts 折线图）、决策时间线表（仅 LLM 模式有数据）。点击决策时间线任一行可展开复用"实时决策"详情组件。在"加载历史会话"标签可下拉选择已保存会话重新查看。

**Why this priority**：回测是策略验证的核心工作流，必须保留 Streamlit 现有能力且体验不能倒退。P1。

**Independent Test**：访问 `/backtest`，填写有效参数提交，看到进度 → 完成后展示 5 项 metric + 权益曲线 + 决策时间线；保存的会话可在历史标签重新加载。

**Acceptance Scenarios**:

1. **Given** 用户填写完整表单（含合法日期范围、币对、初始资金 ≥ 100），**When** 点击"运行回测"，**Then** 提交成功；前端进度卡显示"运行中"+ 取消按钮
2. **Given** 回测任务完成，**When** 轮询返回 `status: completed`，**Then** 自动切到结果视图，所有 metric 与曲线渲染完毕
3. **Given** 策略模式为 `rules`（无 LLM 决策），**When** 结果加载，**Then** 隐藏 Agent 分析区，仅显示曲线 + 交易表
4. **Given** 用户在"加载历史会话"标签下拉选择已保存会话名，**When** 选定后，**Then** 结果视图加载该会话完整数据
5. **Given** 用户在回测进行中点击"取消"按钮，**When** 取消确认，**Then** 前端停止轮询并显示"已取消"提示

---

### User Story 4 — 风控状态管理（风控页，Priority: P1）

作为运维者，我进入"风控"页面，看到当前小时与当日累计交易次数、断路器状态卡（ACTIVE 显红色容器 + 剩余 TTL 倒计时；INACTIVE 显绿色）、关键风控阈值（最大持仓比例、单日最大亏损、最大止损百分比等）。当断路器为 ACTIVE 时可点击"重置断路器"按钮（带二次确认对话框），成功后 toast 通知 + 状态自动刷新。Redis 不可达时整页显示 warning 但不报错。

**Why this priority**：风控是自动交易系统的安全开关。出现异常时必须能在 5 秒内打开此页确认状态、必要时手动重置断路器恢复交易。P1。

**Independent Test**：触发断路器使其进入 ACTIVE → 访问 `/risk` 看到红色卡 + 倒计时 → 点击重置 → 确认对话框 → 提交 → 看到成功 toast + 卡片转为绿色。

**Acceptance Scenarios**:

1. **Given** 断路器处于 ACTIVE 状态，**When** 用户访问 `/risk`，**Then** 断路器卡显示红色背景 + 剩余 TTL 倒计时 + "重置"按钮可用
2. **Given** 用户点击"重置断路器"，**When** 弹出 confirm dialog，**Then** 必须用户主动确认才发起请求
3. **Given** 重置请求成功返回 200，**When** 响应到达，**Then** toast 提示"断路器已重置"，状态自动 refetch，卡片转为 INACTIVE 绿色
4. **Given** Redis 不可达（API 返回 503），**When** 页面加载，**Then** 显示警告"风控状态不可用 — Redis 未连接"且不抛错

---

### User Story 5 — 系统指标观测（指标页，Priority: P1）

作为运维者，我进入"指标"页面，看到 Prometheus 关键计数器（交易总数/订单成功/订单失败/风控拒绝次数）、延迟分位数卡（管线 P50/P95、执行 P50/P95），以及 4 条延迟趋势曲线（管线 P50/P95、执行 P50/P95，最多保留 60 个采样点）。趋势数据持久化到浏览器本地存储，刷新页面不丢失历史。

**Why this priority**：观测系统延迟与失败率是 SRE 日常工作。Streamlit 当前已有此能力，必须 1:1 保留。P1。

**Independent Test**：访问 `/metrics` 等待 30 秒，趋势曲线应至少有 2 个采样点；刷新浏览器，历史样本仍然存在。

**Acceptance Scenarios**:

1. **Given** 后端 `/api/metrics/summary` 返回有效数据，**When** 用户访问 `/metrics`，**Then** 计数器卡 + 分位数卡正确渲染
2. **Given** 页面停留 30 秒，**When** 后台每 15 秒拉取一次新数据，**Then** 趋势图新增 2 个采样点
3. **Given** 累计样本超过 60 个，**When** 第 61 个样本到达，**Then** 最旧样本被丢弃（FIFO）
4. **Given** 用户刷新浏览器后重新打开，**When** 页面加载，**Then** 之前累积的趋势样本从本地存储恢复

---

### User Story 6 — Ad-hoc AI 对话分析（AI 对话页，Priority: P2）

作为研究员，我进入 AI 对话页，输入"分析 BTC 当前是否适合开仓"。系统通过 SSE 流式返回多代理推理过程（消息分块/工具调用/工具结果/内联组件/裁决），消息流中可内联渲染 K 线图、资金费率表、链上指标等沙盒化 widget。

**Why this priority**：增量能力，非 Streamlit 现有功能。基础设施先就位，深度交互可后续迭代。P2。

**Independent Test**：访问 `/chat`，发送一条消息，观察到流式内容逐字显示、至少触发一次工具调用、最终展示裁决卡。

**Acceptance Scenarios**:

1. **Given** 后端 SSE 端点可用，**When** 用户输入消息并发送，**Then** 消息流接收到至少 5 类事件（message_chunk / tool_call / tool_result / inline_widget / verdict）
2. **Given** SSE 流中包含 inline_widget 事件，**When** 事件到达，**Then** widget 在 iframe sandbox 中安全渲染，外层不可访问 DOM
3. **Given** SSE 连接中途断开，**When** 检测到断连，**Then** 自动重连一次；失败则显示"重新连接"按钮

---

### User Story 7 — 加密货币市场看板（市场看板页，Priority: P2）

作为研究员，我进入市场看板页，输入 `ETH/USDT`，看到 TradingView 高级图表 + 侧栏（24h 资金费率、未平仓量、永续与现货价差、清算热图）。可在 Binance 与 OKX 之间切换交易所。

**Why this priority**：增量能力，提升 ad-hoc 市场调研效率。P2。

**Independent Test**：访问 `/market`，输入 `ETH/USDT`，TradingView widget 渲染图表，侧栏显示资金费率等指标。

**Acceptance Scenarios**:

1. **Given** 用户在地址栏输入 `ETH/USDT`，**When** 提交查询，**Then** TradingView widget 加载 ETH/USDT 图表
2. **Given** 用户切换交易所从 Binance 到 OKX，**When** 选择 OKX，**Then** 图表与侧栏数据源切换，重新拉取
3. **Given** TradingView script 加载失败（网络/CSP），**When** 检测到失败，**Then** 降级显示纯 lightweight-charts K 线图

---

### Edge Cases

- **EC-1 后端不可达**：所有页面统一 ErrorState 组件 + Toast 提示"后端不可达，请检查 API 服务"，不出现白屏
- **EC-2 Redis 不可用**：风控页与总览页调度器卡显示降级 warning（FR-104 / FR-405）
- **EC-3 决策 commit_hash 不存在**：404 toast + 列表保留选中清空
- **EC-4 回测长任务（>5 分钟）**：进度卡显示"已运行 N 分钟" + 取消按钮（FR-302）
- **EC-5 SSE 断连（AI 对话）**：自动重连 1 次，失败显示重试按钮（FR-602）
- **EC-6 主题/语言切换闪烁**：首屏从本地存储同步取值，避免 FOUC
- **EC-7 API key 配置缺失**：FastAPI 401 → 前端 Toast 提示"未配置 API Key，请检查 .env"
- **EC-8 浏览器宽度 < 1024px**：显示"桌面专用"提示（不做移动端适配）
- **EC-9 Streamlit 删除后用户旧链接 :8501**：根 README 与 CHANGELOG 公告新地址 5173；不做 :8501 → :5173 自动重定向
- **EC-10 i18n 翻译键缺失**：缺失键自动回退到 zh-CN（不暴露原始 key）
- **EC-11 lightweight-charts 在低端机上 1k 数据点卡顿**：超过 5k 点降级为蜡烛图聚合（按时间窗口）
- **EC-12 TradingView CSP 拦截**：单独域白名单；不行降级到 lightweight-charts

## Requirements *(mandatory)*

### Functional Requirements

#### 1. 项目脚手架与基础设施

- **FR-001**：系统 MUST 在仓库根新建 `web/` 子目录作为独立前端项目，使用 pnpm 作为包管理器
- **FR-002**：系统 MUST 采用 React 19.2 + Vite 7 + TypeScript 5.9（strict mode）+ Tailwind CSS 3 + Radix UI primitives + class-variance-authority 作为核心技术栈
- **FR-003**：系统 MUST 实现 React Router v7 客户端路由：`/` `/decisions` `/backtest` `/risk` `/metrics` `/chat` `/market`
- **FR-004**：系统 MUST 按以下顺序组装 Provider 栈：QueryClient → BrowserRouter → ThemeProvider → I18nProvider → App + Toaster
- **FR-005**：系统 MUST 实现暗/亮主题切换（基于 CSS variables），默认跟随 `prefers-color-scheme`，可手动切换并持久化到 localStorage
- **FR-006**：系统 MUST 实现 i18next 多语言框架：默认 `zh-CN`，同时提供 `en-US` 翻译；TopBar 提供语言切换器并持久化；翻译文件按 namespace 分文件（common/dashboard/decisions/backtest/risk/metrics/chat/market）；缺失键回退到 zh-CN
- **FR-007**：系统 MUST 实现 SSE 客户端（基于 fetch + ReadableStream，移植自 LangAlpha streamFetch），处理 429/413/404 结构化错误；HTTP 客户端使用原生 fetch + React Query 5。所有 API/SSE 调用的 `X-API-Key` 仅从 `useSettingsStore` 读取，**不得**回退到 `env.VITE_API_KEY`（详见 NFR-S-001 / NFR-S-005）。SSE 30s 不活跃必须发 `: keepalive\n\n` 心跳（捕获 `asyncio.TimeoutError`，注意 Py3.10 不与 `builtins.TimeoutError` 等价）。
- **FR-008**：系统 MUST 实现全局 ErrorBoundary 与 Toast 通知组件
- **FR-009**：系统 MUST 通过环境变量 `VITE_API_BASE_URL` 配置 API 地址（默认 `http://localhost:8003`）；dev 模式经 Vite proxy 转发避免 CORS
- **FR-010**：系统 MUST 提供共享 UI 原子组件：Button / Card / Dialog / DropdownMenu / Popover / ScrollArea / Toast / Skeleton / Tabs / Badge / Separator / Tooltip
- **FR-011**：系统 MUST 提供共享布局：Sidebar（左侧导航）+ Main（页面容器）+ TopBar（主题/语言切换）

#### 2. 总览页（Dashboard，P1，替换 Streamlit Overview）

- **FR-100**：路由 `/` MUST 显示投资组合总览
- **FR-101**：MUST 显示 4 项 metric 卡：总权益、可用现金、24 小时盈亏（带涨跌色）、当前回撤
- **FR-102**：MUST 渲染权益曲线（lightweight-charts 折线图），支持时间范围切换：24h / 7d / 30d / 全部
- **FR-103**：MUST 渲染持仓表，列：币对 / 方向（做多/做空 Badge）/ 数量 / 均价 / 未实现盈亏，按未实现盈亏排序
- **FR-104**：MUST 渲染调度器状态卡：下次触发币对 + 触发时间倒计时；Redis 不可用显示警告
- **FR-105**：MUST 通过 React Query 拉取数据，`refetchInterval: 10000`（保持 Streamlit 10s 刷新行为）
- **FR-106**：SHOULD 加载态显示 Skeleton；错误态显示 Toast + 错误卡

#### 3. 实时决策页（Decisions，P1，替换 Streamlit Live Decisions）

- **FR-200**：路由 `/decisions` MUST 显示决策列表 + 详情 split view
- **FR-201**：MUST 提供筛选栏：币对下拉 + 日期范围 + 分页（默认 20/页）
- **FR-202**：MUST 显示列表行：时间戳 / 币对 / 价格 / 裁决（动作+信心度）/ 是否成交；点击展开右侧详情
- **FR-203**：MUST 在详情头显示：时间戳 / 币对 / 价格 / trace_id（若 `OTLP_ENDPOINT` 存在则渲染为 Jaeger 链接）
- **FR-204**：MUST 渲染节点执行 Pipeline（水平 stepper：节点序列与耗时）
- **FR-205**：MUST 渲染 Agent 分析网格：4 卡片（NewsAgent / MacroAgent / SentimentAgent / TechnicalAgent），各显示 score / confidence / reasoning（Markdown 渲染）
- **FR-206**：MUST 渲染经验记忆折叠节：success_patterns / forbidden_zones / strategic_insights
- **FR-207**：MUST 渲染辩论 Section：Bull/Bear 双轮辩论消息流（气泡风格）
- **FR-208**：MUST 渲染裁决 Section：action / size / confidence / reasoning + 是否走 weighted-downgrade 徽章
- **FR-209**：MUST 渲染风控门 Section：通过的检查 ✓ / 拒绝的检查 ✗ + 拒绝原因
- **FR-210**：MUST 渲染执行 Section：order id / status / fill price / 手续费 / 滑点
- **FR-211**：SHOULD URL 同步选中：选中决策 hash 写入 `?id=`，支持浏览器前进/后退

#### 4. 回测页（Backtest，P1）

- **FR-300**：路由 `/backtest` MUST 提供两个标签：新建回测 / 历史会话
- **FR-301**：新建回测表单 MUST 包含：开始日期 / 结束日期 / 币对 / 初始资金 / 策略模式（rules / llm）/ 可选会话名
- **FR-302**：提交后 MUST 异步调用 `POST /api/backtest/run`，每 5 秒轮询进度，完成后跳转到结果视图；进度卡须支持取消
- **FR-303**：历史会话标签 MUST 提供下拉选会话名 → 加载结果
- **FR-304**：结果视图 MUST 显示 5 项 metric 卡：总收益率 / 夏普比率 / 最大回撤 / 胜率 / 交易次数
- **FR-305**：MUST 渲染权益曲线（复用总览页组件）
- **FR-306**：MUST 渲染决策时间线表（仅 LLM 模式有数据），点击单条复用决策详情组件
- **FR-307**：Pure-rules 模式（decisions=[]）MUST 隐藏 Agent 分析区，仅显示曲线 + 交易表

#### 5. 风控页（Risk，P1）

- **FR-400**：路由 `/risk` MUST 显示风控状态
- **FR-401**：MUST 显示 2 卡：小时交易次数 / 日交易次数
- **FR-402**：断路器卡 MUST 区分状态显示：ACTIVE 红色背景 / INACTIVE 绿色；ACTIVE 时显示剩余 TTL 倒计时
- **FR-403**：MUST 显示阈值卡：max_position_pct / max_daily_loss / max_stop_loss_pct 等关键风控阈值
- **FR-404**：MUST 提供"重置断路器"按钮（带 confirm dialog 二次确认）→ POST → 成功 toast + 自动 refetch
- **FR-405**：Redis 不可达 MUST 显示警告"风控状态不可用 — Redis 未连接"，不报错

#### 6. 指标页（Metrics，P1）

- **FR-500**：路由 `/metrics` MUST 显示 Prometheus 指标快照
- **FR-501**：MUST 显示关键计数器卡：trades_total / orders_placed / orders_failed / risk_rejections
- **FR-502**：MUST 显示延迟分位数卡：pipeline_p50/p95、execution_p50/p95
- **FR-503**：MUST 渲染趋势折线图（4 曲线），数据点累积在 React Query cache + IndexedDB（替换 Streamlit session_state），FIFO 上限 60 个样本
- **FR-504**：SHOULD 数据源：`GET /api/metrics/summary`，刷新间隔 15 秒

#### 7. AI 对话页（ChatAgent，P2）

- **FR-600**：路由 `/chat` SHOULD 采用三栏布局：会话列表 / 消息流 / 内联 widget 渲染区
- **FR-601**：输入框 SHOULD 移植 LangAlpha chat-input 简化版（多行 + Cmd+Enter 提交 + 模型选择下拉）
- **FR-602**：消息流 SHOULD 通过 streamFetch SSE 接收事件：`message_chunk` / `tool_call` / `tool_result` / `inline_widget` / `verdict`
- **FR-603**：Inline Widget SHOULD 在 iframe sandbox 内渲染，注入 24+ CSS 主题变量，patch JSON.parse 处理 NaN/Infinity（移植自 LangAlpha InlineWidget）
- **FR-604**：MAY 会话本地持久化（IndexedDB），暂不做云端同步

#### 8. 市场看板页（MarketView，P2）

- **FR-700**：路由 `/market` SHOULD 输入币对后渲染 TradingView Advanced Chart Widget
- **FR-701**：侧栏 SHOULD 显示：24h 资金费率 / 未平仓量 / 永续-现货价差 / 清算热图
- **FR-702**：SHOULD 支持交易所切换：Binance / OKX

#### 9. 后端 FastAPI 扩展

- **FR-800**：`GET /api/portfolio/snapshot` MUST 返回 `{equity, cash, positions, pnl_24h, drawdown}`
- **FR-801**：`GET /api/portfolio/equity-curve?range=24h|7d|30d|all` MUST 返回时间序列点数组
- **FR-802**：`GET /api/scheduler/status` MUST 返回 `{next_pair, next_run_at}`
- **FR-803**：`GET /api/decisions?pair=&from=&to=&page=&size=` MUST 支持筛选与分页
- **FR-804**：`GET /api/decisions/{commit_hash}` MUST 返回完整 DecisionCommit + 关联 ExperienceMemory
- **FR-805**：`POST /api/backtest/run`（异步）+ `GET /api/backtest/runs/{run_id}` MUST 返回 `{status, progress, result?}`
- **FR-806**：`GET /api/backtest/sessions` 与 `GET /api/backtest/sessions/{name}` MUST 列出与加载历史会话
- **FR-807**：`GET /api/risk/status` 与 `POST /api/risk/circuit-breaker/reset` MUST 提供风控查询与断路器重置
- **FR-808**：`GET /api/metrics/summary` MUST 返回 counters + percentiles
- **FR-809**：`POST /api/chat/stream`（SSE，P2）MUST 接收用户 query → 触发 `build_trading_graph` → 流式返回事件
- **FR-810**：`GET /api/market/{pair}/funding-rate`、`/open-interest`、`/liquidations`（P2）MUST 提供 crypto 市场扩展数据

#### 10. Streamlit 完全弃用（终态约束）

- **FR-900**：MUST 删除 `src/dashboard/` 整个目录（含 `__init__.py` / `app.py` / `components.py` / `data_loader.py` / `_pages/` 全部子文件）
- **FR-901**：MUST 删除所有 streamlit 相关测试：`tests/test_dashboard*.py` / `tests/test_live_decisions_page.py` 及任何 import streamlit 的测试文件
- **FR-902**：MUST 从 `pyproject.toml` 移除 `streamlit>=1.55` 依赖；通过 `uv lock --upgrade` 重新生成 lock 文件，确保 lock 中无 streamlit 及其传递依赖（altair / pydeck / watchdog 等若仅 streamlit 引入）
- **FR-903**：MUST 从 `docker-compose.yml` 移除 `dashboard` service 定义（端口 8501、build context、env_file、volumes、depends_on 全部清理）
- **FR-904**：MUST 从 `src/cli/main.py` 移除 `dashboard` 命令；新增 `arena web` 命令（启动 vite preview 或打印部署说明）
- **FR-905**：MUST 新增 `web` 服务到 `docker-compose.yml`（multi-stage：node 构建 → nginx 托管），完全替代 dashboard service
- **FR-906**：MUST 更新根 `README.md`：移除 Streamlit 启动说明 / 截图 / 端口 8501 引用；新增 web 前端启动章节
- **FR-907**：删除节奏 MUST 为：所有 P1 e2e 通过后单 PR 一次性提交，避免半路回滚
- **FR-908**：MUST 删除任何 streamlit 相关的 Dockerfile 步骤、`.dockerignore` 例外项
- **FR-909**：MUST 删除任何 GitHub Actions / CI 配置中针对 dashboard 的 job 或 step
- **FR-910**：MUST 删除任何文档/Markdown 中的 streamlit 引用：`docs/**/*.md`、`CLAUDE.md`、`.kiro/**`、`brainstorm/**`、`CHANGELOG.md` 中描述当前架构的段落
- **FR-911**：MUST 删除 `.claude/settings.local.json` 等本地配置中针对 streamlit 进程的允许列表条目（如有）
- **FR-912**：MUST 删除任何 `scripts/` 目录中启动 streamlit 的 shell 脚本
- **FR-913**：MUST 删除项目内存（`.kiro/steering/*`）的 streamlit dashboard 架构描述条目
- **FR-914**：PR 合并后 MUST 更新 `MEMORY.md` 与 `architecture-review.md` 等文档以反映新前端架构
- **FR-915**：终态校验 MUST 满足 4 条命令全部 0 命中：
  1. `rg -i streamlit src/ tests/ scripts/` → 0 命中（注释也不允许）
  2. `rg -i streamlit pyproject.toml docker-compose.yml Dockerfile` → 0 命中
  3. `rg -i 'src/dashboard' src/ tests/ docs/` → 0 命中
  4. `rg -i ':8501' .` → 0 命中（除历史 brainstorm 文件外）

#### 文案语言原则

- **DG-1 名词约定 (MUST)**：业务术语用中文（决策/回测/持仓/权益/回撤/断路器/辩论/裁决/风控/调度器/经验记忆）；行业既定英文术语保留（Sharpe / PnL / P50/P95 / OI / Funding Rate / API / SSE / Redis）；币对原格式（BTC/USDT）；状态徽章中文（做多/做空/已成交/待成交/已拒绝/触发中）
- **DG-2 数字与时间格式 (MUST)**：金额千分位 + 币种前缀（`$1,234.56`）；时间 `Intl.DateTimeFormat` 本地化，zh-CN 显示 `2026-04-16 21:30:45`，en-US 显示 `Apr 16, 2026 9:30:45 PM`
- **DG-3 反馈文案 (MUST)**：Toast / 错误信息默认中文（如 `t('common.error.backendUnreachable')`）

#### 非功能需求

性能（NFR-P）：

- **NFR-P-001**：首屏 TTI（3G Fast）≤ 2 秒（Lighthouse 测）
- **NFR-P-002**：主 bundle gzipped ≤ 300 KB
- **NFR-P-003**：路由切换 P50 ≤ 100 ms（已缓存的 lazy chunk）
- **NFR-P-004**：权益曲线渲染 1000 点 ≤ 200 ms
- **NFR-P-005**：决策详情数据加载 P95 ≤ 800 ms
- **NFR-P-007**：指标趋势图（60 样本，4 曲线）渲染 ≤ 150 ms
- **NFR-P-008**：每个页面独立 chunk（React.lazy code splitting）
- **NFR-P-010**：TradingView widget iframe 懒加载（进入视口才挂载）

安全（NFR-S）：

- **NFR-S-001**：所有 API 调用携带 `X-API-Key` header；前端 key 仅来自运行时 `useSettingsStore`（in-memory，从不持久化）。后端默认 fail-closed：`AUTH_MODE=enabled`（默认）+ `API_KEY` env 必须设置，未设则**启动失败**；`AUTH_MODE=disabled` 是显式 opt-out，每请求打 WARNING 日志。Key 比较使用 `secrets.compare_digest` 防时序攻击。CORS preflight 必须返回精确 `allow_methods` / `allow_headers` allowlist，不得使用通配（与 `allow_credentials=true` 冲突）。
- **NFR-S-002**：构建输出无 inline script / `eval` / `new Function`；生产构建 sourcemap 必须为 `hidden` 模式（`.map` 文件可上传到私有 error tracker，但 JS bundle 不得含 `sourceMappingURL` 指针，防止源码泄漏）；`VITE_API_KEY` 在生产构建时必须为空，Vite 插件 `forbid-baked-api-key` 强制此契约。
- **NFR-S-003**：ChatAgent InlineWidget iframe 强制 `sandbox="allow-scripts"`（不给 `allow-same-origin`）
- **NFR-S-004**：Markdown 渲染走 `react-markdown` + `rehype-sanitize`，禁止原始 HTML（所有渲染 LLM 输出的位置都必须挂 `rehypeSanitize`，包括市场分析面板）
- **NFR-S-005**：本地持久化（localStorage / IndexedDB）不存任何 API key / 敏感凭证；`useSettingsStore.apiKey` 仅在 dev 模式从 `VITE_API_KEY` hydrate 一次，生产环境 hydrate 值固定为空
- **NFR-S-006**：生产 nginx 配置 security headers：`X-Content-Type-Options: nosniff`、`X-Frame-Options: DENY`、`Referrer-Policy: no-referrer`、`Content-Security-Policy`（连接源白名单）；后端 rate limiter 必须支持多进程部署（Redis-backed fixed window，60 req/min/IP），单进程 dev 可降级为内存
- **NFR-S-008**：断路器重置操作需 confirm dialog 二次确认（防误触）

可访问性（NFR-A）：

- **NFR-A-001**：所有交互元素可键盘访问（Tab / Enter / Escape）
- **NFR-A-002**：颜色对比度 ≥ WCAG AA（4.5:1）；红绿色盲场景用图标 + 颜色双通道（如 PnL 带 ↑↓ 箭头）
- **NFR-A-003**：所有按钮/图标按钮有 `aria-label`；图表容器有 `aria-describedby` 说明
- **NFR-A-004**：暗/亮主题切换不破坏对比度

可维护性（NFR-M）：

- **NFR-M-001**：TypeScript strict mode 全开（含 `noUncheckedIndexedAccess` / `exactOptionalPropertyTypes`）
- **NFR-M-002**：ESLint 规则覆盖 `@typescript-eslint/recommended-type-checked` + `react-hooks/recommended` + `eslint-plugin-i18next` + `eslint-plugin-import`
- **NFR-M-003**：ESLint 0 错 0 警告（对齐项目 ruff 零警告标准），pre-commit 强制
- **NFR-M-004**：Prettier 格式化，line-length 120
- **NFR-M-005**：目录结构扁平：`web/src/{pages,components,lib,hooks,stores,locales,types}`
- **NFR-M-006**：单个组件文件 ≤ 300 行；超过则拆分
- **NFR-M-007**：`useChatMessages` 简化后必须 ≤ 500 行（LangAlpha 原版 186KB 是反面教材）
- **NFR-M-008**：Zustand store 按领域拆分（useUIStore / useChatStore / useSettingsStore），不做 monolithic store
- **NFR-M-009**：禁止 `any`（必须用 `unknown` + type guard）；确无法避免时需带原因注释
- **NFR-M-010**：公共类型集中 `web/src/types/`，zod schema 文件命名 `*.schema.ts`

可观测性（NFR-O）：

- **NFR-O-001**：前端错误：全局 ErrorBoundary 捕获 → console + 预留上报 hook（本期不实装 Sentry）
- **NFR-O-003**：集成 `web-vitals`（LCP / INP / CLS），dev 模式 console 输出
- **NFR-O-004**：streamFetch 支持 `debug: true` 开关，dev 模式打印所有 SSE chunk
- **NFR-O-005**：Decisions 页 trace_id 渲染为 Jaeger 链接（条件：`OTLP_ENDPOINT` 存在）

兼容性（NFR-C）：

- **NFR-C-001**：浏览器：Chrome/Edge ≥ 120，Firefox ≥ 121，Safari ≥ 17
- **NFR-C-002**：屏幕宽度 ≥ 1280px 完整体验；1024–1280px 可用；< 1024px 显示"桌面专用"提示
- **NFR-C-003**：不支持 IE / 老 Edge / 移动浏览器

文档（NFR-D）：

- **NFR-D-001**：`web/README.md` 含开发启动 / 环境变量 / 构建 / 故障排查
- **NFR-D-002**：根 `README.md` 移除 Streamlit 章节，新增 Web Frontend 章节
- **NFR-D-003**：`docs/frontend-architecture.md` 含 Provider 栈 / 路由表 / 数据合约 / 状态管理图
- **NFR-D-004**：所有 API endpoint 在 FastAPI `openapi.json` 可见，前端调用端有对应 zod schema

### Key Entities

- **Portfolio**：投资组合快照，含权益、现金、持仓数组、24h 盈亏、当前回撤
- **Position**：单个币对持仓，含币对、方向（做多/做空）、数量、均价、未实现盈亏
- **EquityPoint**：权益曲线数据点，含时间戳、权益值
- **DecisionCommit**：单次自动交易决策的完整记录，含 commit_hash、时间戳、币对、价格、4 个 Agent 分析、辩论轮次、最终裁决、风控门检查、订单执行结果、关联的经验记忆引用、可选 OTel trace_id
- **AgentAnalysis**：单个 Agent 的输出，含名称（NewsAgent / MacroAgent / SentimentAgent / TechnicalAgent）、score、confidence、reasoning（Markdown）
- **DebateRound**：单轮 Bull/Bear 辩论，含轮次序号、Bull 消息、Bear 消息
- **Verdict**：最终裁决，含 action（做多/做空/持有）、size（仓位比例）、confidence、reasoning、来源（ai / weighted-downgrade）
- **RiskGate**：风控门检查序列，含每项检查名、是否通过、拒绝原因（若拒绝）
- **Execution**：订单执行结果，含 order id、状态、成交价、手续费、滑点
- **ExperienceMemory**：经验记忆，含 success_patterns / forbidden_zones / strategic_insights 三类规则数组
- **SchedulerStatus**：调度器状态，含下次触发币对、下次触发时间
- **RiskStatus**：风控状态，含小时/日交易计数、断路器状态（active / TTL）、阈值集合
- **MetricsSummary**：指标快照，含计数器集合、延迟分位数集合
- **BacktestParams**：回测参数，含开始日期、结束日期、币对、初始资金、策略模式、可选会话名
- **BacktestRun**：回测任务运行状态，含 run_id、status、progress、可选 result
- **BacktestSession**：回测历史会话，含会话名、参数、结果（metrics + 权益曲线 + 决策列表）
- **BacktestResult**：回测结果，含 5 项 metric、权益曲线、决策时间线
- **ChatMessage**（P2）：AI 对话消息，含角色、内容分块、工具调用记录、内联 widget 引用
- **MarketDataPoint**（P2）：单个币对的扩展市场数据，含资金费率、未平仓量、清算数据等

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**：用户在桌面浏览器访问前端首页，2 秒内看到完整内容（4 个 metric 卡 + 权益曲线 + 持仓表 + 调度器卡）
- **SC-002**：原 Streamlit 5 大业务页面（Overview / Live Decisions / Backtest / Risk Status / Metrics）的全部业务能力在新前端可达，**0 功能回退**
- **SC-003**：用户可在不刷新页面的情况下，连续 30 分钟监控自动交易系统并保持数据准确（10 秒自动刷新机制有效）
- **SC-004**：策略研究员可在 Decisions 页面 5 秒内定位任意历史决策（筛选 + 分页），点击后 1 秒内看到完整管线详情
- **SC-005**：运维者可在 10 秒内完成断路器重置流程（访问 → 确认 → 重置 → 看到生效）
- **SC-006**：回测页支持运行时长 ≥ 30 分钟的长任务，进度卡持续更新且可取消
- **SC-007**：中英文 UI 切换无需刷新即时生效；暗/亮主题切换无需刷新即时生效
- **SC-008**：Streamlit 痕迹完全清零：4 条 ripgrep 校验命令全部 0 命中（详见 FR-915）
- **SC-009**：单条命令 `docker compose up -d` 可拉起全栈（postgres + redis + api + web），首次访问 `localhost:5173` 看到总览页
- **SC-010**：5 个 P1 页面在 Playwright e2e 测试中 100% 通过 happy path
- **SC-011**：FastAPI 已有 endpoint 100% 复用，缺失的 P1 endpoint 在前端集成前补齐并通过 pytest
- **SC-012**：用户操作敏感动作（断路器重置、回测取消）时 100% 触发二次确认对话框

## Assumptions

- **A-1**：用户使用现代桌面浏览器（Chrome / Edge / Firefox / Safari 当前主流版本，屏幕宽度 ≥ 1280px）；不做移动端适配
- **A-2**：本期为单用户/单租户，复用 FastAPI 现有 `API_KEY` header 认证；不做多账号、不做 OAuth
- **A-3**：本期不做 SSR / Next.js / PWA / 离线支持
- **A-4**：本期不引入 Sentry 等错误上报后端，仅预留 hook
- **A-5**：自动交易系统的核心后端能力（LangGraph 多代理管线、回测引擎、风控、APScheduler 调度器、Postgres / Redis 持久化）100% 已就绪，本特性不修改后端业务逻辑，仅扩展 FastAPI endpoint 暴露已有数据
- **A-6**：LangAlpha 项目代码遵循 Apache 2.0 许可，本特性以"selective port"方式借鉴其前端实现（不直接 git clone，不作为 submodule）
- **A-7**：前端开发先行，后端按需补 endpoint；前端任何 React Query hook 起步前先用 `curl` 验通对应 endpoint
- **A-8**：Streamlit 删除作为单 PR 一次性合并，前置条件是所有 P1 e2e 通过；不做半路灰度
- **A-9**：测试遵循项目偏好"少 mock"，端到端测试通过 docker compose 起真实 postgres + redis + api + web
- **A-10**：spex 工作流（superpowers + deep-review + teams + worktrees traits）作为本特性的开发与质量保证流程
- **A-11**：实施分 11 个阶段串行推进，每阶段有明确 DoD，过 DoD 才进下一阶段；阶段内允许 teams trait 派发并行子任务
- **A-12**：TradingView Advanced Chart Widget 免费版可用且满足加密货币图表需求；若 CSP / 网络受限可降级到纯 lightweight-charts
- **A-13**：i18n 翻译文件以 zh-CN 为主写文件（中文先行），en-US 翻译可后续补全；缺失键回退到 zh-CN
- **A-14**：现有 brainstorm 已涵盖关键设计决策（详见 [Section 7.2 决策日志](../../brainstorm/) — 14 条决策项），本 spec 视为共识下的形式化文档

## Dependencies

- **D-1**：FastAPI 后端必须暴露 FR-800 ~ FR-808 列出的全部 P1 endpoint（缺失则 Phase 2 集中补齐）
- **D-2**：Postgres + Redis 通过 docker compose 提供（项目偏好"Docker only"，不用 Homebrew）
- **D-3**：LangAlpha (https://github.com/ginlix-ai/LangAlpha) 公开仓库可访问，用于 selective port 参考
- **D-4**：spex / spec-kit 工具链已就绪（`.specify/` 目录已初始化、`spex-traits.json` 已配置）
- **D-5**：Node.js ≥ 20 + pnpm ≥ 10.18 在开发环境与 CI 中可用

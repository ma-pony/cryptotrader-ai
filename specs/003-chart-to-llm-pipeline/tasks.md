# 实施任务：图表视觉分析 — Chart-to-LLM Pipeline

**Feature Branch**: `003-chart-to-llm-pipeline`
**关联 Plan**: `specs/003-chart-to-llm-pipeline/plan.md`
**关联 Spec**: `specs/003-chart-to-llm-pipeline/spec.md`

---

## Phase 1：基础设施（类型定义 / 配置 / OHLCV 端点）

- [X] T001 [US3] 新增前端类型定义文件 `web/src/types/chart-analysis.ts`，定义 `ChartCapturePayload`、`AdditionalContext`、`VisualAnalysisResult`（status: idle/loading/streaming/done/error）、`CandlestickChartHandle` 接口；确保 `exactOptionalPropertyTypes: true` 下零类型错误（SC-006）

- [X] T002 [US3] 在 `src/cryptotrader/config.py` 新增 `ChartAnalysisConfig` dataclass，字段含 `vision_models: list[str]`、`fast_model: str`、`max_image_bytes: int`、`description_max_tokens: int`；在 `AppConfig` 中添加 `chart_analysis: ChartAnalysisConfig` 字段
  - 文件：`src/cryptotrader/config.py`

- [X] T003 [US3] 在 `config/default.toml` 新增 `[chart_analysis]` 配置节，初始值：`vision_models = ["gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet-20241022", "claude-opus-4-5", "gemini-3.1-pro", "gemini-3-flash"]`，`fast_model = ""`（空表示继承 models.analysis），`max_image_bytes = 4718592`（4.5MB），`description_max_tokens = 800`
  - 注意：`gemini-3.1-pro`、`gemini-3-flash` 为项目配置中使用的模型别名，实际模型名通过 `config.models.*` 和 `models.toml` 映射（项目使用 OpenAI-compatible 端点，别名由运维配置决定，此处填写别名即可）
  - 文件：`config/default.toml`

- [X] T004 [US1] [P] 新增后端 OHLCV 数据端点 `GET /api/market/{pair}/ohlcv`，复用 `data/market.py` 的 CCXT 获取逻辑，查询参数 `timeframe`（默认 `1h`）和 `limit`（默认 100）；返回 `{ bars: [{ time, open, high, low, close, volume }] }`；在现有 `src/api/routes/market.py` 中新增此子路由（`{pair}/ohlcv`），`main.py` 已注册 `market.router`，无需额外注册路由器
  - 文件：`src/api/routes/market.py`（修改）

- [X] T005 [US3] [P] 为 `ChartAnalysisConfig` 新增单元测试：验证默认值正确加载，`max_image_bytes` 类型为 int，`vision_models` 为列表；使用 `load_config()` 从测试 TOML 加载并断言字段
  - 文件：`tests/test_config.py`（修改）

---

## Phase 2：后端扩展（additional_context 解析 + context_builder）

- [X] T006 [US1][US2][US3] 新建 `src/api/context_builder.py`，实现 `build_multimodal_messages(ctx: AdditionalContext, model: str, config: AppConfig) -> list` 函数：
  - 判断 `model in config.chart_analysis.vision_models`；
  - 若支持视觉且 `data_url` 不为 None 且大小 ≤ `max_image_bytes`：构造 LangChain `HumanMessage` 含 `text` + `image_url` content block；
  - 否则：仅使用 `description` 文字，返回纯文本 `HumanMessage`；
  - 多 payload（多时间周期）时，使用 `=== 时间周期: {timeframe} ===` 分隔符（FR-014）；
  - 返回值为可直接传入 `ChatOpenAI.ainvoke()` 的消息列表
  - 文件：`src/api/context_builder.py`（新建）

- [X] T007 [US1][US2] 扩展 `src/api/routes/chat.py` 中的 `ChatStreamRequest` Pydantic 模型，新增 `additional_context: AdditionalContext | None = None`；在 `_run_chat_stream` 中，当 `additional_context` 存在时调用 `context_builder.build_multimodal_messages()` 并将结果注入 LLM 消息链；当降级时推送 `context_notice` SSE 事件
  - 文件：`src/api/routes/chat.py`（修改）

- [X] T008 [US2][US3] 为 `context_builder.py` 编写单元测试：
  - 测试视觉模型时图片被注入 content block；
  - 测试非视觉模型（DeepSeek）时图片被忽略，仅有文字（SC-003）；
  - 测试 `data_url` 超过 `max_image_bytes` 时自动降级；
  - 测试多时间周期时分隔符存在（FR-014）；
  - 测试 `data_url = None` 时不报错（FR-011）
  - 文件：`tests/test_context_builder.py`（新建）

- [X] T009 [US1] 为扩展后的 `chat.py` 端点编写集成测试：mock LLM，发送含 `additional_context` 的 POST，断言 SSE 流正常返回 `message_start` + `content_delta` + `done`；验证无 `additional_context` 的请求（旧接口）仍正常工作（SC-008）
  - 文件：`tests/test_integration.py`（修改）

---

## Phase 3：前端核心（CandlestickChart + 截图 API + indicators）

- [X] T010 [US1] 新建 `web/src/lib/indicators.ts`，实现纯函数：
  - `calcRSI(bars: OHLCVBar[], period = 14): number`
  - `calcMACD(bars: OHLCVBar[], fast = 12, slow = 26, signal = 9): { value: number; signalLine: number; histogram: number }`
  - `calcSMA(bars: OHLCVBar[], period: number): number`
  - `calcVolumeRatio(bars: OHLCVBar[], period = 20): number`（最新成交量 / 均值）
  - `generateDescription(bars, indicators, symbol, timeframe, marketData): string`（FR-003 字段全覆盖）
  - 文件：`web/src/lib/indicators.ts`（新建）

- [X] T011 [US1] 为 `indicators.ts` 编写 Vitest 单元测试：
  - 用已知数据验证 RSI 在 RSI=50 附近（中性）；
  - 验证 `generateDescription` 输出包含 FR-003 要求的全部字段（时间周期、价格、成交量比、趋势方向、RSI 状态、MACD 状态、最近 3 根 K 线、资金费率）（SC-002 前置）
  - 文件：`web/src/lib/indicators.test.ts`（新建）

- [X] T012 [US1] 新建 `web/src/components/charts/candlestick-chart.tsx`，使用 `forwardRef<CandlestickChartHandle, CandlestickChartProps>`：
  - Props：`symbol: string`、`timeframe: string`、`exchange: 'binance' | 'okx'`、`height?: number`
  - 内部 `useQuery` 调用 `/api/market/{pair}/ohlcv` 获取 K 线数据
  - 使用 `createChart` + `addCandlestickSeries`；叠加 `addLineSeries` 渲染 SMA20 / SMA50
  - `subscribeVisibleTimeRangeChange` 设置 `isReady = true`；同时监听首次数据加载完成
  - `useImperativeHandle` 暴露 `captureScreenshot(): string | null`：调用 `chartRef.current.takeScreenshot().toDataURL('image/png')`，若 `isReady === false` 返回 `null`；捕获异常返回 `null`（FR-011）
  - `ResizeObserver` 响应容器尺寸变化；`useEffect` 清理
  - 文件：`web/src/components/charts/candlestick-chart.tsx`（新建）

- [X] T013 [US1] [P] 为 `CandlestickChart` 编写 Vitest + @testing-library/react 测试：
  - mock `lightweight-charts`（stub `createChart`），验证 `captureScreenshot` 在 `isReady=false` 时返回 `null`
  - 验证 `isReady=true` 时返回 dataURL 字符串
  - 验证组件 unmount 时调用 `chart.remove()`
  - 文件：`web/src/components/charts/candlestick-chart.test.tsx`（新建）

---

## Phase 4：Fast 模式（useChartAnalysis + AiAnalysisPanel + MarketView 集成）

- [X] T014 [US1] 新建 `web/src/hooks/use-chart-analysis.ts`（≤ 300 行，NFR-003）：
  - 状态：`result: VisualAnalysisResult`（idle 初始）
  - `triggerFast(chartRef, symbol, timeframe, marketData)`:
    1. 置 `status = 'loading'`
    2. 调用 `chartRef.current?.captureScreenshot()` → 若 `null` 标记 screenshotFailed
    3. 调用 `generateDescription(...)` 生成文字描述
    4. 前端预检图片大小（> 4.5MB → 置 `dataUrl = null`）
    5. 构造 `additionalContext`，调用 `streamFetch('/api/chat/stream', { body: { message: '请分析当前图表形态', additional_context }, signal: abortRef.current.signal, onEvent: handleSseEvent })`
    6. SSE `content_delta` 累积 → `status = 'streaming'`，`context_notice` → 更新提示
    7. SSE `done` → `status = 'done'`；错误 → `status = 'error'`（FR-003、FR-006）
  - `triggerDeep(chartRef, symbol, timeframe, marketData)`:
    1. 步骤 1-4 同 triggerFast
    2. `navigate('/chat', { state: { additionalContext } })`（FR-007）
  - `stop()`: 调用 `abortRef.current.abort()`（FR-012）
  - `resetResult()`: 重置为 idle（用于切换交易对/时间周期时）
  - 导出 `UseChartAnalysisReturn` 类型
  - 文件：`web/src/hooks/use-chart-analysis.ts`（新建）

- [X] T015 [US1] 为 `useChartAnalysis` 编写 Vitest 测试（mock `streamFetch`）：
  - 测试截图失败（`null`）时仍能发起请求，`screenshotFailed = true`
  - 测试 `stop()` 调用 abort；
  - 测试 `resetResult()` 清空结果
  - 文件：`web/src/hooks/use-chart-analysis.test.ts`（新建）

- [X] T016 [US1] 新建 `web/src/pages/market/components/ai-analysis-panel.tsx`：
  - Props：`result: VisualAnalysisResult`、`onStop: () => void`、`onRetry: () => void`
  - 渲染逻辑：
    - `idle`：不渲染（`null`）
    - `loading`：骨架屏（3 行 Skeleton）
    - `streaming`：`<ReactMarkdown>` 实时渲染 `content_md`（`aria-live="polite"`）
    - `done`：完整 Markdown + 来源标注 badge（FR-009）
    - `error`：中文错误提示（NFR-004）
  - 当 `screenshotFailed` 标记存在时显示"图片截取失败，使用文字描述分析"提示（FR-011）
  - 当 `contextNotice = 'image_too_large'` 时显示"图片过大，已使用文字摘要"提示（US2 验收场景 3）
  - `role="region"` 和 `aria-live="polite"`（NFR-005）
  - 文件：`web/src/pages/market/components/ai-analysis-panel.tsx`（新建）

- [X] T017 [US1] 新建 `web/src/pages/market/components/chart-tab-panel.tsx`：
  - 使用 `@radix-ui/react-tabs`，Tab 项："TradingView"（默认）| "K 线图（可分析）"
  - 切换到 "K 线图" Tab 时 lazy mount `CandlestickChart`，传递 `chartRef`
  - 当 active tab 为 TradingView 时禁用 AI 分析按钮并显示 tooltip 说明（spec 边界条件）
  - 传递：`symbol`、`exchange`、`timeframe`、`chartRef`（Ref 提升）
  - 文件：`web/src/pages/market/components/chart-tab-panel.tsx`（新建）

- [X] T018 [US1] 修改 `web/src/pages/market/index.tsx`：
  - 引入 `ChartTabPanel` 替换原 `<TradingViewChart>` 所在的 Card
  - 新增状态：`timeframe: string`（默认 `'1h'`）、`chartRef: RefObject<CandlestickChartHandle>`
  - 引入 `useChartAnalysis`，渲染"AI 分析此图（快速）"和"AI 分析此图（深度）"按钮
  - 按钮样式：loading 状态 `aria-busy="true"`、`aria-label`（NFR-005）
  - Fast 按钮旁边提供"停止"按钮（`status === 'streaming'` 时显示，FR-012）
  - `pair` 或 `timeframe` 变化时调用 `resetResult()`（US1 验收场景 2）
  - 在图表下方渲染 `AiAnalysisPanel`（FR-006）
  - 文件：`web/src/pages/market/index.tsx`（修改）

---

## Phase 5：Deep 模式（Chat 页面扩展 + sendMessage 签名扩展）

- [X] T019 [US2] 扩展 `web/src/hooks/use-chat-messages.ts` 的 `sendMessage` 签名：
  - `sendMessage(text: string, additionalContext?: AdditionalContext): void`
  - 当 `additionalContext` 存在时，`streamFetch` body 增加 `additional_context` 字段
  - 所有现有调用点（Chat 页面）不传第二参数，向后兼容（SC-008）
  - 文件：`web/src/hooks/use-chat-messages.ts`（修改）

- [X] T020 [US2] 修改 `web/src/pages/chat/index.tsx`，新增图表上下文注入逻辑：
  - `useLocation` 读取 `state.additionalContext`
  - `useEffect`（仅首次执行，空 deps + `initialContextRef` 防重入）：若存在 `additionalContext`，100ms 后调用 `sendMessage('请分析当前图表', additionalContext)`（SC-004）
  - 渲染"已附加图表上下文"badge：含 symbol、timeframe、截图时间（FR-008）；badge 固定在消息流顶部
  - 文件：`web/src/pages/chat/index.tsx`（修改）

- [X] T021 [US2] 为 Chat 页面上下文注入逻辑编写测试：
  - mock `useLocation` 注入 `additionalContext`，验证 500ms 内 `sendMessage` 被调用（SC-004）
  - 验证 badge 渲染包含正确的 symbol 和 timeframe（FR-008）
  - 验证无 `additionalContext` 时 badge 不渲染（向后兼容）
  - 文件：`web/src/pages/chat/chat-page.test.tsx`（新建）

---

## Phase 6：P2 多时间周期叠加分析

- [X] T022 [US4] [P] 在 `web/src/pages/market/index.tsx` 新增多时间周期选择 UI：下拉多选（最多 2 个时间周期），默认仅 `1h`；选择多个时间周期时，`triggerFast` / `triggerDeep` 依次为每个时间周期生成 `ChartCapturePayload` 并组装到 `payloads` 数组（FR-010）
  - 文件：`web/src/pages/market/index.tsx`（修改）

- [X] T023 [US4] [P] 验证 `context_builder.py` 对多 payload 的处理（已在 T008 覆盖分隔符测试），补充测试：断言当 `payloads` 长度为 2 时，输出消息含两个独立时间周期区段，标注明确（FR-014）
  - 文件：`tests/test_context_builder.py`（修改）

- [ ] T024 [US4] 为多时间周期模式编写端到端测试（Playwright）：（需要本地运行环境，deferred）
  - 在 MarketView 选择"15m + 4h"，点击"AI 分析此图（快速）"，断言结果卡片内存在两个时间周期标注（SC-005）
  - 文件：`web/tests/market-ai-analysis.spec.ts`（新建）

---

## Phase 7：i18n / 无障碍 / TypeScript Strict / 收尾

- [X] T025 [US1] 更新 `web/src/locales/zh-CN/market.json`，新增 `ai_analysis` 命名空间下所有 UI 键值（NFR-004）：
  ```json
  "ai_analysis": {
    "fast_btn": "AI 分析此图（快速）",
    "deep_btn": "AI 分析此图（深度）",
    "stop_btn": "停止",
    "tab_tradingview": "TradingView",
    "tab_candlestick": "K 线图（可分析）",
    "tab_tradingview_disabled_tip": "TradingView 图表不支持 AI 截图分析，请切换到「K 线图」",
    "status_loading": "AI 正在分析图表形态...",
    "status_error": "分析失败，请重试",
    "screenshot_failed": "图片截取失败，使用文字描述分析",
    "image_too_large": "图片过大，已使用文字摘要",
    "context_badge": "已附加图表上下文",
    "result_expired": "分析结果已过期（图表已更新）",
    "source_visual": "视觉识别",
    "source_numerical": "数值数据"
  }
  ```
  - 文件：`web/src/locales/zh-CN/market.json`（修改）

- [X] T026 [US1] [P] 更新 `web/src/locales/en-US/market.json`，新增对应英文键值（与中文键名一致）
  - 文件：`web/src/locales/en-US/market.json`（修改）

- [X] T027 全项目 TypeScript strict 检查：运行 `pnpm typecheck`，确保所有新增文件零类型错误（`exactOptionalPropertyTypes: true`）（SC-006）
  - 重点检查：`CandlestickChartHandle` 的 `forwardRef` 类型、`VisualAnalysisResult` 的 union 状态类型、`AdditionalContext` 可选字段

- [X] T028 后端 lint 检查：运行 `ruff check src/api/context_builder.py src/api/routes/chat.py src/cryptotrader/config.py`，确保零 lint 错误，零 `noqa` 注释（SC-007）

- [X] T029 安全审查：确认 `dataUrl` DataURL 字符串不写入 localStorage / IndexedDB / sessionStorage / 后端数据库（NFR-002）；在 `use-chart-analysis.ts` 的 SSE 结束回调中显式置 `dataUrl` 引用为 `null`

- [ ] T030 性能验证（NFR-001）（需要浏览器 DevTools，deferred）：在浏览器 DevTools Performance 面板录制截图操作，确认 `takeScreenshot().toDataURL()` 耗时 ≤ 200ms（1080p 分辨率下）；若超时，降低图表分辨率 scale 参数（lightweight-charts 支持 `devicePixelRatio` 覆盖）

- [X] T031 [P] 补全 `context_builder.py` 的 docstring，说明视觉降级逻辑和多时间周期分隔符格式；补全 `use-chart-analysis.ts` 的 JSDoc，说明 hook 行数预算（NFR-003）

- [ ] T032 [P] 验证 Fast 模式端到端延迟（SC-001）（需要本地运行环境，deferred）：使用本地 OpenAI 代理或 mock SSE，记录从按钮点击到首 token 出现的时间，目标 p95 ≤ 8 秒；若超标，检查图片大小是否为瓶颈，设置更激进的前端预检阈值

---

## 任务依赖关系

```
T001 → T014, T016, T017, T018, T019
T002 → T003 → T006
T003 → T007
T004 → T012
T006 → T007 → T008 → T009
T010 → T011, T014
T012 → T013, T014, T017
T014 → T015, T018
T016 → T018
T017 → T018
T018 → T019 → T020 → T021
T022 → T023 → T024
T025, T026 → T027（i18n 键值先行，类型检查后做）
T007, T014, T018, T019, T020 → T028, T029, T030, T031, T032
```

## 完成标准

- [ ] SC-001: Fast 模式首 token ≤ 8 秒（p95）
- [ ] SC-002: 截图降级 100% 无异常
- [ ] SC-003: DeepSeek 配置下 100% 无图片发送
- [ ] SC-004: Deep 模式 500ms 内自动发送首条消息
- [ ] SC-005: 多时间周期结果均有时间周期标注（P2）
- [ ] SC-006: 所有新增前端代码 TypeScript strict 零错误
- [ ] SC-007: 所有新增后端代码 ruff 零错误，零 noqa
- [ ] SC-008: 现有 Chat 功能回归测试 100% 通过

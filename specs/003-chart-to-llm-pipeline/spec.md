# 功能规格说明：图表视觉分析 — Chart-to-LLM Pipeline

**Feature Branch**: `003-chart-to-llm-pipeline`
**Created**: 2026-04-17
**Status**: Draft

---

## 背景与动机

CryptoTrader-AI 是基于 LangGraph 的多智能体加密货币交易系统。前端使用 React 19 + TradingView Widget 渲染 K 线图（MarketView 页面），equity-chart / trend-chart 组件使用 lightweight-charts 4 绘制权益曲线与趋势线。

当前所有 Agent 的输入均为结构化数值数据（OHLCV、资金费率、链上数据等），无法感知视觉图形中的形态信息（如双顶、头肩、支撑/阻力突破等）。本功能通过"截图 → 结构化描述 → 多模态 LLM 分析"的 pipeline，使 AI Agent 具备"看图"能力，填补数值数据与视觉形态之间的分析盲区。

---

## 用户故事与验收场景 *(必填)*

### 用户故事 1 — 一键 AI 分析当前图表 (优先级: P0)

**故事**：作为交易者，我在 MarketView 页面查看 BTC/USDT 的 K 线图时，希望点击一个按钮就能让 AI 快速分析当前图表形态，并在页面内直接显示结论，无需跳转。

**为何是 P0**：这是整个 pipeline 的核心用户价值入口。没有此功能，截图和分析能力都无从展示。

**独立测试条件**：不依赖 Deep 模式和 Chat 页面，可独立验收。

**验收场景**:

1. **Given** 用户已打开 MarketView 页面并看到 K 线图，**When** 用户点击"AI 分析此图（快速）"按钮，**Then** 按钮变为 loading 状态，图表区域下方出现分析结果卡片，5 秒内开始流式显示 AI 文字（中文），包含识别到的图表形态（如支撑/阻力、趋势方向、K 线结构），loading 结束后按钮恢复可点击状态。

2. **Given** 分析结果卡片已显示，**When** 用户切换交易对或时间周期，**Then** 旧分析结果卡片自动收起（或显示"已过期"标记），不阻塞新的图表渲染。

3. **Given** 网络超时或 LLM 返回错误，**When** Fast 分析请求失败，**Then** 分析卡片显示错误提示（中文），按钮恢复可点击，不出现白屏或 JS 报错。

---

### 用户故事 2 — Deep 模式：携带图表上下文进入完整对话 (优先级: P1)

**故事**：作为交易者，我希望在 MarketView 页面选择"AI 分析此图（深度）"后，跳转到 Chat 页面，AI 已预加载了图表截图和结构化市场上下文，可以进行多轮深度对话（如"这个形态的历史胜率是多少？"）。

**为何是 P1**：依赖 P0 的截图/描述生成能力，属于增强体验，不阻塞核心 pipeline 的验收。

**独立测试条件**：可通过 React Router state 手动注入 additionalContext 独立测试 Chat 页面。

**验收场景**:

1. **Given** 用户在 MarketView 点击"AI 分析此图（深度）"，**When** 路由跳转到 `/chat`，**Then** Chat 页面的输入框预填充了包含图表描述的上下文文字，消息流中出现一条"图表已附加"的可视化提示，AI 自动开始首次分析流式响应。

2. **Given** Chat 页面携带了 additionalContext，**When** 用户追问"换个角度分析空头风险"，**Then** AI 的后续回复仍能引用图表中的具体数据点（如阻力位价格），体现上下文连续性。

3. **Given** 图表截图数据超过 LLM 输入上限，**When** 系统降级处理，**Then** 自动回落到仅发送结构化文字描述（无图片），Chat 页面显示"图片过大，已使用文字摘要"提示，分析仍能正常完成。

---

### 用户故事 3 — 结构化文字描述作为独立上下文注入 (优先级: P1)

**故事**：作为系统管理员，我希望图表分析 pipeline 除了支持多模态（图片）输入，还能生成纯文字的结构化图表描述，这样在不支持视觉输入的 LLM（如 DeepSeek）上也能工作。

**为何是 P1**：保证 pipeline 在不同 LLM 后端（Claude、GPT-4o、Gemini、DeepSeek）上都可降级运行，是鲁棒性要求。

**独立测试条件**：可单元测试描述生成函数，输入 OHLCV + 指标数据，断言输出字符串格式合规。

**验收场景**:

1. **Given** 后端 LLM 为 DeepSeek（不支持视觉），**When** 图表分析请求到达，**Then** 系统自动检测不支持视觉能力，仅发送文字描述，LLM 返回基于文字的形态分析，不报错。

2. **Given** 用户有 BTC/USDT 1小时图的 OHLCV 数据和 RSI/MACD 指标，**When** 生成结构化文字描述，**Then** 描述包含：时间周期、最新价格、成交量相对均值的比率、趋势方向（基于 MA 关系）、指标状态（RSI 超买/超卖/中性）、最近 3 根 K 线特征（实体/影线比例）。

---

### 用户故事 4 — 多时间周期叠加分析 (优先级: P2)

**故事**：作为高级交易者，我希望 AI 能同时分析 15 分钟图和 4 小时图，识别"大周期趋势 + 小周期入场点"的配合关系。

**为何是 P2**：依赖 P0/P1 核心 pipeline 稳定后扩展，属于高级功能，用户门槛较高。

**独立测试条件**：可通过 Mock 多个时间周期的 OHLCV 数据独立测试叠加分析逻辑。

**验收场景**:

1. **Given** 用户在 MarketView 选择了"15m + 4h"叠加分析模式，**When** 点击"AI 分析"，**Then** 分析结果包含两段内容：大周期趋势判断（4h）和小周期入场建议（15m），并在结果中明确标注每条结论对应的时间周期。

2. **Given** 两个时间周期的信号相互矛盾（4h 看涨，15m 看跌），**When** AI 输出分析，**Then** 结果中包含矛盾点说明，不自动选边，提示用户需要等待信号收敛。

---

### 边界条件

- TradingView Widget 不暴露 `takeScreenshot()` API；截图功能仅适用于 lightweight-charts 组件（equity-chart、trend-chart 及新增的 K 线图组件）
- LLM 图片输入大小限制：Claude 最大 5MB / base64，GPT-4o 最大 20MB，Gemini 最大 10MB；超限时降级为文字描述
- `captureChartAsDataUrl()` 是 lightweight-charts 4.x 提供的 Canvas 截图 API，仅在图表完全渲染后可调用；渲染未完成时调用应返回错误而非空图
- 单用户自托管系统，无需处理多用户并发截图请求的竞态
- Fast 模式目标响应时间 ≤ 8 秒（首 token 到达），Deep 模式无硬性时限但需显示进度
- 分析结果不持久化到数据库（属于临时辅助信息，非交易决策的正式记录）

---

## 功能需求 *(必填)*

### 功能需求列表

**FR-001**: 系统须在 MarketView 页面的 K 线图区域旁提供"AI 分析此图"操作入口，提供 Fast（快速）和 Deep（深度）两种分析模式的选择。

**FR-002**: 当用户触发 Fast 分析时，系统须截取当前 lightweight-charts 图表的可视区域为 PNG DataURL（base64 编码），无需用户手动操作。

**FR-003**: 系统须为每次分析请求生成结构化文字描述，内容包括：交易对名称、时间周期、截图时间戳、最新收盘价、成交量（相对 20 周期均值的比率）、趋势方向（基于短/长期均线关系）、RSI 状态（超买 / 超卖 / 中性）、MACD 状态（金叉 / 死叉 / 散开 / 收敛）、最近 3 根 K 线的实体/影线特征，以及当前资金费率和未平仓量。

**FR-004**: `/api/chat/stream` 端点须扩展支持 `additional_context` 字段，字段值为包含图表 DataURL（可选）和结构化文字描述的对象，端点须将图片和文字描述一并注入 LLM 消息链。

**FR-005**: 系统须根据当前配置的 LLM 模型自动检测是否支持视觉输入（Claude、GPT-4o、Gemini 支持；DeepSeek、其他文字模型不支持），不支持时自动降级为仅使用文字描述。

**FR-006**: Fast 分析结果须以流式方式（SSE）在 MarketView 页面内的分析结果卡片中实时渲染，不跳转页面。

**FR-007**: Deep 分析须通过 React Router state 将 `additionalContext`（含文字描述和可选图片 DataURL）传递至 `/chat` 页面，Chat 页面须在初始化时检测 state 并自动发送首条分析请求。

**FR-008**: Chat 页面须在消息流顶部显示"已附加图表上下文"的视觉标记（含交易对、时间周期、截图时间）。

**FR-009**: 系统须在分析结果中区分并标注结论的来源：来自图表视觉识别还是来自数值数据。

**FR-010**: MarketView 页面须支持用户手动选择参与叠加分析的时间周期（最多 2 个），并同时生成对应的图表描述和截图提交分析。

**FR-011**: 当 lightweight-charts 图表尚未完成渲染或截图失败时，系统须仅使用文字描述发起分析请求，并在 UI 中显示"图片截取失败，使用文字描述分析"的提示。

**FR-012**: Fast 模式须有独立的"停止"按钮，用户可在流式输出过程中中止请求（复用 `streamFetch` 的 AbortController 机制）。

**FR-013**: 系统须在 `config/default.toml` 中提供 `[chart_analysis]` 配置节，支持配置：`vision_models`（支持视觉的模型列表）、`fast_model`（Fast 模式使用的模型名）、`max_image_bytes`（图片上限，超出则降级）、`description_max_tokens`（文字描述的 token 上限）。

**FR-014**: 多时间周期叠加分析时，LLM Prompt 须明确区分各时间周期的数据区段，避免数据混淆。

---

### 关键实体

**ChartCapturePayload（图表截图载荷）**: 前端生成，包含 `dataUrl`（PNG base64，可为 null）、`symbol`（交易对）、`timeframe`（时间周期，如 "1h"）、`capturedAt`（ISO 时间戳）、`description`（结构化文字描述字符串）。

**AdditionalContext（附加上下文）**: 注入 LLM 的额外输入，包含一组 `ChartCapturePayload`（支持多时间周期），由后端负责组装为 LangChain 消息格式。

**ChatStreamRequest（扩展后）**: 在现有 `session_id`、`message`、`model` 字段基础上增加 `additional_context` 字段（`AdditionalContext | null`）。

**ChartAnalysisConfig（图表分析配置）**: `config.py` 中新增的配置数据类，对应 `[chart_analysis]` TOML 配置节。

**VisualAnalysisResult（视觉分析结果）**: 前端状态，包含 `status`（idle / loading / streaming / done / error）、`content_md`（流式 Markdown 文本）、`source_context`（触发分析时的 ChartCapturePayload，用于"已过期"判断）。

---

## 成功标准 *(必填)*

### 可量化的验收结果

**SC-001**: Fast 分析模式下，从用户点击按钮到首 token 出现在 MarketView 页面内的时间 ≤ 8 秒（p95，本地网络）。

**SC-002**: 截图降级机制有效：当图表未渲染完成时，100% 的请求自动切换为纯文字描述模式，不出现请求中断或 JS 异常。

**SC-003**: 视觉 LLM 检测有效：配置 DeepSeek 为 fast_model 时，100% 的请求不携带图片，LLM 调用成功率不低于文字模式基准。

**SC-004**: Deep 模式路由跳转后，Chat 页面 100% 成功接收 additionalContext 并在 500ms 内自动发起首条消息。

**SC-005**: 多时间周期分析（P2）：2 个时间周期的分析结果在单次响应中均有体现，各自有明确的时间周期标注。

**SC-006**: 所有新增前端组件通过 TypeScript strict 编译（`exactOptionalPropertyTypes: true`），0 类型错误。

**SC-007**: 所有新增后端代码通过 ruff lint，0 lint 错误，0 `noqa` 注释。

**SC-008**: 新增 SSE 协议字段 `additional_context` 与现有 `streamFetch` + `useChatMessages` 向后兼容，现有 Chat 功能不受影响（回归测试 100% 通过）。

---

## 假设

- lightweight-charts 4.x 的 `IChartApi.takeScreenshot()` 方法在图表完全渲染后可返回 HTMLCanvasElement，前端通过 `.toDataURL('image/png')` 获得 base64 字符串。
- TradingView Widget（当前 MarketView 页面使用）不支持程序化截图；MarketView 页面将新增一个基于 lightweight-charts 的自绘 K 线图组件，与 TradingView Widget 并存（用户可切换），Fast/Deep 分析仅对 lightweight-charts 组件有效。
- 后端 LangChain 已支持 OpenAI Vision / Anthropic Vision API 格式；多模态消息构造无需引入新的 LangChain 依赖。
- 分析结果不写入 journal 数据库，不影响 Agent 的正式决策流程；pipeline 为纯辅助分析工具。
- 单用户部署，前端无需处理截图的并发锁或防重复提交（单次点击即发送）。
- `config.chart_analysis.vision_models` 列表由运维配置维护，系统不在运行时自动探测 LLM 的视觉能力。
- 结构化文字描述由前端生成（基于已有的 OHLCV 数据和指标计算结果），不依赖后端额外 API 调用。

---

## 非功能需求

**NFR-001（性能）**: 图表截图操作（Canvas → DataURL）须在主线程完成，但不得阻塞图表的后续渲染或 React 更新；截图时间 ≤ 200ms（1080p 分辨率下）。

**NFR-002（安全）**: 图片 DataURL 仅在单次 HTTP 请求生命周期内存在于内存，不写入 localStorage、IndexedDB 或后端数据库，请求完成后前端丢弃引用。

**NFR-003（可维护性）**: Fast 模式的分析逻辑封装为独立的 React hook（`useChartAnalysis`），文件行数 ≤ 300 行；后端 `additional_context` 处理逻辑封装为独立函数，不与现有 chat stream 逻辑混合。

**NFR-004（国际化）**: UI 文字（按钮标签、状态提示、错误信息）须通过 i18next `market` 命名空间提供中文键值，与现有 i18n 规范一致。

**NFR-005（可访问性）**: "AI 分析此图"按钮须有 `aria-label` 属性，loading 状态须有 `aria-busy="true"`，分析结果卡片须有 `role="region"` 和 `aria-live="polite"`。

---

## 范围外（Out of Scope）

- 将图表视觉分析结果自动纳入正式交易决策（Agent 分析流程不变，本 feature 为独立辅助工具）
- 分析历史记录的持久化与回放
- 移动端 / 小屏幕适配（系统已有 `desktop-only-banner` 组件，本功能同样仅桌面端支持）
- 自动定时截图与分析（无触发式后台任务）
- TradingView Widget 截图（技术限制，不在本 spec 范围内）

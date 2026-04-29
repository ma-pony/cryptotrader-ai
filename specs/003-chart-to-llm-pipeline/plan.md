# 技术实施方案：图表视觉分析 — Chart-to-LLM Pipeline

**Feature Branch**: `003-chart-to-llm-pipeline`
**规划日期**: 2026-04-17
**关联 Spec**: `specs/003-chart-to-llm-pipeline/spec.md`

---

## 一、技术上下文

### 1.1 现有图表组件分析

| 组件 | 文件路径 | 截图能力 | 备注 |
|------|----------|----------|------|
| `EquityChart` | `web/src/components/charts/equity-chart.tsx` | 可行 | 基于 lightweight-charts，`chartRef.current.takeScreenshot()` 可调用 |
| `TrendChart` | `web/src/components/charts/trend-chart.tsx` | 可行 | 同上 |
| `TradingViewChart` | `web/src/pages/market/components/tradingview-chart.tsx` | **不可行** | TradingView Widget CDN 方式加载，无截图 API |

**关键发现**：MarketView 页面目前只有 `TradingViewChart` 组件（iframe 嵌入方式），不支持程序化截图。根据 spec 假设，需要新增一个基于 lightweight-charts 的自绘 K 线图组件（`CandlestickChart`），与 TradingView Widget 并存，用户可通过 Tab 切换。Fast/Deep 分析仅对 `CandlestickChart` 有效。

### 1.2 lightweight-charts 截图 API

lightweight-charts 4.x 提供 `IChartApi.takeScreenshot()` 方法，返回 `HTMLCanvasElement`，前端通过 `.toDataURL('image/png')` 获取 base64 字符串。截图时机须在图表首次渲染完成后（通过 `useEffect` + `subscribeVisibleTimeRangeChange` 事件或延时标志位判断）。截图时间目标 ≤ 200ms（NFR-001）。

### 1.3 现有 Chat API 分析

`POST /api/chat/stream`（`src/api/routes/chat.py`）目前接收 `ChatStreamRequest`（`session_id`、`message`、`model`），响应为 SSE 流。需新增 `additional_context` 字段注入图表载荷。现有 `streamFetch`（`web/src/lib/stream-fetch.ts`）基于 AbortController，完全兼容扩展，无需改动。`useChatMessages` hook（`web/src/hooks/use-chat-messages.ts`，220 行）目前 `sendMessage(text: string)` 签名为纯文本，需扩展为可携带附加上下文。

### 1.4 现有配置体系

`src/cryptotrader/config.py` 使用 `@dataclass` + TOML 解析体系，`AppConfig` 聚合各子 config。需新增 `ChartAnalysisConfig` dataclass 并添加至 `AppConfig`，对应 `config/default.toml` 中的 `[chart_analysis]` 配置节。

### 1.5 多模态 LLM 支持现状

`create_llm()` 工厂函数（`src/cryptotrader/agents/base.py`）使用 `ChatOpenAI` 统一接入。LangChain `ChatOpenAI` 对 OpenAI Vision（GPT-4o）、Anthropic Claude、Google Gemini 均支持多模态消息格式（`HumanMessage` 中嵌入 `image_url` 类型 content block）。DeepSeek 不支持视觉。是否支持视觉由 `config.chart_analysis.vision_models` 列表静态配置，系统不做运行时探测。

---

## 二、架构决策

### 决策 1：新增 `CandlestickChart` 组件复用现有 lightweight-charts 模式
- **选择**：新建 `web/src/components/charts/candlestick-chart.tsx`，使用与 `EquityChart` / `TrendChart` 相同的 `createChart` + `useRef<IChartApi>` 模式，通过 `useImperativeHandle` 暴露 `captureScreenshot(): string | null` 方法。
- **理由**：与现有组件风格一致；`IChartApi.takeScreenshot()` 只对 lightweight-charts canvas 有效，不依赖 DOM 截图库，零新增依赖。

### 决策 2：图表切换采用 Tab 控件，TradingView 与 Candlestick 并存
- **选择**：在 MarketView 的图表 Card 顶部添加 `@radix-ui/react-tabs` Tab（"TradingView | K 线图"），`CandlestickChart` 在 Tab 切换为 K 线图时才渲染（lazy mount），避免两个图表同时订阅数据。
- **理由**：已有 `@radix-ui/react-tabs` 依赖（`web/package.json`），无需新增包。TradingView 依然是默认主视图，不影响现有用户体验。

### 决策 3：结构化文字描述由前端生成
- **选择**：前端利用已有的 K 线 OHLCV 数据和指标（RSI/MACD 在前端计算）生成结构化文字描述，不增加额外后端 API 调用。
- **理由**：spec 假设中明确描述由前端生成；避免后端往返增加延迟；前端持有全部必要数据。

### 决策 4：`useChartAnalysis` hook 封装 Fast 模式逻辑（≤ 300 行）
- **选择**：新建 `web/src/hooks/use-chart-analysis.ts`，封装截图、描述生成、SSE 调用、状态管理（`VisualAnalysisResult`）等全部逻辑。
- **理由**：NFR-003 明确要求；与 `useChatMessages` 职责分离，互不干扰。

### 决策 5：后端 `additional_context` 处理封装为独立函数
- **选择**：新建 `src/api/context_builder.py`，提供 `build_multimodal_messages()` 函数，接受 `AdditionalContext` 并返回 LangChain 消息列表。`chat.py` 路由仅调用该函数，不混入内联逻辑。
- **理由**：NFR-003；便于单元测试视觉检测降级逻辑。

### 决策 6：视觉能力检测基于静态配置列表
- **选择**：`config.chart_analysis.vision_models: list[str]`，由后端在 `build_multimodal_messages()` 中判断当前模型是否在列表内，不在则忽略图片字段。
- **理由**：spec 假设明确禁止运行时自动探测；静态配置可维护性更高。

### 决策 7：Deep 模式通过 React Router state 传递上下文
- **选择**：复用 React Router v7 的 `useNavigate` + `state` 参数，在 Chat 页面 `useLocation().state` 读取 `additionalContext`，通过 `useEffect` 自动触发首次 `sendMessage`。
- **理由**：spec FR-007 明确指定此方案；无需 Zustand 额外 store 或 sessionStorage，路由跳转后自动清理。

### 决策 8：指标（RSI / MACD）在前端计算
- **选择**：在 `web/src/lib/indicators.ts` 实现轻量 RSI（14 期）、MACD（12/26/9）纯函数，输入为 OHLCV 数组，输出为指标值。
- **理由**：不引入大型 TA 库（如 `technicalindicators`，~300KB）；RSI/MACD 算法简单，20-50 行即可实现；零新增依赖。

---

## 三、文件结构（新增 / 修改）

### 新增文件

```
web/src/
├── components/charts/
│   └── candlestick-chart.tsx          # lightweight-charts K 线图组件（含截图 API）
├── hooks/
│   └── use-chart-analysis.ts          # Fast 模式核心 hook（≤ 300 行）
├── lib/
│   └── indicators.ts                  # RSI / MACD 纯函数计算
├── pages/market/components/
│   ├── ai-analysis-panel.tsx          # 分析结果卡片（VisualAnalysisResult 渲染）
│   └── chart-tab-panel.tsx            # TradingView / CandlestickChart Tab 容器
├── types/
│   └── chart-analysis.ts              # ChartCapturePayload / AdditionalContext / VisualAnalysisResult 类型
└── locales/
    ├── zh-CN/market.json              # 新增 ai_analysis 命名空间键值（修改）
    └── en-US/market.json              # 同上（修改）

src/
├── api/
│   └── context_builder.py             # build_multimodal_messages() 独立函数
└── cryptotrader/
    └── config.py                      # 新增 ChartAnalysisConfig dataclass + AppConfig 字段（修改）

config/
└── default.toml                       # 新增 [chart_analysis] 配置节（修改）
```

### 修改文件

```
web/src/
├── pages/market/index.tsx             # 引入 chart-tab-panel + ai-analysis-panel
├── pages/chat/index.tsx               # 检测 router state，自动发送首条消息
├── hooks/use-chat-messages.ts         # sendMessage 签名扩展 additionalContext 参数
└── types/api.ts                       # 引入 chart-analysis 类型（可选 re-export）

src/api/
└── routes/chat.py                     # ChatStreamRequest 增加 additional_context 字段，调用 context_builder
```

---

## 四、API 设计

### 4.1 `/api/chat/stream` 扩展

**新增请求字段**（Pydantic 模型扩展）：

```python
class ChartCapturePayload(BaseModel):
    data_url: str | None = None          # PNG base64，None 表示截图失败
    symbol: str                          # 交易对，如 "BTC/USDT"
    timeframe: str                       # 时间周期，如 "1h"
    captured_at: str                     # ISO 时间戳
    description: str                     # 结构化文字描述

class AdditionalContext(BaseModel):
    payloads: list[ChartCapturePayload]  # 支持多时间周期（最多 2 个）

class ChatStreamRequest(BaseModel):
    session_id: str = ""
    message: str = ""
    model: str = ""
    additional_context: AdditionalContext | None = None  # 新增
```

**后端处理流程**：

```
ChatStreamRequest
  └─ additional_context 存在？
       ├─ 是 → context_builder.build_multimodal_messages(ctx, fast_model)
       │         ├─ fast_model in vision_models？
       │         │    ├─ 是 → HumanMessage([text_block, image_url_block, ...])
       │         │    └─ 否 → HumanMessage([text_block_only, ...])  ← 降级
       │         └─ data_url size > max_image_bytes？→ 降级到文字
       └─ 否 → 现有 LLM 调用路径（向后兼容）
```

### 4.2 SSE 新增事件（可选）

为前端显示降级提示，后端在降级时可推送：

```
event: context_notice
data: {"notice": "image_too_large" | "vision_not_supported", "message": "..."}
```

---

## 五、前端设计

### 5.1 新增组件

#### `CandlestickChart`（`web/src/components/charts/candlestick-chart.tsx`）

- 使用 `forwardRef` + `useImperativeHandle` 暴露 `captureScreenshot(): string | null`
- 通过 `subscribeVisibleTimeRangeChange` 设置 `isReady` 标志位，未就绪时 `captureScreenshot` 返回 `null`
- Props：`symbol`、`timeframe`、`exchange`、`height`（OHLCV 数据通过内部 `useQuery` 获取 `/api/market/{pair}/ohlcv`）
- 叠加 SMA（20/50 期）辅助线，供 AI 识别均线方向

#### `AiAnalysisPanel`（`web/src/pages/market/components/ai-analysis-panel.tsx`）

- Props：`result: VisualAnalysisResult`，`onStop: () => void`，`onRetry: () => void`
- 状态渲染：`idle` 隐藏，`loading` 骨架屏，`streaming` 流式 Markdown，`done` 完整结果，`error` 错误提示
- `role="region"` + `aria-live="polite"`（NFR-005）
- 显示来源标注（FR-009）：图标区分"视觉识别"和"数值数据"

#### `ChartTabPanel`（`web/src/pages/market/components/chart-tab-panel.tsx`）

- 包含 Radix Tabs：`tradingview`（默认）| `candlestick`
- 切换到 `candlestick` 时才 mount `CandlestickChart`（避免双向订阅）
- 传递 `chartRef` 给父组件用于截图

### 5.2 `useChartAnalysis` hook

```typescript
interface UseChartAnalysisReturn {
  result: VisualAnalysisResult;
  triggerFast: (chartRef: RefObject<CandlestickChartHandle>, timeframe: string) => void;
  triggerDeep: (chartRef: RefObject<CandlestickChartHandle>, timeframe: string) => void;
  stop: () => void;
}
```

**Fast 模式流程**：
1. 调用 `chartRef.current.captureScreenshot()` → 若为 `null` 则仅使用文字描述，显示提示
2. 调用 `generateDescription(ohlcvData, indicators, symbol, timeframe)` 生成结构化文字
3. 检查图片大小（前端预检），超过 `MAX_IMAGE_BYTES_FRONTEND = 4.5MB` 则置 `dataUrl = null`
4. 构造 `AdditionalContext`，调用 `streamFetch('/api/chat/stream', { body: { additional_context, message: 'analyze' } })`
5. 更新 `VisualAnalysisResult` 状态，渲染到 `AiAnalysisPanel`

**Deep 模式流程**：
1. 与 Fast 模式步骤 1-3 相同
2. 构造 `additionalContext` 后调用 `navigate('/chat', { state: { additionalContext } })`
3. 不发起 SSE，页面跳转后由 Chat 页面自动触发

**切换交易对/时间周期时重置**：
- MarketView 的 `pair` / `timeframe` state 变化时，调用 `resetResult()` 清空旧分析并标记"已过期"

### 5.3 Chat 页面扩展

`web/src/pages/chat/index.tsx` 新增逻辑：

```typescript
const location = useLocation();
const additionalContext = (location.state as { additionalContext?: AdditionalContext } | null)?.additionalContext;

useEffect(() => {
  if (!additionalContext) return;
  // 500ms 内自动发送，SC-004
  const timer = setTimeout(() => {
    sendMessage('请分析当前图表', additionalContext);
  }, 100);
  return () => clearTimeout(timer);
}, []); // 仅首次
```

图表上下文标记 badge（FR-008）渲染在消息流顶部，显示交易对、时间周期、截图时间。

### 5.4 `sendMessage` 签名扩展

`useChatMessages.sendMessage` 扩展为：

```typescript
sendMessage: (text: string, additionalContext?: AdditionalContext) => void;
```

`streamFetch` 的 body 加入 `additional_context`（当存在时）。现有调用点（Chat 页面、其他 hook）不传第二参数，向后兼容。

---

## 六、结构化文字描述生成（前端）

`web/src/lib/indicators.ts` 提供：

```typescript
interface OHLCVBar { time: number; open: number; high: number; low: number; close: number; volume: number; }
interface IndicatorSnapshot { rsi14: number; macd: { value: number; signal: number; hist: number }; sma20: number; sma50: number; }

function calcIndicators(bars: OHLCVBar[]): IndicatorSnapshot
function generateDescription(bars: OHLCVBar[], indicators: IndicatorSnapshot, symbol: string, timeframe: string, marketData: { fundingRate: number | null; openInterest: number | null }): string
```

描述格式示例（FR-003）：

```
交易对: BTC/USDT | 时间周期: 1h | 截图时间: 2026-04-17T10:00:00Z
最新收盘价: 84,500 USDT | 成交量: 1.32× 20期均值
趋势方向: SMA20(83,100) > SMA50(81,500)，短期上方 → 上升趋势
RSI(14): 62.4 → 中性（接近超买区域）
MACD: MACD线 120 > 信号线 95，柱状图 +25 → 金叉，动能扩张
最近3根K线:
  - [0] 实体 1.8%，上影 0.3%，下影 0.1% → 强阳线
  - [-1] 实体 0.6%，上影 0.9%，下影 0.2% → 上影压力
  - [-2] 实体 0.4%，上影 0.1%，下影 0.8% → 下影支撑
资金费率: +0.0125% | 未平仓量: $4.23B
```

---

## 七、依赖变更

### 前端（`web/package.json`）

**无新增 npm 依赖**。所有功能使用现有依赖实现：
- `lightweight-charts ^4.2.3`：`takeScreenshot()` API
- `@radix-ui/react-tabs ^1.1.13`：图表 Tab 切换
- `react-router ^7.14.1`：路由 state 传递
- `react-markdown ^10.1.0`：分析结果 Markdown 渲染

### 后端（`pyproject.toml`）

**无新增 Python 依赖**。LangChain 已支持多模态消息格式：
- `langchain-openai`：`ChatOpenAI` 支持 Vision API（GPT-4o、Gemini 兼容接口）
- `langchain-anthropic`（若已安装）：Claude Vision

---

## 八、新增后端 API 端点

`GET /api/market/{pair}/ohlcv?timeframe=1h&limit=100`（为 `CandlestickChart` 提供 K 线数据）

- 复用现有 `data/market.py` 的 CCXT 数据获取逻辑
- 返回格式：`{ bars: [{ time, open, high, low, close, volume }] }`
- 此端点为前端渲染 K 线图所需；分析功能依赖此端点数据

---

## 九、风险与缓解

| 风险 | 严重度 | 缓解措施 |
|------|--------|----------|
| `takeScreenshot()` 在图表未就绪时返回空白图 | 高 | `isReady` 标志位（`subscribeVisibleTimeRangeChange`），未就绪时捕获错误降级为文字描述（FR-011） |
| 图片 base64 超过 LLM 输入限制（Claude 5MB） | 中 | 前端预检大小 → 后端 `max_image_bytes` 双重拦截 → 降级文字（FR-013） |
| `useChatMessages.sendMessage` 签名变更破坏现有 Chat 页面 | 中 | 第二参数可选，所有现有调用点无需修改（向后兼容），SC-008 回归测试覆盖 |
| `CandlestickChart` OHLCV 数据获取延迟（新增端点） | 中 | 截图前等待数据就绪；图表未就绪自动降级文字模式 |
| Fast 模式 > 8 秒首 token（SC-001） | 中 | `fast_model` 配置独立的快速模型（如 `gpt-4o-mini-vision`）；截图在前端同步完成（≤ 200ms）；图片过大预检降级减少 payload |
| 多时间周期（P2）数据区段混淆 | 低 | Prompt 中用 `=== 时间周期: 15m ===` 分隔符明确区分（FR-014） |
| Deep 模式路由 state 在页面刷新后丢失 | 低 | 属于预期行为（spec 不要求持久化）；Chat 页面检测 state 为空时正常显示空会话 |

---

## 十、实施阶段规划

| 阶段 | 内容 | 优先级 | 预估工时 |
|------|------|--------|----------|
| **Phase 1** | 基础设施：类型定义、配置、OHLCV 端点 | P0 | 4h |
| **Phase 2** | 后端扩展：`additional_context` 解析 + `context_builder.py` | P0 | 4h |
| **Phase 3** | 前端核心：`CandlestickChart` + 截图 API + `indicators.ts` | P0 | 6h |
| **Phase 4** | Fast 模式：`useChartAnalysis` hook + `AiAnalysisPanel` + MarketView 集成 | P0 | 6h |
| **Phase 5** | Deep 模式：Chat 页面扩展 + `sendMessage` 签名扩展 | P1 | 4h |
| **Phase 6** | P2 多时间周期叠加分析 | P2 | 4h |
| **Phase 7** | 测试、i18n、无障碍、TypeScript strict | 全部 | 4h |

**总计估算**：32 小时（P0: 20h，P1: 4h，P2: 4h，测试/收尾: 4h）
